import boto3
import logging
import argparse
import subprocess
import uuid
from datetime import datetime
import io
import numpy as np
import pandas as pd
import os
import keras
import time
import json
import sys

logger = logging.getLogger('root')
logFormatter = logging.Formatter('%(asctime)s %(process)d [%(funcName)s] %(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logFormatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

REQUIREMENTS = ['boto3', 'scikit-learn==0.19.0', 'scipy==1.0.0']
PIP_REQUIREMENTS = ['tpot']


s3 = boto3.resource('s3')


# HELPER FUNCTIONS FOR USE IN EXPERIMENT SCRIPTS

def sparse_generator(x, y=None, batch_size=32):
    index = np.arange(x.shape[0])
    start = 0
    while True:
        if start == 0 and y is not None:
            np.random.shuffle(index)

        batch = index[start:start + batch_size]

        if y is not None:
            yield x[batch].toarray(), y[batch]
        else:
            yield x[batch].toarray()

        start += batch_size
        if start >= x.shape[0]:
            start = 0


class S3Checkpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, s3_resource, bucket, s3_folder,
                 monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(S3Checkpoint, self).__init__(filepath, monitor=monitor, verbose=verbose,
                                           save_best_only=save_best_only, save_weights_only=save_weights_only,
                                           mode=mode, period=period)
        self.s3_resource = s3_resource
        self.bucket = s3_resource.Bucket(bucket)
        self.s3_folder = s3_folder

    def on_epoch_end(self, epoch, logs=None):
        super(S3Checkpoint, self).on_epoch_end(epoch, logs)
        if self.epochs_since_last_save == 0:
            local_filepath = self.filepath.format(epoch=epoch + 1, **logs)
            self.bucket.upload_file(local_filepath, os.path.join(self.s3_folder, os.path.basename(local_filepath)))


class S3HistoryLogger(keras.callbacks.Callback):
    def __init__(self, s3_resource, bucket, model_id, history_folder):
        super(S3HistoryLogger, self).__init__()
        self.s3_resource = s3_resource
        self.bucket = bucket
        self.model_id = model_id
        self.history_folder = history_folder

    def to_csv_s3(self, df, key, index=False):
        buf = io.StringIO()
        df.to_csv(buf, index=index)
        self.s3_resource.Object(self.bucket, key).put(Body=buf.getvalue())

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}
        self.time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.epoch_start_time

        logs = logs or {}
        self.epoch.append(epoch)

        # get history - see keras.callbacks.History
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        # add extra stuff
        self.history.setdefault('epoch', []).append(epoch)
        self.history.setdefault('elapsed_time', []).append(elapsed_time)
        self.history.setdefault('model_id', []).append(self.model_id)

        # save to s3
        self.to_csv_s3(pd.DataFrame(self.history), os.path.join(self.history_folder, '{}.csv'.format(self.model_id)))


def get_s3(bucket, key):
    obj = s3.Object(bucket, key)
    return io.BytesIO(obj.get()['Body'].read())


def delete_recursive_s3(bucket, key):
    objects_to_delete = s3.meta.client.list_objects(Bucket=bucket, Prefix=key)

    delete_keys = {'Objects': [{'Key': k} for k in [obj['Key'] for obj in objects_to_delete.get('Contents', [])]]}

    if delete_keys['Objects']:
        s3.meta.client.delete_objects(Bucket=bucket, Delete=delete_keys)


def to_csv_s3(df, bucket, key, index=False):
    buf = io.StringIO()
    df.to_csv(buf, index=index)
    s3.Object(bucket, key).put(Body=buf.getvalue())


def json_dump_s3(obj, bucket, key):
    buf = io.StringIO()
    json.dump(obj, buf, indent=4)
    s3.Object(bucket, key).put(Body=buf.getvalue())


def copy_dir_s3(path, bucket, key):
    for f in os.listdir(path):
        s3.Bucket(bucket).upload_file(os.path.join(path, f), os.path.join(key, f))


def copy_s3(path, bucket, key):
    s3.Bucket(bucket).upload_file(path, key)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help="Path to Python script")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("pem", help="Name of the pem file.")
    parser.add_argument("s3", help="S3 bucket name")
    parser.add_argument("--instance-type", default="m4.large")
    parser.add_argument("--bid-price", help="Max bid price for spot instance. If null, will use on-demand.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script = args.script
    config = args.config
    pem_file = args.pem
    key_name = os.path.basename(pem_file).split('.')[0]
    s3_bucket = args.s3
    instance_type = args.instance_type
    bid_price = args.bid_price

    ec2 = boto3.client('ec2')

    experiment_id = uuid.uuid4()
    experiment_date = datetime.now()
    logger.warn('Experiment ID: {}'.format(experiment_id))

    logger.warn('Launching instance')
    tags = [{
        'ResourceType': 'instance',
        'Tags': [
            {
                'Key': 'app',
                'Value': 'machine_learning'
            }
        ]
    }]
    if bid_price:
        create = ec2.run_instances(
            ImageId='ami-5c9aa926',
            InstanceType=instance_type,
            MaxCount=1,
            MinCount=1,
            KeyName=key_name,
            IamInstanceProfile={
                'Name': 'ec2_role'
            },
            InstanceInitiatedShutdownBehavior='terminate',
            InstanceMarketOptions={
                'MarketType': 'spot',
                'SpotOptions': {
                    'MaxPrice': bid_price,
                    'InstanceInterruptionBehavior': 'terminate'
                }
            },
            TagSpecifications=tags
        )
    else:
        create = ec2.run_instances(
            ImageId='ami-5c9aa926',
            InstanceType=instance_type,
            MaxCount=1,
            MinCount=1,
            KeyName=key_name,
            IamInstanceProfile={
                'Name': 'ec2_role'
            },
            InstanceInitiatedShutdownBehavior='terminate',
            TagSpecifications=tags
        )

    instance_id = create['Instances'][0]['InstanceId']
    logger.warn('Instance ID: {}'.format(instance_id))

    waiter = ec2.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=[instance_id])

    instance = ec2.describe_instances(InstanceIds=[instance_id])
    public_dns = instance['Reservations'][0]['Instances'][0]['PublicDnsName']
    logger.warn('To view output:  ssh -i {pem} ec2-user@{host} "tail -100f log.txt"'.format(pem=pem_file, host=public_dns))

    exp_data = {'experiment_id': '{}'.format(experiment_id),
                'experiment_date': '{}'.format(experiment_date),
                'instance_id': '{}'.format(instance_id),
                'instance_type': '{}'.format(instance_type),
                'public_dns': '{}'.format(public_dns),
                'terminate': 'aws ec2 terminate-instances --instance-ids {}'.format(instance_id),
                'tail_log': 'ssh -i {pem} ec2-user@{host} "tail -100f log.txt"'.format(pem=pem_file, host=public_dns)
                }
    json_dump_s3(exp_data, s3_bucket, 'experiments/{}_{}.json'.format(experiment_date.strftime('%Y-%m-%d_%H:%M:%S'), experiment_id))

    commands = """
    source activate tensorflow_p36

    instanceid={instance_id}

    echo 'Installing dependencies'
    conda install {dep} -c conda-forge
    pip install {pip_dep}

    echo 'Running script'
    python {script} {config} --experiment-id {exp_id} --bucket {bucket}

    echo 'Save logs'
    logdate=$(date '+%Y-%m-%d_%H:%M:%S_')
    logfile="$logdate$instanceid"
    aws s3 cp log.txt "s3://{bucket}/logs/$logfile.txt"
    aws s3 cp resources.csv "s3://{bucket}/logs/$logfile-resources.csv"

    # Terminate instance
    sudo shutdown -h now
    """.format(instance_id=instance_id,
               dep=' '.join(REQUIREMENTS),
               pip_dep=' '.join(PIP_REQUIREMENTS),
               script=os.path.basename(script),
               config=os.path.basename(config).split('.')[0],
               exp_id=experiment_id,
               bucket=s3_bucket)
    with open("/tmp/command.sh", "w") as text_file:
        text_file.write(commands)

    log_resources = """
    #!/usr/bin/env bash

    echo 'datetime,mem_used,mem_free,disk_used,disk_free,cpu'
    while true
    do
        sleep 1
        echo "$(date '+%Y-%m-%d %H:%M:%S'),$(free -m | awk 'NR==2{printf "%s,%s\n", $3,$4 }'),$(df / -h | grep -v Filesystem | awk -F' ' '{printf "%s,%s\n", $3,$4}'),\"$(uptime)\""
    done
    """
    with open("/tmp/log_resources.sh", "w") as text_file:
        text_file.write(log_resources)

    tensorboard = """
    #!/usr/bin/env bash

    source activate tensorflow_p36
    tensorboard --logdir=/tmp/tensorboard --host=0.0.0.0
    """
    with open("/tmp/tensorboard.sh", "w") as text_file:
        text_file.write(tensorboard)

    # copy scripts to instance (sys.argv[0] is this script "run_experiment.py")
    for s in [script, config, sys.argv[0], '/tmp/command.sh', '/tmp/log_resources.sh', '/tmp/tensorboard.sh']:
        ssh_command = "scp -r -i {} -o StrictHostKeyChecking=no {} ec2-user@{}:/home/ec2-user".format(pem_file, s, public_dns)
        subprocess.check_call(ssh_command, shell=True)

    # launch experiment
    ssh_command = 'ssh -i {pem} -o StrictHostKeyChecking=no ec2-user@{host} "nohup bash command.sh &> log.txt &"'.format(pem=pem_file, host=public_dns)
    subprocess.Popen(ssh_command, shell=True)

    # launch resource logging
    ssh_command = 'ssh -i {pem} -o StrictHostKeyChecking=no ec2-user@{host} "nohup bash log_resources.sh &> resources.csv &"'.format(pem=pem_file, host=public_dns)
    subprocess.Popen(ssh_command, shell=True)

    # launch tensorboard
    ssh_command = 'ssh -i {pem} -o StrictHostKeyChecking=no ec2-user@{host} "nohup bash tensorboard.sh &> tensorboard_log.txt &"'.format(pem=pem_file,                                                                                                                           host=public_dns)
    subprocess.Popen(ssh_command, shell=True)
