import boto3
import logging
import argparse
import subprocess
import uuid
from datetime import datetime
import os
import io
import json
import sys

logger = logging.getLogger('root')
logFormatter = logging.Formatter('%(asctime)s %(process)d [%(funcName)s] %(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logFormatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

REQUIREMENTS = ['boto3', 'pandas', 'numpy', 'scikit-learn==0.19.0', 'scipy==1.0.0', 'feather-format', 'xgboost']
PIP_REQUIREMENTS = ['tpot', 'imblearn']

s3 = boto3.resource('s3')


def json_dump_s3(obj, bucket, key):
    buf = io.StringIO()
    json.dump(obj, buf, indent=4)
    s3.Object(bucket, key).put(Body=buf.getvalue())


def valid_file(x):
    if os.path.exists(x):
        return x
    else:
        raise ValueError('Path does not exist: {}'.format(x))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to experiments folder", type=valid_file)
    parser.add_argument("script", help="Name of Python script")
    parser.add_argument("config", help="Path to config file", type=valid_file)
    parser.add_argument("pem", help="Name of the pem file.")
    parser.add_argument("s3", help="S3 bucket name")
    parser.add_argument("--security-group", help="Security group")
    parser.add_argument("--py-files", nargs='+', help="Extra Python files to send to the EC2 instance")
    parser.add_argument("--instance-type", default="m4.large")
    parser.add_argument("--bid-price", help="Max bid price for spot instance. If null, will use on-demand.")
    parser.add_argument("--no-terminate", action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folder = args.folder
    script = os.path.join(folder, args.script)
    config = args.config
    pem_file = args.pem
    key_name = os.path.basename(pem_file).split('.')[0]
    s3_bucket = args.s3
    instance_type = args.instance_type
    bid_price = args.bid_price
    py_files = args.py_files if args.py_files else []
    terminate = not args.no_terminate
    security_group = args.security_group

    ec2 = boto3.client('ec2')

    experiment_id = uuid.uuid4()
    experiment_date = datetime.now()
    logger.warning('Experiment ID: {}'.format(experiment_id))

    logger.warning('Launching instance')
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
            ImageId='ami-3a533040',
            InstanceType=instance_type,
            MaxCount=1,
            MinCount=1,
            KeyName=key_name,
            IamInstanceProfile={
                'Name': 'ec2_role'
            },
            NetworkInterfaces=[{
                'DeviceIndex': 0,
                'AssociatePublicIpAddress': True,
                'Groups': [security_group]
            }],
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
            NetworkInterfaces=[{
                'DeviceIndex': 0,
                'AssociatePublicIpAddress': True,
                'Groups': [security_group]
            }],
            IamInstanceProfile={
                'Name': 'ec2_role'
            },
            InstanceInitiatedShutdownBehavior='terminate',
            TagSpecifications=tags
        )

    instance_id = create['Instances'][0]['InstanceId']
    logger.warning('Instance ID: {}'.format(instance_id))

    waiter = ec2.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=[instance_id])

    instance = ec2.describe_instances(InstanceIds=[instance_id])
    public_dns = instance['Reservations'][0]['Instances'][0]['PublicDnsName']
    logger.warning('To view output:  ssh -i {pem} ec2-user@{host} "tail -100f log.txt"'.format(pem=pem_file, host=public_dns))

    jupyter_command = 'ssh -i {pem} -L 8000:localhost:8888 ec2-user@{host}'.format(pem=pem_file, host=public_dns)
    logger.warning('Jupyter tunnel command: {}'.format(jupyter_command))
    logger.warning('Jupyter notebook command: jupyter notebook --no-browser --port=8888')

    exp_data = {'experiment_id': '{}'.format(experiment_id),
                'experiment_date': '{}'.format(experiment_date),
                'instance_id': '{}'.format(instance_id),
                'instance_type': '{}'.format(instance_type),
                'public_dns': '{}'.format(public_dns),
                'terminate': 'aws ec2 terminate-instances --instance-ids {}'.format(instance_id),
                'tail_log': 'ssh -i {pem} ec2-user@{host} "tail -100f log.txt"'.format(pem=pem_file, host=public_dns)
                }
    json_dump_s3(exp_data, s3_bucket, 'experiments/json/{}_{}.json'.format(experiment_date.strftime('%Y-%m-%d_%H:%M:%S'), experiment_id))

    commands = """
    source activate tensorflow_p36

    instanceid={instance_id}

    echo 'Installing dependencies'
    sudo yum -y install htop
    conda install {dep} -c conda-forge
    pip install {pip_dep}

    echo 'Running script'
    python {script} {config} --experiment-id {exp_id} --bucket {bucket}

    echo 'Save logs'
    logdate=$(date '+%Y-%m-%d_%H:%M:%S_')
    logfile="$logdate$instanceid"
    aws s3 cp log.txt "s3://{bucket}/logs/$logfile.txt"
    aws s3 cp resources.csv "s3://{bucket}/experiments/logs/$logfile-resources.csv"
    """.format(instance_id=instance_id,
               dep=' '.join(REQUIREMENTS),
               pip_dep=' '.join(PIP_REQUIREMENTS),
               script=os.path.basename(script),
               config=os.path.basename(config).split('.')[0],
               exp_id=experiment_id,
               bucket=s3_bucket)
    terminate_command = """
    # Terminate instance
    sudo shutdown -h now
    """
    if terminate:
        commands += terminate_command

    command_script = 'command-{}.sh'.format(experiment_id)
    command_path = os.path.join('/tmp', command_script)
    with open(command_path, "w") as text_file:
        text_file.write(commands)

    log_resources = """
    #!/usr/bin/env bash

    echo 'datetime,mem_used,mem_free,disk_used,disk_free,cpu'
    while true
    do
        sleep 1
        echo "$(date '+%Y-%m-%d %H:%M:%S'),$(free -m | awk 'NR==2{printf \'%s,%s\n\', $3,$4 }'),$(df / -h | grep -v Filesystem | awk -F' ' '{printf \'s,%s\n\', $3,$4}'),\"$(uptime)\""
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
    for s in [folder, script, config, sys.argv[0], command_path, '/tmp/log_resources.sh', '/tmp/tensorboard.sh'] + py_files:
        ssh_command = "scp -r -i {} -o StrictHostKeyChecking=no {} ec2-user@{}:/home/ec2-user".format(pem_file, s, public_dns)
        subprocess.check_call(ssh_command, shell=True)

    # launch experiment
    ssh_command = 'ssh -i {pem} -o StrictHostKeyChecking=no ec2-user@{host} "nohup bash {script} &> log.txt &"'.format(pem=pem_file, host=public_dns,
                                                                                                                       script=command_script)
    subprocess.Popen(ssh_command, shell=True)

    # launch resource logging
    ssh_command = 'ssh -i {pem} -o StrictHostKeyChecking=no ec2-user@{host} "nohup bash log_resources.sh &> resources.csv &"'.format(pem=pem_file,
                                                                                                                                     host=public_dns)
    subprocess.Popen(ssh_command, shell=True)

    # launch tensorboard
    ssh_command = 'ssh -i {pem} -o StrictHostKeyChecking=no ec2-user@{host} "nohup bash tensorboard.sh &> tensorboard_log.txt &"'.format(pem=pem_file,
                                                                                                                                         host=public_dns)
    subprocess.Popen(ssh_command, shell=True)
