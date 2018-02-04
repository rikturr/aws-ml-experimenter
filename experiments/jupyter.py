import boto3
import logging
import argparse
import subprocess
import os
import uuid
from datetime import datetime
from experiments.helpers import json_dump_s3

logger = logging.getLogger('root')
logFormatter = logging.Formatter('%(asctime)s %(process)d [%(funcName)s] %(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logFormatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

REQUIREMENTS = ['boto3', 'scikit-learn==0.19.0', 'scipy==1.0.0', 'feather-format']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to Python scripts")
    parser.add_argument("script", help="Name of the script to execute (not including .py)")
    parser.add_argument("config", help="Name of the config module (not including .py)")
    parser.add_argument("pem", help="Name of the pem file.")
    parser.add_argument("s3", help="S3 bucket name")
    parser.add_argument("--instance-type", default="m4.large")
    parser.add_argument("--bid-price", help="Max bid price for spot instance. If null, will use on-demand.")

    return parser.parse_args()


args = parse_args()
path = args.path
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

echo 'Running script'
python {script}.py {config} --experiment-id {exp_id} --bucket {bucket}

echo 'Save logs'
logdate=$(date '+%Y-%m-%d_%H:%M:%S_')
logfile="$logdate$instanceid"
aws s3 cp log.txt "s3://{bucket}/logs/$logfile.txt"
aws s3 cp resources.csv "s3://{bucket}/logs/$logfile-resources.csv"

# Terminate instance
sudo shutdown -h now
""".format(instance_id=instance_id,
           dep=' '.join(REQUIREMENTS),
           path=path,
           script=script,
           config=config,
           exp_id=experiment_id,
           bucket=s3_bucket)
with open("/tmp/command.sh", "w") as text_file:
    text_file.write(commands)

# copy scripts to instance
ssh_command = "scp -r -i {} -o StrictHostKeyChecking=no {}/* ec2-user@{}:/home/ec2-user".format(pem_file, path, public_dns)
subprocess.check_call(ssh_command, shell=True)
ssh_command = "scp -r -i {} -o StrictHostKeyChecking=no /tmp/command.sh ec2-user@{}:/home/ec2-user".format(pem_file, public_dns)
subprocess.check_call(ssh_command, shell=True)

# launch experiment
ssh_command = 'ssh -i {pem} -o StrictHostKeyChecking=no ec2-user@{host} "nohup bash command.sh &> log.txt &"'.format(pem=pem_file, host=public_dns)
subprocess.Popen(ssh_command, shell=True)

# launch resource logging
ssh_command = 'ssh -i {pem} -o StrictHostKeyChecking=no ec2-user@{host} "nohup bash log_resources.sh &> resources.csv &"'.format(pem=pem_file, host=public_dns)
subprocess.Popen(ssh_command, shell=True)
