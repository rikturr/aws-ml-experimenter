import boto3
import logging
import argparse
import os

logger = logging.getLogger('root')
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter('%(asctime)s %(process)d [%(funcName)s] %(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logFormatter)
logger.addHandler(console_handler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pem", help="Name of the pem file.")
    parser.add_argument("--port", help="Local port to map jupyter notebook to", default='8888')
    parser.add_argument("--instance-type", default="m4.large")
    parser.add_argument("--bid-price", help="Max bid price for spot instance. If null, will use on-demand.")

    return parser.parse_args()


args = parse_args()
pem_file = args.pem
key_name = os.path.basename(pem_file).split('.')[0]
port = args.port
instance_type = args.instance_type
bid_price = args.bid_price

ec2 = boto3.client('ec2')


logger.info('Launching instance')
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
        InstanceInitiatedShutdownBehavior='terminate',
        IamInstanceProfile={
            'Name': 'ec2_role'
        },
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
        InstanceInitiatedShutdownBehavior='terminate',
        IamInstanceProfile={
            'Name': 'ec2_role'
        },
        TagSpecifications=tags
    )

instance_id = create['Instances'][0]['InstanceId']
logger.info('Instance ID: {}'.format(instance_id))

waiter = ec2.get_waiter('instance_status_ok')
waiter.wait(InstanceIds=[instance_id])

instance = ec2.describe_instances(InstanceIds=[instance_id])
public_dns = instance['Reservations'][0]['Instances'][0]['PublicDnsName']
logger.info('Public DNS name: {}'.format(public_dns))

jupyter_command = 'ssh -i {pem} -L 8000:localhost:{port} ec2-user@{host}'.format(pem=pem_file, port=port, host=public_dns)
logger.info('SSH command: {}'.format(jupyter_command))
logger.info('Jupyter commands: jupyter notebook --no-browser --port=8888')
logger.info("Don't forget to copy notebooks and terminate instance when you're done!!")
