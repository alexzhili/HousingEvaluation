import os
import boto3

ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
MLFLOW_S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

session = boto3.Session(
aws_access_key_id=ACCESS_KEY,
aws_secret_access_key=SECRET_KEY
)

#Creating S3 Resource From the Session.
s3 = session.resource('s3',endpoint_url=MLFLOW_S3_ENDPOINT_URL,config=boto3.session.Config(signature_version='s3v4'))

try:
    s3.Object('sf-listing', 'sf-listings.csv').load()
except:
    s3.Bucket('sf-listing').upload_file('/home/sf-listings.csv','sf-listings.csv')

try:
    s3.Object('funda', 'woonhuis_beschikbaar.csv').load()
except:
    s3.Bucket('funda').upload_file('/home/woonhuis_beschikbaar.csv','woonhuis_beschikbaar.csv')