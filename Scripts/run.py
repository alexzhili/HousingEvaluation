import os
import time


os.system('python ./upload_data.py')
os.system('python ./pyspark_ml_training.py')
os.system('python ./pysparkling_automl_training.py')