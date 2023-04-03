from pysparkling.ml import H2OAutoML
from pysparkling import *

from pyspark.sql import SparkSession
import socket
import os

import mlflow
import mlflow.spark

mlflow.set_experiment('MLOps_Experiment')
#mlflow.pyspark.ml.autolog(log_models=False)


def get_sparkSession(appName = 'H2OautoML'):
    spark_master = os.environ.get('SPARK_MASTER') # "spark://spark-master:7077" 
    driver_host = socket.gethostbyname(socket.gethostname()) # setting driver host is important in k8s mode, ortherwise excutors cannot find diver host

    spark = SparkSession \
        .builder \
        .master(spark_master)\
        .appName(appName) \
        .config("spark.driver.host", driver_host) \
        .config('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:3.3.1') \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
            
    ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
    SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    MLFLOW_S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

    hadoopConf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoopConf.set('fs.s3a.access.key', ACCESS_KEY)
    hadoopConf.set('fs.s3a.secret.key', SECRET_KEY)
    hadoopConf.set("fs.s3a.endpoint", MLFLOW_S3_ENDPOINT_URL)
    hadoopConf.set('fs.s3.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem')
    hadoopConf.set("fs.s3a.connection.ssl.enabled", "true")
    hadoopConf.set("fs.s3a.path.style.access", 'true')

    return spark



def clean_impute_dataframe(spark, file_uri, keep_cols, impute_cols, impute_strategy = "median"):
    
    raw_df = spark.read.csv(file_uri ,header="true", inferSchema="true", multiLine="true", escape='"')
    base_df = raw_df.select(*keep_cols)

    from pyspark.sql.functions import col, translate, when
    from pyspark.sql.types import IntegerType

    #cast datatypes into doubles & simply remove outliers with price beyond normal ranges
    doubles_df= base_df.withColumn("price", translate(col("price"), "$,", "").cast("double")) \
                            .filter(col("price") > 0)

    integer_columns = [x.name for x in doubles_df.schema.fields if x.dataType == IntegerType()]

    for c in integer_columns:
        doubles_df = doubles_df.withColumn(c, col(c).cast("double"))

    for c in impute_cols:
        doubles_df = doubles_df.withColumn(c + "_na", when(col(c).isNull(), 1.0).otherwise(0.0))    

    from pyspark.ml.feature import Imputer
    imputer = Imputer(strategy=impute_strategy, inputCols=impute_cols, outputCols=impute_cols)
    imputer_model = imputer.fit(doubles_df)
    imputed_df = imputer_model.transform(doubles_df)

    return imputed_df




if __name__ == "__main__":

    file_uri = "s3://funda/woonhuis_beschikbaar.csv"


    keep_cols = [
        "address",
        "zip_code",
        "city",
        "price",
        "living_area",
        "plot_size",
        "rooms"
    ]

    impute_cols = [
        "living_area",
        "plot_size",
        "rooms"
    ]

    spark = get_sparkSession(appName = 'H2OautoML')
    imputed_df = clean_impute_dataframe(spark, file_uri, keep_cols, impute_cols, impute_strategy = "median")
    train_df, test_df = imputed_df.randomSplit([.8, .2] , seed=42)

    hc = H2OContext.getOrCreate()

    with mlflow.start_run(run_name="H2O-autoML") as run:
        
        automl = H2OAutoML(labelCol="price", convertUnknownCategoricalLevelsToNa=True)
        automl.setExcludeAlgos(["GLM","DeepLearning"])
        automl.setMaxModels(10)
        automl.setSortMetric("rmse")

        model = automl.fit(train_df)
        from pyspark.ml.evaluation import RegressionEvaluator

        pred_df = model.transform(test_df)
        regression_evaluator = RegressionEvaluator(labelCol='price', predictionCol="prediction")
        rmse = regression_evaluator.evaluate(pred_df)
        r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)


        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.spark.log_model(model, 'model')

    #spark.stop()
    