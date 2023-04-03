from pyspark.sql import SparkSession
import socket
import os

import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

from hyperopt import fmin, tpe, Trials, hp
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


def get_sparkSession(appName = 'MLOps'):
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



def run_LinearRegression(imputed_df, labelCol="price"):
    
    train_df, test_df = imputed_df.randomSplit([.8, .2] , seed=42)

    with mlflow.start_run(run_name="LinearRegression") as run:
        
        # Define pipeline
        categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
        index_output_cols = [x + "Index" for x in categorical_cols]
        ohe_output_cols = [x + "OHE" for x in categorical_cols]
        string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")
        ohe_encoder = OneHotEncoder(inputCols=index_output_cols, outputCols=ohe_output_cols)
        numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != labelCol))]
        assembler_inputs = ohe_output_cols + numeric_cols
        vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
        
        lr = LinearRegression(labelCol=labelCol, featuresCol="features")
        
        stages = [string_indexer, ohe_encoder, vec_assembler, lr]

        pipeline = Pipeline(stages=stages)
        pipeline_model = pipeline.fit(train_df)
        
        # Log parameters
        # mlflow.log_param("label", labelCol)
        # mlflow.log_param("features", "multiple")

        # Evaluate predictions
        pred_df = pipeline_model.transform(test_df)
        regression_evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction")
        rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
        r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)

        # Log both metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas()) 




def run_RandomForestCV(imputed_df, maxBins=40, labelCol="price"):

    train_df, test_df = imputed_df.randomSplit([.8, .2] , seed=42)

    with mlflow.start_run(run_name="RF-GridSearchCV") as run:
        
        categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
        index_output_cols = [x + "Index" for x in categorical_cols]

        string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

        numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != labelCol))]
        assembler_inputs = index_output_cols + numeric_cols
        vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

        rf = RandomForestRegressor(labelCol=labelCol, maxBins=maxBins)
        
        param_grid = (ParamGridBuilder()
                    .addGrid(rf.maxDepth, [2, 5])
                    .addGrid(rf.numTrees, [5, 10])
                    .build())

        evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction")

        # Pipeline in CV: take much longer time if there are estimators in pipeline, which have to be refitted in every validation
        
    #     stages = [string_indexer, vec_assembler, rf]
    #     pipeline = Pipeline(stages=stages)
    #     cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=param_grid, 
    #                         numFolds=3, , parallelism=4, seed=42)
    #     cv_model = cv.fit(train_df)

        # CV in pipeline: potential risk of data leakage
        cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=param_grid, 
                        numFolds=10, parallelism=4, seed=42)
        stages_with_cv = [string_indexer, vec_assembler, cv]
        pipeline = Pipeline(stages=stages_with_cv)
        pipeline_model = pipeline.fit(train_df)

        # Log parameter
        # mlflow.log_param("label", "price")
        # mlflow.log_param("features", "all_features")

        # Create predictions and metrics
        best_model = pipeline_model.stages[-1].bestModel
        best_pipeline_model = Pipeline(stages=[string_indexer, vec_assembler, best_model]).fit(train_df)
        pred_df = best_pipeline_model.transform(test_df)
        rmse = evaluator.setMetricName("rmse").evaluate(pred_df)
        r2 = evaluator.setMetricName("r2").evaluate(pred_df)

        # Log both metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        mlflow.spark.log_model(best_pipeline_model, "model", input_example=train_df.limit(5).toPandas())
        
        # Log feature_importance
        features_df = pd.DataFrame(list(zip(vec_assembler.getInputCols(), best_model.featureImportances)), columns=["feature", "importance"])
        features_df = features_df.sort_values(by = 'importance',ascending=False).head(10)
        fig, ax = plt.subplots()
        features_df.plot(kind='barh', x='feature', y='importance', ax=ax)
        mlflow.log_figure(fig, "feature_importance.png")



def run_RandomForest_Hyperopt(imputed_df, maxBins=40, labelCol="price"):

    train_df, val_df, test_df = imputed_df.randomSplit([.6, .2, .2], seed=42)

    categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
    index_output_cols = [x + "Index" for x in categorical_cols]
    string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")
    numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != labelCol))]
    assembler_inputs = index_output_cols + numeric_cols
    vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    rf = RandomForestRegressor(labelCol=labelCol, maxBins=maxBins)
    pipeline = Pipeline(stages=[string_indexer, vec_assembler, rf])
    evaluator = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction")


    def objective_function(params):    
        # set the hyperparameters that we want to tune
        max_depth = params["max_depth"]
        num_trees = params["num_trees"]
        with mlflow.start_run():
            estimator = pipeline.copy({rf.maxDepth: max_depth, rf.numTrees: num_trees})
            model = estimator.fit(train_df)
            preds = model.transform(val_df)
            rmse = evaluator.evaluate(preds)
            #mlflow.log_metric("rmse_val", rmse)
        return rmse


    search_space = {
        "max_depth": hp.quniform("max_depth", 2, 5, 1),
        "num_trees": hp.quniform("num_trees", 10, 100, 1)
    }

    num_evals = 4
    trials = Trials()
    best_hyperparam = fmin(fn=objective_function, 
                        space=search_space,
                        algo=tpe.suggest, 
                        max_evals=num_evals,
                        trials=trials,
                        rstate=np.random.default_rng(42))


    with mlflow.start_run(run_name="RF-Hyperopt") as run:
        best_max_depth = best_hyperparam["max_depth"]
        best_num_trees = best_hyperparam["num_trees"]
        estimator = pipeline.copy({rf.maxDepth: best_max_depth, rf.numTrees: best_num_trees})
        combined_df = train_df.union(val_df) # Combine train & validation together

        pipeline_model = estimator.fit(combined_df)
        pred_df = pipeline_model.transform(test_df)
  
        rmse = evaluator.setMetricName("rmse").evaluate(pred_df)
        r2 = evaluator.setMetricName("r2").evaluate(pred_df)

        # Log param and metrics for the final model
        # mlflow.log_param("maxDepth", best_max_depth)
        # mlflow.log_param("numTrees", best_num_trees)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.spark.log_model(pipeline_model, "model", input_example=combined_df.limit(5).toPandas())
        
        best_model = pipeline_model.stages[-1]
        features_df = pd.DataFrame(list(zip(vec_assembler.getInputCols(), best_model.featureImportances)), columns=["feature", "importance"])
        features_df = features_df.sort_values(by = 'importance',ascending=False).head(10)
        fig, ax = plt.subplots()
        features_df.plot(kind='barh', x='feature', y='importance', ax=ax)
        mlflow.log_figure(fig, "feature_importance.png")




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

    # keep_cols = [
    #     "host_is_superhost",
    #     "cancellation_policy",
    #     "instant_bookable",
    #     "host_total_listings_count",
    #     "neighbourhood_cleansed",
    #     "latitude",
    #     "longitude",
    #     "property_type",
    #     "room_type",
    #     "accommodates",
    #     "bathrooms",
    #     "bedrooms",
    #     "beds",
    #     "bed_type",
    #     "review_scores_rating",
    #     "review_scores_accuracy",
    #     "review_scores_cleanliness",
    #     "review_scores_checkin",
    #     "review_scores_communication",
    #     "review_scores_location",
    #     "review_scores_value",
    #     "price"
    # ]



    # impute_cols = [
    #     "bedrooms",
    #     "bathrooms",
    #     "beds", 
    #     "review_scores_rating",
    #     "review_scores_accuracy",
    #     "review_scores_cleanliness",
    #     "review_scores_checkin",
    #     "review_scores_communication",
    #     "review_scores_location",
    #     "review_scores_value"
    # ]


    spark = get_sparkSession(appName = 'MLOps')
    imputed_df = clean_impute_dataframe(spark, file_uri, keep_cols, impute_cols, impute_strategy = "median")
    


    mlflow.set_experiment('MLOps_Experiment')
    mlflow.pyspark.ml.autolog(log_models=False)
    run_LinearRegression(imputed_df, labelCol="price")
    run_RandomForestCV(imputed_df, maxBins=40, labelCol="price")
    run_RandomForest_Hyperopt(imputed_df, maxBins=40, labelCol="price")