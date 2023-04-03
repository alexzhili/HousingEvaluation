from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.functions import regexp_replace, trim, col, lower, length
import nltk
nltk.download('stopwords')


def clean_text(dataframe):
    dataframe = dataframe.select(trim(regexp_replace(col('text'),'http\S+',' ')).alias("text")) # remove urls
    dataframe = dataframe.select(trim(regexp_replace(col('text'),'\\p{Punct}',' ')).alias("text"))  #remove punct
    dataframe = dataframe.select(trim(regexp_replace(col('text'),'[0-9]',' ')).alias("text")) #remove numbers
    dataframe = dataframe.select(trim(regexp_replace(col('text'),'[^a-zA-Z]',' ')).alias("text")) # remove all non-alpha numeric characters
    dataframe = dataframe.select(lower(col('text')).alias("text")) # change to lower case
    return dataframe


def get_stopwords():
    from nltk.corpus import stopwords

    defined_stopwords = ['echt','haha','gaan','gaat','even']
    words_nl = stopwords.words('dutch')
    words_en = stopwords.words('english')
    words_es = stopwords.words('spanish')
    words = words_nl + words_en + words_es + defined_stopwords

    return words


def SparkTrends(fileName, spark_master, mongo_uri):

    mongo_read = mongo_uri + "/Twitter." + fileName

    import socket

    driver_host = socket.gethostbyname(socket.gethostname()) # setting driver host is important in k8s mode, ortherwise excutors cannot find diver host

    spark = SparkSession \
        .builder \
        .master(spark_master)\
        .appName("TwitterTrendingApp") \
        .config("spark.driver.host", driver_host) \
        .config("spark.mongodb.input.uri", mongo_read) \
        .config("spark.mongodb.output.uri", mongo_read) \
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1') \
        .getOrCreate()

    sample_df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()



    # extract ['created_at','text'] columns from original dataframe for further analysis
    if fileName.lower() == 'sample':
        text_df = sample_df.select("text").na.drop()
        text_df = clean_text(text_df)
    elif fileName.lower() == 'twitter-sample':
        text_df = sample_df.select( col("interaction.content").alias("text")).na.drop()
        text_df = clean_text(text_df)
    else:
        print('file name not supported!')



    # do word frequency count on the text field
    # use simplest stop word ' '
    trending = text_df.withColumn('word', f.explode(f.split(f.col('text'), ' ')))\
        .groupBy('word')\
        .count()\
        .sort('count', ascending=False)

    words = get_stopwords()
    trending = trending.filter(~trending.word.isin(words))\
                       .filter(length(trending.word) > 3)    # length of words > 3 and word is not in stopword list
        
    # write results into a new collection in mongodb
    mongo_write = mongo_uri + "/Twitter.trends-" + fileName
    trending.select("word","count").write\
        .format('com.mongodb.spark.sql.DefaultSource')\
        .mode("overwrite")\
        .option( "uri", mongo_write) \
        .save()

    spark.stop()


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description = 'Load json files into MongoDB.')
    parser.add_argument('--fileName', type = str, default='sample', help = 'The name of the file to load.')
    args = parser.parse_args()
    
    # fileName = 'sample' or "twitter-sample"
    fileName = args.fileName.replace('.json','').lower()

    # can be replaced by os.
    import os
    spark_master = os.environ.get('SPARK_MASTER') # "spark://spark-master:7077" 
    mongo_uri = "mongodb://mongo:27017"  or os.environ.get('MONGO_HOST')
    
    SparkTrends(fileName, spark_master, mongo_uri)

