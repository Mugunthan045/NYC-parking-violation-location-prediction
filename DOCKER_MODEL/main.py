import os
import sys
from pyspark.context import SparkContext
from pyspark.sql import SparkSession



if os.path.exists('src.zip'):
    sys.path.insert(0,'src.zip')
else:
    sys.path.insert(0,'./src')

from utilities import processing as proc
from utilities import prediction as pred

CSV_PATH = '/opt/application/parking.csv'

if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName('Parking') \
        .getOrCreate()

    df =  spark.read.option('header','true') \
    .option('inferSchema','true') \
    .csv(CSV_PATH)


    df = proc.splitting_columns(df)
    df = proc.target_column_processing(df)
    df = proc.cleaning_values(df)
    df = proc.dropping_columns(df)
    df = proc.dropping_null_rows(df)
    df = pred.indexing(df)


    #spark
    prediction = pred.prediction(df)

    pred.accuracy(prediction)

    

