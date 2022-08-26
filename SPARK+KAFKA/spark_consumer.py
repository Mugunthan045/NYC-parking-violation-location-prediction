
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import from_json,col
from pyspark.sql.types import StringType, StructType, DoubleType
from pyspark.sql.functions import *
from pyspark.ml.classification import RandomForestClassificationModel


KAFKA_TOPIC = "kafka-spark"
KAFKA_SERVER = "localhost:9092"
MODEL_PATH = 'D:\project\MODELS\RFmodel'
MODEL = RandomForestClassificationModel

if __name__ =='__main__':


    spark = SparkSession.builder \
            .appName('Spark Streaming') \
            .getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    kaf_df = spark.readStream.format("kafka")\
        .option("kafka.bootstrap.servers", KAFKA_SERVER)\
        .option("subscribe", KAFKA_TOPIC)\
        .option("startingOffsets", "latest")\
        .load()

    kaf_df1 = kaf_df.selectExpr('CAST(value AS STRING)')
    
    model = MODEL.load(MODEL_PATH)

    schema = StructType()\
        .add('Registration_State_index',StringType())\
        .add('Plate_Type_index',StringType())\
        .add('Violation_Code_index', StringType())\
        .add('Vehicle_Body_Type_index',StringType())\
        .add('Vehicle_Make_index',StringType())\
        .add('Issuing_Agency_index',StringType())\
        .add('Street_Code1_index',StringType())\
        .add('Street_Code2_index',StringType())\
        .add('Street_Code3_index',StringType())\
        .add('Violation_Location_index',StringType())\
        .add('Issuer_Precinct_index',StringType())\
        .add('Issuer_Command_index',StringType())\
        .add('Violation_County_index',StringType())\
        .add('Violation_In_Front_Of_Or_Opposite_index',StringType())\
        .add('Month',StringType())\
        .add('Day',StringType())\
        .add('Meridiem_index',StringType())\
        .add('Time_Hour',StringType()) \
        
        
    interval = kaf_df1.select(from_json(col('value'),schema).alias('data'))
    interval2 = interval.select('data.*')
    for i in interval2.columns:
        interval2 = interval2.withColumn(i, interval2[i].cast(DoubleType()))
    #interval2.printSchema()
    
    assembler = VectorAssembler(inputCols=['Month','Day','Time_Hour',
    'Violation_In_Front_Of_Or_Opposite_index','Street_Code1_index',
    'Issuer_Command_index','Violation_Location_index','Vehicle_Body_Type_index',
    'Meridiem_index','Registration_State_index','Plate_Type_index','Issuer_Precinct_index',
    'Street_Code2_index','Issuing_Agency_index','Violation_Code_index','Vehicle_Make_index',
    'Street_Code3_index'],outputCol='features')

    df = assembler.transform(interval2)
    pred = model.transform(df)

    predict= pred.select(['prediction'])

    label ={0.0:'NY',1.0:'K',2.0:'Q',3.0:'BX',4.0:'R'}
    predict = predict.withColumn('prediction',predict.prediction.cast(StringType()))
    predict = predict.withColumn('prediction', regexp_replace('prediction', '0.0', 'NY')) \
                    .withColumn('prediction', regexp_replace('prediction', '1.0', 'K')) \
                    .withColumn('prediction', regexp_replace('prediction', '2.0', 'Q')) \
                    .withColumn('prediction', regexp_replace('prediction', '3.0', 'BX'))   \
                    .withColumn('prediction', regexp_replace('prediction', '4.0', 'R'))        
    
    test = predict.writeStream.trigger(processingTime='15 seconds') \
        .outputMode('update') \
        .option('truncate','false') \
        .format('console') \
        .start()
    
    test.awaitTermination()
