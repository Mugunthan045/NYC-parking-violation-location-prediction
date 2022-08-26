
import datetime 
from pyspark.sql import functions as func
from pyspark.sql.functions import col,regexp_replace
from pyspark.sql.types import IntegerType,DoubleType
from pyspark.sql import SparkSession

INPUT_PATH = "s3n://parking2013-2014/parking.csv"
OUTPUT_PATH = "s3a://parking2013-2014/processed_data"

if __name__ == '__main__':

    session = SparkSession.builder.appName("Pre-Processing").getOrCreate()
    df =  session.read.option('header','true') \
        .option('inferSchema','true') \
        .csv(INPUT_PATH)

    
    def day_finder(x):
        return datetime.datetime.strptime(x, '%m/%d/%Y').weekday()
    day_udf = func.udf(lambda x: day_finder(x), IntegerType())

    df = df.withColumnRenamed('Violation Time', 'Violation_Time')

    df_new = df.withColumn('Month',func.split('Issue Date','/')[0]) \
                .withColumn('Year',func.split('Issue Date','/')[2]) \
                .withColumn('Day',day_udf(func.col('Issue Date')))  \
                .withColumn('Meridiem', \
                        func.when(func.isnan(df.Violation_Time) \
                                | func.col('Violation_Time').isNull()\
                                , func.lit(None))\
                        .otherwise(func.substring(df.Violation_Time,5,1))) \
                .withColumn('Time_Hour', \
                        func.when(func.isnan(df.Violation_Time) \
                                | func.col('Violation_Time').isNull()\
                                , func.lit(None))\
                        .otherwise(func.substring(df.Violation_Time,1,2)))

    df_new = df_new.withColumn('Year',df_new['Year'].cast(IntegerType())) \
        .withColumn("Month",df_new["Month"].cast(DoubleType())) \
        .withColumn("Day",df_new["Day"].cast(DoubleType())) \
        .withColumn("Time_Hour",df_new["Time_Hour"].cast(DoubleType()))

    df_new = df_new.where(func.col('Year')<2015)
    df_new = df_new.where(func.col('Year')>2012)

    df_new = df_new.withColumn("Meridiem", \
        func.when(col("Meridiem")=="" ,None) \
            .otherwise(col("Meridiem"))) 

    df_new = df_new.withColumn("Time_Hour", func.when((func.col("Time_Hour") <= 0.0) | (func.col("Time_Hour") > 12.0),1.0) \
                .otherwise(df_new.Time_Hour))

    df_new = df_new.withColumnRenamed('Registration State', 'Registration_State')
    df_new = df_new.filter(df_new.Registration_State != '99')

    df_new = df_new.withColumnRenamed('Plate Type', 'Plate_Type')
    df_new = df_new.filter(df_new.Plate_Type != '999')    

    df_new = df_new.withColumnRenamed('Days Parking In Effect    ','Days Parking In Effect')
    df_new = df_new.withColumnRenamed('Community Council ','Community Council')

    columns_to_drop = ['Summons Number','Plate ID','Issuer Code','Vehicle Expiration Date','Violation Precinct','House Number','Street Name','Date First Observed','Law Section','Sub Division','Vehicle Color','Vehicle Year','Feet From Curb','Violation Post Code','Violation Description']
    columns_missing2 = ['Days Parking In Effect','From Hours In Effect','To Hours In Effect']
    columns_missing = ['Time First Observed','Intersecting Street','Violation Legal Code','Unregistered Vehicle?','Meter Number',"No Standing or Stopping Violation","Hydrant Violation","Double Parking Violation","Latitude","Longitude","Community Board","Community Council","Census Tract","BIN","BBL","NTA"]
    columns_to_drop2 = ['Year','Violation_Time','Issue Date','Issuer Squad']


    final_columns = columns_missing+columns_missing2+columns_to_drop+columns_to_drop2

    df_new =df_new.drop(*final_columns)

    df_new = df_new.withColumn('Violation County', regexp_replace('Violation County', 'KINGS', 'K')) \
                    .withColumn('Violation County', regexp_replace('Violation County','QUEEN', 'Q')) \
                    .withColumn('Violation County', regexp_replace('Violation County', 'BRONX', 'BX')) \
                    .withColumn('Violation County', regexp_replace('Violation County', 'RC', 'R'))   \
                    .withColumn('Violation County', regexp_replace('Violation County', 'RICH','R'))  \
                    .withColumn('Violation County', regexp_replace('Violation County', 'NYC', 'NY'))

    df_new = df_new.dropna(how='any',subset=['Violation County'])

    df_new = df_new.withColumnRenamed('Violation County', 'Violation_County')
    df_new = df_new.filter(df_new.Violation_County != '103')

    df_new=df_new.select([func.when(func.col(c)=="",None) \
        .otherwise(func.col(c)).alias(c) for c in df_new.columns])

    df_new = df_new.dropna(how='any')
    df_new = df_new.select([func.col(col).alias(col.replace(' ', '_')) for col in df_new.columns])



    df_new.coalesce(1).write.format('csv') \
                .option("header",True) \
                .save(OUTPUT_PATH)

    session.stop()