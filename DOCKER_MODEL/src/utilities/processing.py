import datetime
from pyspark.sql import functions as func
from pyspark.sql.functions import col,regexp_replace
from pyspark.sql.types import IntegerType,DoubleType

def splitting_columns(df):
    def day_finder(x):
        return datetime.datetime.strptime(x, '%m/%d/%Y').weekday()
    day_udf = func.udf(lambda x: day_finder(x), IntegerType())
    df = df.withColumnRenamed('Violation Time','Violation_Time')
    df = df.withColumn('Month',func.split('Issue Date','/')[0]) \
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

    df = df.withColumn('Year',df['Year'].cast(IntegerType())) \
    .withColumn("Month",df["Month"].cast(DoubleType())) \
    .withColumn("Day",df["Day"].cast(DoubleType())) \
    .withColumn("Time_Hour",df["Time_Hour"].cast(DoubleType()))
        
        
    return df

def target_column_processing(df):
    df = df.withColumn('Violation County', regexp_replace('Violation County', 'KINGS', 'K')) \
                .withColumn('Violation County', regexp_replace('Violation County','QUEEN', 'Q')) \
                .withColumn('Violation County', regexp_replace('Violation County', 'BRONX', 'BX')) \
                .withColumn('Violation County', regexp_replace('Violation County', 'RC', 'R'))   \
                .withColumn('Violation County', regexp_replace('Violation County', 'RICH','R'))  \
                .withColumn('Violation County', regexp_replace('Violation County', 'NYC', 'NY'))

    df = df.dropna(how='any',subset=['Violation County'])

    df = df.withColumnRenamed('Violation County', 'Violation_County')
    df = df.filter(df.Violation_County != '103')

    return df

def cleaning_values(df):
    df = df.where(func.col('Year')<2015)
    df = df.where(func.col('Year')>2012)
    df = df.withColumn("Meridiem", \
       func.when(col("Meridiem")=="" ,None) \
          .otherwise(col("Meridiem")))

    df = df.withColumn("Time_Hour", func.when((func.col("Time_Hour") <= 0.0) | (func.col("Time_Hour") > 12.0),1.0) \
            .otherwise(df.Time_Hour))

    return df

def dropping_columns(df):
    
    df = df.withColumnRenamed('Days Parking In Effect    ','Days Parking In Effect')
    df = df.withColumnRenamed('Days Parkin In Effect','Days Parking In Effect')
    df = df.withColumnRenamed('Community Council ','Community Council')

    
    columns_to_drop = ['Summons Number','Plate ID','Issuer Code','Vehicle Expiration Date','Violation Precinct','House Number','Street Name','Date First Observed','Law Section','Sub Division','Vehicle Color','Vehicle Year','Feet From Curb','Violation Post Code','Violation Description']
    columns_missing2 = ['Days Parking In Effect','From Hours In Effect','To Hours In Effect']
    columns_missing = ['Time First Observed','Intersecting Street','Violation Legal Code','Unregistered Vehicle?','Meter Number',"No Standing or Stopping Violation","Hydrant Violation","Double Parking Violation","Latitude","Longitude","Community Board","Community Council","Census Tract","BIN","BBL","NTA"]
    columns_to_drop2 = ['Year','Violation_Time','Issue Date','Issuer Squad']
    final_columns = columns_missing+columns_missing2+columns_to_drop+columns_to_drop2

    
    df =df.drop(*final_columns)

    return df

def dropping_null_rows(df):
    df=df.select([func.when(func.col(c)=="",None) \
    .otherwise(func.col(c)).alias(c) for c in df.columns])

    df = df.dropna(how='any')
    df = df.select([func.col(col).alias(col.replace(' ', '_')) for col in df.columns])


    return df