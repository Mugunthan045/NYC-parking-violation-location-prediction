from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier


INPUT_PATH = "s3n://parking2013-2014/processed_data/processed_parking.csv"

if __name__ == '__main__':

    
    session = SparkSession.builder.appName("Model_building").getOrCreate()
    df =  session.read.option('header','true') \
        .option('inferSchema','true') \
        .csv(INPUT_PATH)


    index = [StringIndexer(inputCol=column, outputCol=column+"_index",handleInvalid='keep').fit(df) for column in list(set(df.columns)-set(['Month','Day','Time_Hour','Violation_County']))]
    target_index = StringIndexer(inputCol="Violation_County", outputCol="label",handleInvalid='keep').fit(df)
    assembler = VectorAssembler(inputCols=['Month','Day','Time_Hour','Violation_In_Front_Of_Or_Opposite_index','Street_Code1_index','Issuer_Command_index','Violation_Location_index','Vehicle_Body_Type_index','Meridiem_index','Registration_State_index','Plate_Type_index','Issuer_Precinct_index','Street_Code2_index','Issuing_Agency_index','Violation_Code_index','Vehicle_Make_index','Street_Code3_index'],outputCol='features')


    pipeline = Pipeline(stages=index+[target_index,assembler])
    df = pipeline.fit(df).transform(df)

    train, test = df.randomSplit([0.8,0.2])

    rf = RandomForestClassifier(maxBins=6700,labelCol="label", featuresCol="features")
    model_rf = rf.fit(train)
    #model_rf.save('s3a//parking2013-2014/model')
    pred_rf = model_rf.transform(test)

    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features",maxBins=6700)
    model_dt = dt.fit(train)
    #model_dt.save('s3a//parking2013-2014/model')
    pred_dt = model_dt.transform(test)

    evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction')
    accuracy_rf = evaluator.evaluate(pred_rf)
    accuracy_dt = evaluator.evaluate(pred_dt)
    print("Accuracy for Random Forest = %s" % (accuracy_rf))
    print("Test Error for Random Forest = %s" % (1.0 - accuracy_rf))
    print("Accuracy for Decision Tree = %s" % (accuracy_dt))
    print("Test Error for Decision Tree = %s" % (1.0 - accuracy_dt))

    session.stop()