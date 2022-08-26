from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.context import SparkContext
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

MODEL_PATH = './RFmodel_copy'

sc = SparkContext.getOrCreate()

evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction')
def indexing(df):
    index = [StringIndexer(inputCol=column, outputCol=column+"_index",handleInvalid='keep').fit(df) for column in list(set(df.columns)-set(['Month','Day','Time_Hour','Violation_County']))]
    target_index = StringIndexer(inputCol="Violation_County", outputCol="label",handleInvalid='keep').fit(df)
    assembler = VectorAssembler(inputCols=['Month','Day','Time_Hour','Violation_In_Front_Of_Or_Opposite_index','Street_Code1_index','Issuer_Command_index','Violation_Location_index','Vehicle_Body_Type_index','Meridiem_index','Registration_State_index','Plate_Type_index','Issuer_Precinct_index','Street_Code2_index','Issuing_Agency_index','Violation_Code_index','Vehicle_Make_index','Street_Code3_index'],outputCol='features')


    pipeline = Pipeline(stages=index+[target_index,assembler])
    df = pipeline.fit(df).transform(df)

    return df

def prediction(df):
    rf = RandomForestClassifier(maxBins=6700,labelCol="label", featuresCol="features")
    #model = RandomForestClassificationModel.load('./RFmodel_copy')
    train, test = df.randomSplit([0.8,0.2])
    model = rf.fit(train)
    prediction = model.transform(test)

    return prediction

def accuracy(pred):
    accuracy = evaluator.evaluate(pred)
    print("Accuracy  = %s" % (accuracy))


