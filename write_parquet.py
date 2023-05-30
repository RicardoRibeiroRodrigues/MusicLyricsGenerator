import pyspark
from pyspark.pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType 
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark.sql.functions import col, array_contains
from pyspark.sql.functions import monotonically_increasing_id 
import os

spark = SparkSession.builder \
    .appName('MusicGen') \
    .master("local[*]") \
    .config("spark.default.parallelism", "10") \
    .config("spark.executor.cores", "7") \
    .getOrCreate()

sc = spark.sparkContext

schema = StructType() \
      .add("title", StringType(),True) \
      .add("tag", StringType(), True) \
      .add("artist", StringType(), True) \
      .add("year", IntegerType(), True) \
      .add("views", IntegerType(), True) \
      .add("features", StringType(), True) \
      .add("lyrics", StringType(), False) \
      .add("id", IntegerType(), True) \
      .add("language_cld3", StringType(), True) \
      .add("language_ft", StringType(), True) \
      .add("language", StringType(), True)

# df = spark.read.csv("song_lyrics.csv")
# df.printSchema()
DATASET_PATH = "song_lyrics.csv"
df = spark.read.format("csv") \
      .option("header", True) \
      .option("multiLine", True) \
      .option("escape","\"") \
      .schema(schema) \
      .load(DATASET_PATH)
df.printSchema()

cols = ("artist", "year", "views", "id", "language_cld3", "language_ft")

df = df.drop(*cols)
df = df.filter("tag = 'pop' AND language = 'en'")
print("Number of pop musics in english: ", df.count())

seed = 69
train, test = df.randomSplit([0.8, 0.2], seed)

train = train.select("*").withColumn("id", monotonically_increasing_id())
test  = test.select("*").withColumn("id", monotonically_increasing_id())

train.printSchema()

# Write the 'train' DataFrame to Parquet with 10 output files
train.repartition(25).write.format("parquet").save("train.parquet")

# Write the 'test' DataFrame to Parquet
test.repartition(10).write.format("parquet").save("test.parquet")
