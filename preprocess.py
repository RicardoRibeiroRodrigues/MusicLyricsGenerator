import pyspark
from pyspark.pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType 
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark.sql.functions import col, array_contains
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np


# add costumized ctrl C handler
import signal
import sys
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    print("Stopping SparkSession")
    spark.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

spark = SparkSession.builder \
    .appName("MusicGen") \
    .master("local[*]") \
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

cols = ("artist", "year", "views", "features", "id", "language_cld3", "language_ft")

df = df.drop(*cols)

wanted_tag = 'pop'
df = df.filter(f"tag = '{wanted_tag}' AND language = 'en'")

train, test = df.randomSplit([0.8, 0.2], 69)

train = train.select("*").withColumn("id", monotonically_increasing_id())

train.printSchema()

# spark.sql("CREATE DATABASE IF NOT EXISTS train_db")
# train.write.mode('overwrite').saveAsTable("train_db.musics")
# train.createOrReplaceTempView("view_musics")

# Partitioning the dataset.
# train = train.repartitionByRange(10, "id")

n_rows = train.count()
# n_rows = spark.sql("SELECT COUNT(*) FROM view_musics").collect()[0][0]
print(f"N rows train: ", n_rows)

vocab_size = 20_000
n_grams = 150
batch_size = 50_000

partition_ranges = np.arange(0, n_rows, batch_size, dtype=int)
print(partition_ranges)
print("------------------ Start of partitioning ------------------")
# Assign partitions based on the partition ranges
df_with_partitions = train.withColumn('partition', next(
    i for i, start in enumerate(partition_ranges) if start <= col('id')
))

# Write the partitioned DataFrame to Parquet files
df_with_partitions.write.partitionBy('partition').parquet('partitioned_data_train')

# Make the same for test dataset
test = test.select("*").withColumn("id", monotonically_increasing_id())
n_rows = test.count()
print(f"N rows test: ", n_rows)

partition_ranges = np.arange(0, n_rows, batch_size, dtype=int)
print(partition_ranges)
print("------------------ Start of partitioning ------------------")
# Assign partitions based on the partition ranges
df_with_partitions = test.withColumn('partition', next(
    i for i, start in enumerate(partition_ranges) if start <= col('id')
))

# Write the partitioned DataFrame to Parquet files
df_with_partitions.write.partitionBy('partition').parquet('partitioned_data_test')

spark.stop()