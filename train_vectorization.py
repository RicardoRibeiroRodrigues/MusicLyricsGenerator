import pyspark
from pyspark.pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType 
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark.sql.functions import col, array_contains
from pyspark.sql.functions import monotonically_increasing_id 
from keras.layers import (
    Input, Dense, Activation, TimeDistributed, Softmax, TextVectorization, Reshape,
    RepeatVector, Conv1D, Bidirectional, AveragePooling1D, UpSampling1D, Embedding,
    Concatenate, GlobalAveragePooling1D, LSTM, Multiply, MultiHeadAttention
)
from keras.models import Model
import tensorflow as tf
import keras
import numpy as np

spark = SparkSession.builder.appName('MusicGen').getOrCreate()

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


train, test = df.randomSplit([0.8, 0.2], 69)

train = train.select("*").withColumn("id", monotonically_increasing_id())

train.printSchema()

wanted_tag = 'pop'
train_ = train.filter(df.tag == wanted_tag)
n_rows = train_.count()
print(f"N rows: ", n_rows)

class BatchDataset(tf.keras.utils.Sequence):    
    def __init__(self, dataset_spark, batch_size, dataset_len):
        self.batch_size = batch_size
        self.dataset_spark = dataset_spark
        self.dataset_full_len = dataset_len
    
    def __len__(self):
        return int(np.ceil(self.dataset_full_len / self.batch_size))

    def __getitem__(self, idx):
        print(f"Iter: {idx}")
        rows = self.dataset_spark \
                    .where(df.id > (self.batch_size * idx)) \
                    .limit(self.batch_size)
        rows = rows.toPandas()
        X = rows["lyrics"].to_numpy()
        return X
        
    
#     def _shuffle(self):
#         import random
#         temp = list(zip(self.list_of_files, self.list_of_labels))
#         random.shuffle(temp)
#         res1, res2 = zip(*temp)
#         # res1 and res2 come out as tuples, and so must be converted to lists.
#         self.list_of_files, self.list_of_labels = list(res1), list(res2)

#     def on_epoch_end(self):
#         self._shuffle()

vocab_size = 10_000
n_grams = 10
batch_size = 40_000

vectorize_layer = TextVectorization(
        max_tokens=vocab_size, output_sequence_length=n_grams
    )
dataset = BatchDataset(train_, batch_size, n_rows)

vectorize_layer.adapt(dataset)
