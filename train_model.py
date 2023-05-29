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

print(df.rdd.getNumPartitions())
print(sc.getConf().get("spark.executor.cores"))
print(sc.getConf().get("spark.default.parallelism"))

cols = ("artist", "year", "views", "id", "language_cld3", "language_ft")

df = df.drop(*cols)
df = df.filter("tag = 'pop' AND language = 'en'")
print("Number of pop musics in english: ", df.count())

seed = 69
train, test = df.randomSplit([0.8, 0.2], seed)

train = train.select("*").withColumn("id", monotonically_increasing_id())

train.printSchema()

n_rows = train.count()
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
                    .where(f"id > ({self.batch_size * idx})") \
                    .limit(self.batch_size)
        rows = rows.toPandas()
        X = rows["lyrics"].to_numpy()
        return X
        

vocab_size = 10_000
n_grams = 10
batch_size = 50_000
dataset = BatchDataset(train, batch_size, n_rows)

vectorize_layer = TextVectorization(
        max_tokens=vocab_size, output_sequence_length=n_grams
    )

print("Adaptando o vetorizador")
# vectorize_layer.adapt(dataset)

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(1, ), dtype=tf.string),
    vectorize_layer
])
filepath = "vectorizer-model"
# model.compile()
model.load_weights(filepath)
print("Depois do Load")
vectorize_layer = model.layers[0]

# # Save.
print("Salvando o vetorizador")
model.save(filepath)
print("Vetorizador foi salvo!")


# Quantos para frente
N = 50
def get_last_token(x):
        """
        Function to map the dataset to (x, y) pairs.
        The y is last token of x.
        x is output of vectorization - last token.
        """
        vectorized_x = vectorize_layer(x)
        X = []
        Y = []
        # y_ = x_[:,-1:]
        # x_ = x_[:, :-1]
        i = 0
        while i + N < len(vectorized_x):
            X.append(vectorized_x[i:i+N])
            Y.append(vectorized_x[i+N])
            i += 1
        return np.array(X), np.array(Y)


def predict_word(seq_len, latent_dim, vocab_size):
    input_layer = Input(shape=(seq_len-1,))
    x = input_layer
    x = Embedding(vocab_size, latent_dim, name='embedding', mask_zero=True)(x)
    x = MultiHeadAttention(num_heads=3, key_dim=2)(x, value=x)
    x = GlobalAveragePooling1D()(x)
    latent_rep = x
    x = Dense(vocab_size)(x)
    x = Softmax()(x)
    return Model(input_layer, x), Model(input_layer, latent_rep)

predictor, latent = predict_word(n_grams, 15, vocab_size)
predictor.summary()
#opt = keras.optimizers.SGD(learning_rate=1, momentum=0.9)
opt = keras.optimizers.Nadam(learning_rate=0.1)
loss_fn = keras.losses.SparseCategoricalCrossentropy(
    ignore_class=1,
    name="sparse_categorical_crossentropy",
)

predictor.compile(loss=loss_fn, optimizer=opt, metrics=["accuracy"])

history = predictor.fit(dataset.map(get_last_token), epochs=50, verbose=1)

predictor.save("modelo_inhouse")