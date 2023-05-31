import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Input, Dense, Softmax, GlobalAveragePooling1D, MultiHeadAttention
from keras.models import Model
from tensorflow import keras
import pyspark
from pyspark.pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType 
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark.sql.functions import col, array_contains, monotonically_increasing_id , avg, length
import os
import numpy as np
from tqdm import tqdm
from random import randint
import re
import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .appName('MusicGen') \
    .master("local[*]") \
    .config("spark.default.parallelism", "10") \
    .config("spark.executor.cores", "7") \
    .getOrCreate()

sc = spark.sparkContext

train = spark.read.format("parquet").load("train.parquet")
train.printSchema()

n_rows = train.count()
print(f"N rows: ", n_rows)

vocab_size = 20_000
n_grams = 10
vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=n_grams)

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
        

# batch_size = 8_192
batch_size = 16_384
dataset = BatchDataset(train, batch_size, n_rows)
vectorizer.adapt(dataset)


model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(1, ), dtype=tf.string),
    vectorizer
])

filepath = "vectorizer-glove"
model.compile()
# model.load_weights(filepath)
# loaded_model = tf.keras.models.load_model(filepath)

# print("Depois do Load")
# vectorizer = loaded_model.layers[0]
print(vectorizer.get_vocabulary()[0:25])

voc = vectorizer.get_vocabulary()
# # Save.
print("Salvando o vetorizador")
model.save(filepath)
print("Vetorizador foi salvo!")

word_index = dict(zip(voc, range(len(voc))))


path_to_glove_file = os.path.join(
    "./glove.6B.100d.txt"
)

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

def predict_word(seq_len, vocab_size):
    input_layer = Input(shape=(seq_len,))
    x = input_layer
    x = embedding_layer(x)
    x = MultiHeadAttention(num_heads=3, key_dim=2)(x, value=x)
    x = GlobalAveragePooling1D()(x)
    latent_rep = x
    x = Dense(vocab_size)(x)
    x = Softmax()(x)
    return Model(input_layer, x), Model(input_layer, latent_rep)

predictor, latent = predict_word(n_grams, num_tokens)
predictor.summary()
#opt = keras.optimizers.SGD(learning_rate=1, momentum=0.9)
opt = keras.optimizers.Nadam(learning_rate=0.1)
loss_fn = keras.losses.SparseCategoricalCrossentropy(
    ignore_class=1,
    name="sparse_categorical_crossentropy",
)

predictor.compile(loss=loss_fn, optimizer=opt, metrics=["accuracy"])

N = n_grams
TOKEN_REGEX = r'\b\w+\b'
def get_last_token(x):
    """
    Function to map the dataset to (x, y) pairs.
    The y is last token of x.
    x is output of vectorization - last token.
    """
    X = []
    Y = []
    for music in x:
        tokens = re.findall(TOKEN_REGEX, music)
        # Get one random part of the music
        if len(tokens) - N - 1 <= 0:
            print("MUSICA INVALIDA: ")
            print(len(tokens))
            print(music)
        i = randint(0, len(tokens) - N - 1)
        X.append(" ".join(tokens[i:i+N]))
        Y.append(str(tokens[i+N]))
        
    return (np.array(X), np.array(Y))

class BatchDatasetTrain(tf.keras.utils.Sequence):    
    def __init__(self, dataset_spark, batch_size, dataset_len, vectorizer):
        self.batch_size = batch_size
        self.dataset_spark = dataset_spark
        self.dataset_full_len = dataset_len
        self.vectorizer = vectorizer
    
    def __len__(self):
        return int(np.ceil(self.dataset_full_len / self.batch_size))

    def __getitem__(self, idx):
        rows = self.dataset_spark \
                    .where(f"id > ({self.batch_size * idx})") \
                    .limit(self.batch_size)
        # rows = rows.
        rows = rows.toPandas()
        X = rows["lyrics"].to_numpy()
        X, y = get_last_token(X)

        X = self.vectorizer(X)
        y = self.vectorizer(y)[:, 0]
        return X, y

dataset_train = BatchDatasetTrain(train, batch_size, n_rows, vectorizer)

# Training checkpoints 
checkpoint_path = "training_glove/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)


history = predictor.fit(dataset_train, epochs=50, verbose=1, callbacks=[cp_callback])

predictor.save("modelo-glove")

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()

# Save the figure
plt.savefig('validation_curves_glove.png')