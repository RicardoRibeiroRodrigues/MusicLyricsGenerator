import tensorflow as tf
from tensorflow import keras
import pyspark
from pyspark.pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType 
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark.sql.functions import col, array_contains, monotonically_increasing_id , avg, length, udf
from random import randint
import re
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .appName('MusicGen') \
    .master("local[*]") \
    .config("spark.default.parallelism", "10") \
    .config("spark.executor.cores", "7") \
    .getOrCreate()

sc = spark.sparkContext

test = spark.read.format("parquet").load("test.parquet")
test.printSchema()

n_rows = test.count()
print(f"N rows: ", n_rows)

accuracies = []

MODEL_PATH = 'modelo_inhouse'
VECTORIZER_PATH = 'vectorizer-model'
model = keras.models.load_model(MODEL_PATH)

loaded_model = keras.models.load_model(VECTORIZER_PATH)

# print("Depois do Load")

vectorizer = loaded_model.layers[0]

print(vectorizer.get_vocabulary()[0:10])

def complete_song(lyrics, n_new_tokens=10) -> str:
    # print(lyrics)
    context = lyrics
    phrase = [lyrics]

    for _ in range(n_new_tokens):
        vectorized = vectorizer([context])
        pred = model.predict(vectorized)

        while True:
            candidatos = tf.math.top_k(pred, k=10).indices[0,:]
            idx = np.random.choice(candidatos.numpy())
            word = vectorizer.get_vocabulary()[idx]
            if word in phrase or word == '' or word == ' ':
                pred[0][idx] = 0
            else:
                break
        phrase.append(word)
        context = context + " " + word
        #print(frase)
        context = ' '.join(context.split()[1:])
    return " ".join(phrase)


# print(complete_song("The stars remind of"))
N = 10
TOKEN_REGEX = r'\b\w+\b'

def get_music_part(row):
    music = row['lyrics']
    tokens = re.findall(TOKEN_REGEX, music)
    i = randint(0, len(tokens) - N - 1)
    text = " ".join(tokens[i:i+N])
    return text, tokens[i+N]

def got_row_right(row):
    # music = row['lyrics']
    music = row['snippet']

    tokens = re.findall(TOKEN_REGEX, music)
    # Get one random part of the music
    i = randint(0, len(tokens) - N - 1)
    text = " ".join(tokens[i:i+N])
    vectorized = vectorizer([text])
    pred = model.predict(vectorized, verbose=0)
    idx = tf.argmax(pred, axis=1)[0]
    word = vectorizer.get_vocabulary()[idx]

    return word == row['next_word']
   
# For para o intervalo de confianca
voc = vectorizer.get_vocabulary()
for _ in range(10):
    right = 0
    sampled = test.sample(fraction=0.1)
    res = sampled.rdd.map(get_music_part).toDF(['snippet', 'next_word']).toPandas()
    total = len(res)
    print(total)
    # Got right needs next_word
    print(res.iloc[0])
    snippet_vectorized = vectorizer(res['snippet'].to_numpy())
    print("Fez a vetorizacao")
    pred = model.predict(snippet_vectorized, verbose=1)
    print("Fez as predicoes")
    idx = tf.argmax(pred, axis=1)
    print("Fez o argmax")
    res['predicted'] = [voc[index_pred] for index_pred in idx]
    print(res['predicted'].iloc[0])
    print("Contabilizando acertos")
    res['got_right'] = res['predicted'] == res['next_word']
    print("Somando acertos")
    right = res['got_right'].sum() 
    print(f"Right: {right} Total: {total}, Accuracy: {right/total}")

    # print(f"Right: {right} Total: {total}, Accuracy: {right/total}")
    accuracies.append(right/total)
    print("Proximo intervalo")

print(f"Mean: {np.mean(accuracies)}")

# Plot accuracies in confidence interval
plt.hist(accuracies, bins=10)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Accuracy Histogram')
plt.savefig('accuracy_histogram.png', dpi=300)


# Save the accuracies
with open('accuracies.pickle', 'wb') as f:
    pickle.dump(accuracies, f)
