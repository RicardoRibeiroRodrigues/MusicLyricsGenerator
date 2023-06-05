import pyspark
from pyspark.sql import SparkSession
from random import randint
import re
import numpy as np
from tensorflow import keras
from tensorflow.math import top_k
import nltk
from transformers import pipeline
import time
import pickle


spark = SparkSession.builder \
    .appName('MusicGen') \
    .master("local[*]") \
    .getOrCreate()

test = spark.read.format("parquet").load("test.parquet")

n_rows = test.count()
print(f"N rows: ", n_rows)

# Comment for the baseline model
VECTORIZER_PATH = 'vectorizer-glove'
# vectorizer = keras.models.load_model(VECTORIZER_PATH)
# vectorizer = vectorizer.layers[0]

MODEL_PATH = 'baseline-model.pickle'
# model = keras.models.load_model(MODEL_PATH)

# For the baseline model
# with open(MODEL_PATH, 'rb') as f:
#     model = pickle.load(f)
# Modelo gpt2
gpt_2 = pipeline('text-generation', model='gpt2')

def baseline_generate(text, n_tokens, model):
    tokenized_previous = nltk.word_tokenize(text.lower())

    generated_text = model.generate(n_tokens, text_seed=tokenized_previous)
    
    texto_gerado = [token for token in generated_text if token != '<s>' and token != '</s>']
    return text + " " + ' '.join(texto_gerado)

# Calculate the average inference time for completing 10 tokens of a song
def complete_song_nn(lyrics, n_new_tokens=10) -> str:
    context = lyrics
    phrase = [lyrics]
    voc = vectorizer.get_vocabulary()

    for _ in range(n_new_tokens):
        vectorized = vectorizer([context])
        pred = model.predict(vectorized)
        n_iter = 0

        while True:

            k_best_predictions = top_k(pred, k=10).indices[0,:]
            idx = np.random.choice(k_best_predictions.numpy())
            word = voc[idx]

            if word in phrase or word == '' or word == ' ':
                pred[0][idx] = 0
            else:
                break

            if n_iter > 100:
                print("Cant find a word, using random")
                word = np.random.choice(voc)
                break
            n_iter += 1
        phrase.append(word)
        context = context + " " + word
        context = ' '.join(context.split()[1:])
    return " ".join(phrase)

def gpt_2_generate(text, n_tokens) -> str:
    prompt_text = "Complete the song: \n"
    generated_text = gpt_2(prompt_text + text, max_new_tokens=n_tokens, num_return_sequences=1)[0]['generated_text']
    return generated_text.replace(prompt_text, '')

def get_random_snippet(lyrics):
    tokens = re.findall(r"\b\w+\b", lyrics)
    start = randint(0, len(tokens) - 10)
    snippet = tokens[start:start+10]
    return " ".join(snippet)

times = []
for _ in range(25):
    idx = randint(0, n_rows)
    lyrics = test.select("lyrics").where(f"id == {idx}").collect()[0][0]
    snippet = get_random_snippet(lyrics)
    print("Snippet: ", snippet)
    start = time.time()
    # print("Inhouse/Glove: ", complete_song_nn(snippet, 10))
    # print("Baseline: ", baseline_generate(snippet, 10, model))
    print("GPT-2: ", gpt_2_generate(snippet, 10))
    end = time.time()
    delta = end - start
    times.append(delta)
    print("Time: ", delta)

print("Average time: ", np.mean(times))