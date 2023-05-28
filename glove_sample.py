import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Input, Dense, Softmax, GlobalAveragePooling1D, MultiHeadAttention
from keras.models import Model
from tensorflow import keras
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

# Get only a small chunk of the data
df = pd.read_csv("song_lyrics.csv", nrows=10_000)

# Get only the columns we need
df = df[["title", "lyrics", "tag", "language"]]

# Filter only english songs
df = df[df.language == "en"]

print(df.iloc[[0, 1, 2]])

vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
# USAR DATASET COM TRAIN E TEST
text_ds = tf.data.Dataset.from_tensor_slices(df["lyrics"].to_numpy()).batch(128)
vectorizer.adapt(text_ds)

voc = vectorizer.get_vocabulary()
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

predictor, latent = predict_word(15, num_tokens)
predictor.summary()
#opt = keras.optimizers.SGD(learning_rate=1, momentum=0.9)
opt = keras.optimizers.Nadam(learning_rate=0.1)
loss_fn = keras.losses.SparseCategoricalCrossentropy(
    ignore_class=1,
    name="sparse_categorical_crossentropy",
)

predictor.compile(loss=loss_fn, optimizer=opt, metrics=["accuracy"])
# TROCAR PARA TRAIN.


def take_last_token(dataset):
    N = 50
    X = []
    Y = []
    for text in tqdm(dataset):
        vectorized_x = vectorizer(text)
        i = 0
        while i + N < len(vectorized_x):
            X.append(vectorized_x[i:i + N])
            Y.append(vectorized_x[i + N])
            i += 1
    return np.array(X), np.array(Y)

X, Y = take_last_token(df["lyrics"])
predictor.fit(X, Y, batch_size=128, epochs=2, verbose=1)