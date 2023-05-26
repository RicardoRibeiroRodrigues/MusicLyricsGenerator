from keras.layers import Input, Dense, Activation, Softmax, TextVectorization
from keras.models import Model
import tensorflow as tf
import keras
import numpy as np

vocab_size = 5000
vectorize_layer = TextVectorization(max_tokens=vocab_size, output_sequence_length=10)
vectorize_layer.adapt(["aaaaaaa", 'bbbbbbb', 'cccccc'])

print(vectorize_layer.get_vocabulary())


# Create model.
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(1, ), dtype=tf.string),
    vectorize_layer
])
print("Antes compile")
model.compile()

# # Save.
print("ANtes do Save")
filepath = "tmp-model"
model.save(filepath)

# # Load.
print("Depois do Save")
loaded_model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(1, ), dtype=tf.string),
    vectorize_layer
])
# loaded_model = tf.keras.models.load_model(filepath)
loaded_model.load_weights(filepath)
print("Depois do Load")
loaded_vectorizer = loaded_model.layers[0]
print(loaded_vectorizer.get_vocabulary())
print("Chega no final")