from transformers import pipeline
from tensorflow.keras.models import load_model
from tensorflow.config import list_physical_devices
from tensorflow.math import top_k
import pickle
from random import randint
import nltk
import numpy as np


# Print n gpus
print("Num GPUs Available: ", len(list_physical_devices('GPU')))

# Load all the models
inhouse = load_model('models/modelo_inhouse')
vectorizer = load_model('models/vectorizer-model')
vectorizer = vectorizer.layers[0]

# Modelo glove
glove = load_model('models/modelo-glove')
vec_glove = load_model('models/vectorizer-glove')
vec_glove = vec_glove.layers[0]

# Baseline
with open('models/baseline-model.pickle', 'rb') as f:
    baseline = pickle.load(f)


# model = pipeline('text-generation', model='gpt2')
# Modelo gpt2
gpt_2 = pipeline('text-generation', model='gpt2')


def baseline_generate(text, n_tokens):
    tokenized_previous = nltk.word_tokenize(text.lower())

    generated_text = baseline.generate(n_tokens, text_seed=tokenized_previous)
    
    texto_gerado = [token for token in generated_text if token != '<s>' and token != '</s>']
    return text + " " + ' '.join(texto_gerado)


def neural_network_generate(model_id, model, vectorizer, lyrics, n_new_tokens=10) -> str:
    context = lyrics
    phrase = [lyrics]
    voc = vectorizer.get_vocabulary()

    for _ in range(n_new_tokens):
        vectorized = vectorizer([context])
        # if model_id == 1:
        #     pred = model.predict(vectorized[:,:-1])
        # else:
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
                idx = np.random.choice(len(voc))
                # If the word is [UNK] or ''
                if idx == 0 or idx == 1:
                    idx = 2
                word = voc[idx]
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


def generate_with_random_model(base_text, n) -> tuple:
    model_number = randint(0, 3)
    if model_number == 0:
        res = baseline_generate(base_text, n), 'baseline'
    elif model_number ==  1:
        res = neural_network_generate(1, inhouse, vectorizer, base_text, n), 'inhouse'
    elif model_number == 2:
        res = neural_network_generate(2, glove, vec_glove, base_text, n), 'glove'
    else:
        res = gpt_2_generate(base_text, n), 'gpt2'
    return res

