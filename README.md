# NLP final project - Music Generator

## Overview
This project aimed to explore various Natural Language Processing (NLP) techniques to create a model capable of completing music lyrics. Leveraging a range of methods, from classical machine learning to deep learning and Large Language Models (LLMs) (such as GPT-2), our objective was to enhance the creative process in songwriting by predicting and generating lyrics that seamlessly align with the style and theme of a given musical piece.

## Stages
1. Data preprocessing: Before it was possible to train the models, it was needed to transform the [data](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information) from csv to parquet, due to the size and the inefficiency of reading from the original file, this was done in `write_parquet.py` file, separating in train and test as well.
1. Baseline: Machine learning model trained in 'proof-of-concepy.ipynb' - trained in less than 1 hour.
2. Deep Learning: Fully trained Deep learning model trained using 'train_model.py' - trained for 4 hours in a GPU.
3. Glove: Deep learning model + Stanford's Glove - Trained for 7 hours in a CPU.
4. GPT-2: Using OPENAI's GPT-2 available in [hugging face](https://huggingface.co/gpt2).

all the models are available in this [zipped folder](https://drive.google.com/file/d/1EqeK3p-7DsB_IAL2_jywZgLU1YXBhJiq/view?usp=sharing)

## Getting Started

To reproduce the results or build upon this project, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/RicardoRibeiroRodrigues/MusicLyricsGenerator/
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Explore the Code:**
   - Navigate through the project folders and explore individual notebooks or scripts for each component.

## Made by

- [Ricardo Ribeiro Rodrigues](https://github.com/RicardoRibeiroRodrigues) - ricardorr7@al.insper.edu.br

## License

This project is licensed under the [Creative Commons License](LICENSE).
