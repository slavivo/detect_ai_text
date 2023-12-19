# Detect AI generated text

## Introduction

This project is on the public kaggle competition [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text). In this competition the task is to determine whether an essay was written by a human or by an AI. The used model can be any chatbot.

This repository contains the code for the three approaches I used in this competition. The first one is an ensemble of classifiers with a tfidf vectorizer. The second one is a classifier pretrained DebertaV3 model (encoder approach). The third one is a classifier with a pretrained Mistral-7B model with LoRA and quantization (decoder approach).

Disclaimer: the TF-IDF method was copied from these notebooks: [Create your own tokenizer](https://www.kaggle.com/code/datafan07/train-your-own-tokenizer/notebook), [LLM daigttext](https://www.kaggle.com/code/yongsukprasertsuk/llm-daigtext-0-961).

## Instructions

You can run the code on kaggle or locally, but it is **very** recommended to run it on kaggle, as it is way easier to setup and run. For local setup you will need a GPU with at least 16GB of VRAM and also install the requirements, which are big and will take a while to install (requirements are taken from native kaggle notebooks).

### Kaggle setup

TF-IDF method:

1. Copy the [notebook](https://www.kaggle.com/code/vojtchslavk/tf-idf-detect-ai) to your kaggle account.
2. Run the notebook.

DebertaV3 and Mistral-7B method:

Inference:

1. Copy the [notebook](https://www.kaggle.com/code/vojtchslavk/detect-ai-llm-inference) (contains inference for both encoder and decoder based approaches) to your kaggle account.
2. Run the notebook.

Training:

1. Copy the [notebook](https://www.kaggle.com/code/vojtchslavk/detect-ai-llm-train) (contains training for both encoder and decoder based approaches) to your kaggle account.
2. Run the notebook.

### Local setup

1. Clone the repository.
2. Install the requirements: `pip install -r requirements.txt` (beware it contains all requirements of native kaggle notebook).
3. Download the data as described in the [data section](##data).
4. Change the paths in the notebooks to the paths on your local machine (e.g. from /kaggle/input/* to ./*)
5. Run the notebooks.

## Data

[TF-IDF approach](https://www.kaggle.com/code/vojtchslavk/tf-idf-detect-ai):

1. [Original dataset](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)

[DebertaV3 and Mistral-7B training](https://www.kaggle.com/code/vojtchslavk/detect-ai-llm-train):

1. [Original dataset](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)
2. [Additional dataset](https://www.kaggle.com/datasets/radek1/llm-generated-essays/)
3. [Additional dataset](https://www.kaggle.com/datasets/alejopaullier/daigt-external-dataset/)
4. [Additional dataset](https://www.kaggle.com/datasets/nbroad/daigt-data-llama-70b-and-falcon180b/)

[DebertaV3 and Mistral-7B inference](https://www.kaggle.com/code/vojtchslavk/detect-ai-llm-inference):

1. [Original dataset](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)
2. [Mistal 7b model](https://www.kaggle.com/datasets/datafan07/mistral-7b-v0-1/)
3. [Pip dependencies](https://www.kaggle.com/code/hotchpotch/llm-detect-pip)
4. [Trained models](https://www.kaggle.com/code/vojtchslavk/detect-ai-llm-train)

The pip dependencies are taken from other notebooks, because the competition does not allow internet access when submitting.