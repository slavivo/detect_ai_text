# Milestone

## Introduction

This project is on the public kaggle competition [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text). In this competition the task is to determine whether an essay was written by a human or by an AI. This AI can be any chatbot. 

For me the motivation behind this work is to learn more about the current state of the art in NLP and the different machine learning techniques. ALso to learn more about how chatbots work and how they differ from human writing.

## Data

The dataset provided in the competition contains 10,000 essays, however only 1000 are labeled and can be used fo training. According to the competition the essays contain human-generated ones and also LLM-generated ones (from various LLMs). According to this [discussion](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453410) 7 essay prompts were used: "Car-free cities", "Does the electoral college work?", "Exploring Venus", "The Face on Mars", "Facial action coding system", "A Cowboy Who Rode the Waves", "Driverless cars".

Due to the low amount of essays in the training dataset it is crucial to use other data. Thankfully some kaggle users generated additional data - human and also LLM essays. These do not only contain the 7 prompts but also other ones. These are some examples: (2421 chatGPT essays with also human essays)[https://www.kaggle.com/datasets/alejopaullier/daigt-external-dataset/data] and (500 essays chatGTP3.5 )[https://www.kaggle.com/datasets/radek1/llm-generated-essays]. If more will be needed it is no problem to find some in the kaggle discussion section.

## Research

In this section I will describe different approaches and works related to this work. 

- Encoder-based architecture 
    - We can use various architectures, but the current state-of-the-art is debertav3. 
    - [Example - Deberta](https://www.kaggle.com/code/thedrcat/detectai-transformers-baseline).
- Decoder-based architecture 
    - In the last year it's been shows that decoder-based arachitectures can acheve better results in classification tasks than the encoder-based ones. Which is surprising and is something I want to compare in my work. However, these models are way larger than encoder-based ones and if I want to use it in the kaggle environment I will need to apply [LoRA](https://arxiv.org/abs/2106.09685) (is used to reduce the number of trainable parameters by inserting new trainable weights into the model while the rest is frozen) and [quantization](https://huggingface.co/docs/optimum/concept_guides/quantization) (is used to reduce the size of the loaded model - works because we don't train the model itsel because of LoRA). 
    - [Example - Mistral-7B with LoRA and quantizaation](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452362).
- [The Science of Detecting LLM-Generated Texts](https://arxiv.org/pdf/2303.07205.pdf)
    - THe works says that it seems like LLM text is generally a bit more formal and precise, human text is more emorional (very generally).
    - Zipf's law - in a corpus of NL the most frequent word will occur approx twice as much has the second, thrice as muc has the third and so on - this may not be the case in AI text as they *might* use the most frequent words way more than humans.
    - Measuring perplexity - we take a GPT model give it the sentence and then compare true word with its probability of generation in GPT - if it is generally low it should be a human text and vice versa.
    - Linguistic patterns - average word length, vocabulary size - e.g. chatGPT texts tend to have more diverse vocab and be longer.
    - Models tend to be less negative and contain no hate speech - could use pre-learnt classifier to ensemble.
- [Detecting ChatGPT: A Survey of the State of Detecting ChatGPT-Generated Text](https://arxiv.org/pdf/2309.07689.pdf)
    - Contains a summary of various methods and datasets used for AI text detection.
- [Differentiate ChatGPT-generated and Human-written Medical Texts](https://arxiv.org/pdf/2304.11567v1.pdf)
    - Use of NLTK for part-of-speech analysis.
    - Use of stanford-corenlp for dependency parsing.
    - Use of pre-trained sentiment analysis model.
    - Use of perplexity via BioGPT.
    - Use of vocabulary and sentece features - size of vocabulary, number of word stem, avg length of sentence, number of sentences.
- [Training own tokenizer](https://www.kaggle.com/code/datafan07/train-your-own-tokenizer)
    - In this competition it could make sense to train our own tokenizer, as the data seems to be artificially corrected (even the human-generated essays), which is most likely done to increase the difficulty of the task, as we could otherwise just count the typos and detect the AI text.
- [Tfidf vectorizer](https://www.kaggle.com/code/bhanupratapbiswas/detect-ai-gt-sub)
    - Use of tfidf vectorizer of sklearn library. This method is used to convert a collection of raw documents to a matrix of TF-IDF features. It is used to detect the most important words in a document. One can use these features to train a classifier e.g. XGBoost.

## Conclusion

In my work I will try to use these architectures:

- Debertav3 - encoder-based architecture.
- Mistral-7B with LoRA and quantization - decoder-based architecture.
- Tfidf vectorizer and a classifier. If I have time left I will also try to ise tricks used in [The Science of Detecting LLM-Generated Texts](https://arxiv.org/pdf/2303.07205.pdf) and [Differentiate ChatGPT-generated and Human-written Medical Texts](https://arxiv.org/pdf/2304.11567v1.pdf).
- Perplexity - if I have time left I will try this as well.

I will compare these methods and also try to ensemble them.