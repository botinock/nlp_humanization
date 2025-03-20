# nlp_humanization
Test task for Boosters

https://earthy-windshield-303.notion.site/Test-case-for-NLP-Engineer-JustDone-ai-22fe02e514424d5abc8b1fa5cd86f6a5#800e4b60a6a340e4bb721d03f3b26973

## Requirements
- Git
- Docker
## Installation
- ```git pull https://github.com/botinock/nlp_humanization.git```
- ```docker compose up -d```
- Attach IDE or shell to the container
## Training
```python3 train.py```
## Metrics 
```tensorboard --logdir runs/```
## Problem statement
The task is to transform LLM-generated text into a human-like style that is basically a Text Style Transfer NLP task. The main goal is to rewrite LLM-generated text in an indistinguishable manner from human texts **for AI detectors** while preserving the general idea of the text. 

## Exploratory data analysis
The Dataset contains 3000 pairs of parallel data of human and LLM texts. The texts have a markdown.

![image](https://github.com/user-attachments/assets/add8b2a0-ff56-46e5-a298-e077817119df)

---
Disclaimer: Despite it seems like both texts are AI-generated, and in a majority of cases the human ones are more likely AI-generated, I assume 'human' texts are ground truth.

![image](https://github.com/user-attachments/assets/2b645462-5971-4ac2-abe1-d48cf3b843ce)

---

Human texts are more varied in length, total and unique word counts. At the same time, LLM texts tend to have longer sentences and words. Also, LLM texts have more unique words in general and they're longer a little bit.

![image](https://github.com/user-attachments/assets/206bdb4e-8037-4322-a619-5e52fda53a8c)
![image](https://github.com/user-attachments/assets/08b105bf-681b-4069-b62d-12b313253bea)
![image](https://github.com/user-attachments/assets/81559909-5c55-4cbd-b36d-925f6d9b187e)
![image](https://github.com/user-attachments/assets/1e1c6248-53a2-4bfe-980b-ffe4a30f47e8)
![image](https://github.com/user-attachments/assets/383bf277-af35-4ce8-9e66-faa767cbd78b)

---
The human texts generally contain citations while LLM texts are kind of more imperative, confident and provide subjective statements.
For example, texts have different n-grams: 

Human: Et al, say, analyst say, chief financial/executive officer.
LLM: play crucial/critical/significant/pivotal role, state-of-the-art

![image](https://github.com/user-attachments/assets/aa92b325-0c4f-4087-8117-73f924d3ae50)
![image](https://github.com/user-attachments/assets/f4421769-938c-4d2b-af26-6863c0c1299a)
![image](https://github.com/user-attachments/assets/6579d5e2-fd17-46cf-b666-2a9e9677ba3a)

---

Both texts are similar in terms of sentiment or polarity, they are rather neutral-positive. But LLM texts tend to be significantly easier to understand than human ones.

![image](https://github.com/user-attachments/assets/4c96a25a-743c-49c6-bd64-587ddd2eaf39)
![image](https://github.com/user-attachments/assets/4ea7bd9d-3b8d-4a05-b3da-4bce032fddbf)

---

The pairs of texts are very different structurally (low ROUGE and BLEU scores) but quite similar semantically (high BERT Score and SBERT embeddings cosine similarity)

## Approach
The task generally can be achieved with different approaches such as:
- Custom seq2seq models
- LLM with the appropriate prompt
- Heuristics
- Combinations of all the above

I will stick to Custom seq2seq model training.

Also, depending on the number and nature of the present data, seq2seq model can trained in different ways.

As we have a small amount of data, but the data is parallel (paired text) the first thought is to fine-tune seq2seq model.
I've tried T5 and BART models, and for some reason, T5 with a frozen encoder was performing the best. Unfortunately, I did not have enough time to test all the ideas.

## Evaluation Metrics
Complete:

- ROUGE-1, ROUGE-2, ROUGE-L: generated-human. Supervised metric.
- BLEU F!: generated-human. Supervised metric.
- BERT Score: generated-LLM; generated-human. Semantic similarity.
- Embeddings Cosine Similarity: generated-LLM; generated-human. Semantic similarity.

Incomplete:

- Supervised human-likeness (style) classification model. generated.
- Length normalization to control generated text length

## More ideas
### Models
- Different and more recent architectures
- Larger models
- Variational autoencoders
- Adversarial or diffusion models
### Custom loss
- Supervised human-likeness (style) classification loss.
- Embedding similarity loss. The generated text should be semantically close to the source (LLM) texts
- Contrastive or Triplet loss for better embeddings and classification
### Advanced Techniques
- Different decoding algorithms to prevent over-confidence or adjust randomness (Nucleus Sampling, Contrastive Decoding)
- RL approaches such as RLHF and more 
- RAG to expand topic contexts
- LoRA + Adapters to more efficient fine-tuning without overfit
- Human in the loop 
## QA
* - Q: How can the chosen NLP model architecture and training process be adjusted to minimize the probability of hallucinations?
  - A: Embedding similarity loss. Contrastive or Triplet loss for better embeddings and classification. LoRA for more stability. RL and KL Divergence. RAG.
  - Additional Resources: https://github.com/EdinburghNLP/awesome-hallucination-detection
* - Q: Describe the integration of the updated architecture and parameters into the existing model training functionality.
  - A: Modify training loop for custom loss calculations. Use peft lib for LoRA and other fine tuning algorithms. Quantify and prune the model for inference.
  - Additional Resources: https://huggingface.co/docs/peft/package_reference/lora
* - Q: How will you set up the system to monitor and track the occurrence of hallucinations? Describe the main mechanism, components, and the way of integration.
  - A: Embeddings similarity (BertScore, Cosine similarity). Model based hallucination detection methods (using llm). Detect new named entities. Human evaluation. Low confidence generations. Metrics such as ROUGE and BLEU are performing badly because of significant text transformation. Integrate into evaluation loop and log into tensorboard. Some metrics are very expensive to compute, so it does not make sense to compute it each epoche on the small dataset like this.
* - Assume you have completed the training using the provided code and monitored the process. Using the plot as a reference, brainstorm and answer the following questions.
  ![image](https://github.com/user-attachments/assets/5a768c00-e176-403c-8066-474db2889aad)
* - Q: Which problems with the training can you highlight according to the presented losses?
  - A: The more likely overfit is present. The presence of the overfit can be spotted on epoch 3-5. But on the epoch 50 it's clearly that the learning rate is so high so model is jumped far over local optima. Also, seems like the model can't generalize validation set at all, but anyway it can be due to high learning rate.
* - Q: What can be changed in the training process to tackle highlighted issues?
  - A: Lower learning rate first of all. Try to freeze weights or use advanced techniques like LoRA. Try additional intrinsic losses or regularization.
* - Q: How can you explain the unexpected fluctuation in the training loss around epoch 50?
  - A: Learning rate is too high
* - Q: How can you explain the different ranges of the training/validation losses?
  - A: Bad generalization and overfit.
* - Q: How can we minimize the probability of the presented situation from the very beginning?
  - A: Try lower learning rate, freeze weights while fine-tuning. Check out if dataset is prepared properly.
 
