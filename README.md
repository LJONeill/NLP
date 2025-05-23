# What insights can CheckList provide about the cross-domain robustness of sentiment classifiers?

![](report/images/wordcloud.png)


## Introduction

This GitHub repository represents the coding part our NLP project focused on leveraging the CheckList evaluation method in combination with Named Entity Recognition (NER) tagging to investigate sentiment classification models. The core aim is to analyze whether sentiment models are influenced by the presence of named entities—even though such information should ideally not affect sentiment predictions. Our work examines potential biases, cross-domain robustness, and the underlying behavior of sentiment classifiers.

## Datasets

### imdb

We used the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) released by Maas et al. (2011) from the Stanford AI Lab. It contains 50,000 highly polar movie reviews.

### Yelp

We used the [Yelp Restaurant Reviews Dataset](https://www.kaggle.com/datasets/farukalam/yelp-restaurant-reviews) published by Faruk Alam on Kaggle. The dataset consists of restaurant reviews paired with 1–5 star ratings. While it does not include explicit sentiment labels, the star ratings were used to infer sentiments.

## workflow outline

Below is a brief outline of our workflow and the sequence in which the notebooks and Python scripts are intended to be used:

1. Train Baseline BERT NER Model
2. Clean and Analyze IMDB Dataset
3. Clean and Analyze Yelp Dataset
4. Apply NER Tagging to IMDB Data
5. Fine-Tune BERT Sentiment Classifier using and tag imdb data using HPC
6. Generate Perturbations with CheckList
7. Evaluate Cross-Domain and Perturbed Performance
8. Conduct Accuracy and Robustness Analysis

## how to reproduce our results

To ensure reproducibility, all core functionality has been implemented in Jupyter notebooks, whereas the training and fine-tuning of the models were conducted using HPC (High-Performance Computing). To preserve data integrity and ensure consistency across notebooks and scripts, we have commented out the cells responsible for generating or overwriting data. This avoids discrepancies in intermediate datasets. Most of the key results are already stored within the notebooks, so re-running all code is not necessary unless desired. A `requirements.txt` file is included to support clean environment setup if needed.

Our final results can be found in the notebook `accuracy_analyses.ipynb`, located under the `notebooks/imdb` directory. For a step-by-step guide to recreate the datasets and results, please refer to the workflow outlined above.


## Disclaimers and aknlowledgements

We acknowledge that generative AI tools were used throughout this project—for code assistance and help understanding external implementations. We also drew inspiration from publicly available models and solutions hosted on Hugging Face and Kaggle.

## Sources

### Articles

- Shnayderman, N., Elazar, Y., Goldberg, Y., & Dagan, I. (2021). *Exploring the Efficacy of Automatically Generated Counterfactuals for Sentiment Analysis*. arXiv preprint [arXiv:2106.15231](https://arxiv.org/pdf/2106.15231)

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. In *Proceedings of NAACL-HLT* (pp. 4171–4186). [PDF](https://aclanthology.org/N19-1423.pdf)

- Wu, T., Ribeiro, M. T., & Singh, S. (2022). *CheckList Test Suites for Evaluating Model Generalization*. In *Proceedings of ACL 2022*. [PDF](https://aclanthology.org/2022.acl-long.577.pdf)

- Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*. In *Proceedings of ACL 2020*. [PDF](https://aclanthology.org/2020.acl-main.442.pdf)

- Wu, T., Ribeiro, M. T., & Singh, S. (2021). *CheckList from an Outsider’s Perspective*. In *WOAH 2021*. [PDF](https://aclanthology.org/2021.woah-1.9.pdf)

- *Adapted CheckList for Image Generation*. In *EMNLP Demos 2022*. [PDF](https://aclanthology.org/2022.emnlp-demos.4.pdf)

- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis*. In *ACL 2011*. [PDF](https://aclanthology.org/P11-1015.pdf)

- Anonymous (2021). *Model Development with Counterfactuals*. In *OpenReview*. [PDF](https://openreview.net/pdf?id=B329drNt9Dj)

- *Sentiment Analysis Overview*. [PDF](https://aclanthology.org/N18-1171.pdf)

- *The Importance of Sentiment Analysis*. [Article](https://journal.arrus.id/index.php/soshum/article/view/1992/1297)

- *Task-Specific Models with Less Data*. In *Findings of ACL 2023*. [PDF](https://aclanthology.org/2023.findings-acl.507.pdf)

### Dataset

#### IMDB

[Large Movie Review Dataset v1.0](https://ai.stanford.edu/~amaas/data/sentiment/)  
> Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis*. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

#### Yelp

[Yelp Restaurant Reviews Dataset](https://www.kaggle.com/datasets/farukalam/yelp-restaurant-reviews) by Faruk Alam on Kaggle

### Kaggle

- [Text Preprocessing | NLP | Steps to Process Text](https://www.kaggle.com/code/abdmental01/text-preprocessing-nlp-steps-to-process-text) by Sheikh Muhammad Abdullah  
- [NLP - Data Preprocessing and Cleaning](https://www.kaggle.com/code/colearninglounge/nlp-data-preprocessing-and-cleaning) by Co-learning Lounge  
- [Sentiment Analysis of IMDB Movie Reviews](https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/notebook) by Lakshmi Narayan

### Hugging Face

- [Original BERT model (bert-base-uncased)](https://huggingface.co/bert-base-uncased) Pretrained sentiment classification models (referenced in our fine-tuning)
