# 🎬 IMDb Sentiment Analysis with BERT and NLP

![](report/images/wordcloud.png)

**Python Version:** 3.11  
**Transformers Version:** Hugging Face Transformers  
**Frameworks Used:** PyTorch, scikit-learn, NLTK, Matplotlib, Seaborn

---

## About the Project

About the Project
This project explores sentiment analysis using the IMDb dataset of 50,000 movie reviews, aiming not only to build a high-performing classification model, but also to evaluate its behavioral robustness through structured perturbation testing.

We fine-tune BERT (bert-base-uncased) for binary sentiment classification and go beyond standard accuracy metrics by integrating CheckList, a behavioral testing framework designed to expose weaknesses in NLP models.

Specifically, we implement two CheckList test types:

Invariance (INV): Named Entities (e.g., people, places, organizations) are substituted with similar entities. A robust model should produce consistent sentiment predictions if the entity change doesn't affect textual sentiment.

Directional Expectation (DIR): Sentiment-rich modifiers (e.g., emotionally loaded adjectives/adverbs) are added to test if the model's confidence in sentiment predictions increases appropriately in line with human expectations.

This dual approach allows us to inspect how well the model generalizes beyond raw accuracy—examining whether predictions shift irrationally with irrelevant changes (INV), or respond appropriately to emotional cues (DIR).

Additionally, we perform cross-domain evaluation, using a cleaned and processed Yelp dataset as out-of-domain test data. This highlights how well the model generalizes to real-world data it wasn't trained on.

A full research paper accompanies this repository, detailing the motivation, methodology, experimental design, and findings. The project demonstrates the importance of evaluating NLP models not just by what they get right, but by understanding why they succeed or fail.

---

## Dataset

The dataset used is the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle.  
It contains 50,000 labeled movie reviews split evenly between positive and negative sentiments.

To use the dataset:
1. Download from the [Kaggle link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2. Place the CSV in the `data/` directory as `raw_imdb_dataset.csv`.

---
## Workflow of the github and notebooks:

1. Training the BERT NER tagging model ()
2. Cleaning the raw IMDB dataset
3. Using the trained BERT model to tag IMDB Data
4. Fine tuning pretrained sentiment model from hugging face with 90% of cleaned IMDB data
5. Clean and tag yelp data same way of IMDB
6. Use cleaned YELP data as test data for the fine tuned model
7. Perturb cleaned IMDB data and use for sentiment analysis test
8. Test remaining 10% of cleaned IMDB data, perturbed IMDB data


---

## How to run the code and reproduce results

To run the code first you will have to download and install the 'requirements.txt' found in the "data" folder. There was both used .py scrips as well as notebooks to execute our code. Notebooks were primarily used for simple tasks such as EDA and cleaning, where ask the scripts were used during HPC trianing of our models. 

### 1. Project baseline model / BERT NER tagging model:

We have trained the baseline model using the dataset ["en_ewt-ud-test.iob2"](https://learnit.itu.dk/pluginfile.php/418423/mod_resource/content/1/en_ewt-ud-test.iob2) . The trained model parameters can be found in our sharepoint [HERE](https://ituniversity.sharepoint.com/:f:/s/NLP572/EmPch8O89UtCgIcIPi6vapMBxB_O0rVGicvcW1p6u64x0A?e=kRt6Qf) for replicating the NER tagging process for our data.

### 2. EDA and cleaning data

To run and replicate the cleaned data set that was used for further analysis, you can simply run the notebook "eda_clean_imdb". (Output of the cleaned dataset code block has been frozen in case the outcome will be diffrent, even thought no randomisation was used in the process.)










---


