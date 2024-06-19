# CoheSentia: 

This repository present the data for coherence evaluation as well as code for finetuning base models on it.


## Data
The Data is under 'data' folder with explaination about its format. 
The data is in 'data' folder and is in 3 files for different splits: train, dev and test.

The data format:

* holistic or incremental final score data should be named: "<train/dev/test>_<holistic/incremental>.csv"
the csv file has 3 columns: 'title', 'text', 'label':
    * "title" - string with the stroy title 
    * "text" - string with the text
    *"label" - int with the consensus score
* sentence level data should be names: "<train/dev/test>_per_sent.csv"
the csv file has those columns:
    *"title" - string with the stroy title
    * "text" - string with the text
    * "sents" - the text splitted into list 
    * "coherence_per_sent" - dictionary with sentences id as key and True/False if it is incoherent 
    * "cohesion_per_sent" - dictionary with sentences id as key and True/False if it is incohesive
    * "consistency_per_sent" - dictionary with sentences id as key and True/False if it is inconsistent
    * "relevance_per_sent" - dictionary with sentences id as key and True/False if it is irrelevant


## Getting started
install dependencies 
conda create -n cohesentia python 3.9 anaconda 
conda activate cohesentia

pip install -r requirements.txt


## Important arguments: 
* model_name: model name
* coherence_type: "incremental" or "holistic" based on method
* classification_label: 
    * "score" - final paragraph coherence score
    * "sent_binary" - per sentence coherence detection
    * "sent_cohesion" - per sentence cohesion detection
    * "sent_consistency" - per sentence consistency detection
    * "sent_relevance" - per sentence relevance detection
* only_prediction: True for zero-shot


## Preprocess data
for the data to change from the given json format you can run: 
python preprocess_data.py 


## Finetune
In order to finetune models run: 
main.py

In order to finetune models from openai run: 
1. python main.py with --model_name="gpt" and the other wanted parameters
2. python main_gpt.py with the wanted parameters

### Output
The output will be in 'output' folder in a "predict_results.txt" file 

## Citiation

