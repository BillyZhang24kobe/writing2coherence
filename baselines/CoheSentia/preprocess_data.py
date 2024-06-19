import os

import sys
import time
import datetime
import random
import argparse

import numpy as np
import pandas as pd
from typing import *
import json

from sklearn.model_selection import train_test_split


def split_data_train_dev(data: Dict[str, Any]):
    all_titles = [d['Title'] for k, d in data.items()]
    train_titles, dev_titles = train_test_split(all_titles, test_size=0.15)
    train_data = {k: d for k,d in data.items() if d['Title'] in train_titles}
    dev_data = {k: d for k,d in data.items() if d['Title'] in dev_titles}
    return train_data, dev_data


def get_final_score_dfs(data: Dict[str, Any]):
    titles, texts, holistic_scores, incremental_scores = [], [], [], []
    for story_id, story_data in data.items():
        titles.append(story_data['Title'])
        texts.append(story_data['Text'])
        holistic_scores.append(story_data['HolisticData']['consensus_score'])
        incremental_scores.append(story_data['IncrementalData']['consensus_score'])
    holistic_df = pd.DataFrame(list(zip(titles, texts, holistic_scores)),
                                columns =['title', 'text', 'label'])
    incremental_df = pd.DataFrame(list(zip(titles, texts, incremental_scores)),
                                columns =['title', 'text', 'label'])
    return holistic_df, incremental_df


def get_reason_group(reason: int) -> str:
    if reason in [1,2,3]: return 'cohesion'
    if reason in [4,5]: return 'consistency'
    if reason in [6,7]: return 'relevance'
    return 'other'


def get_per_sent_df(data: Dict[str, Any]) -> pd.DataFrame:
    titles, texts, all_sents = [], [], []
    coherences, cohesions, consistencies, relevances = [], [], [], []
    for story_id, story_data in data.items():
        titles.append(story_data['Title'])
        texts.append(story_data['Text'])
        inc_data = story_data['IncrementalData']
        sents = inc_data['sentences']
        all_sents.append(sents)
        reasons_data = inc_data['reasons']
        num_annot = inc_data['num_annotators']
        num_sents = len(sents)
        sents_data = {} # dict for the sentences in the story
        coherence_data, cohesion_data, consistency_data, relevance_data = {}, {}, {}, {}
        for annot_id, annot_data in reasons_data.items():
            for sent_i in range(num_sents):
                sents_data[sent_i] = {i: 0 for i in ['coherent', 'cohesion', 'consistency', 'relevance']}
                if str(sent_i) in annot_data:
                    sent_reasons = annot_data[str(sent_i)] # list of reasons this sent is not coherent 
                    for reason in sent_reasons:
                        group = get_reason_group(reason)
                        sents_data[sent_i]['coherent'] += 1
                        if group != 'other': sents_data[sent_i][group] += 1
                        
        for sent_i, sent_data in sents_data.items():
            if sent_data['coherent'] >= num_annot / 2:
                coherence_data['sent'+str(sent_i)] = True
            else:
                coherence_data['sent'+str(sent_i)] = False
            
            if sent_data['cohesion'] >= num_annot / 2:
                cohesion_data['sent'+str(sent_i)] = True
            else:
                cohesion_data['sent'+str(sent_i)] = False
            
            if sent_data['consistency'] >= num_annot / 2:
                consistency_data['sent'+str(sent_i)] = True
            else:
                consistency_data['sent'+str(sent_i)] = False
            
            if sent_data['relevance'] >= num_annot / 2:
                relevance_data['sent'+str(sent_i)] = True
            else:
                relevance_data['sent'+str(sent_i)] = False
        coherences.append(coherence_data)
        cohesions.append(cohesion_data)
        consistencies.append(consistency_data)
        relevances.append(relevance_data)
    
    per_sents_df = pd.DataFrame(list(zip(titles, texts, all_sents, coherences, 
                                        cohesions, consistencies, relevances)),
                                columns =['title', 'text', 'sents', 'coherence_per_sent', 
                                'cohesion_per_sent', 'consistency_per_sent', 'relevance_per_sent'])
    return per_sents_df


def create_df(data: Dict[str, Any]): 
    holistic_df, incremental_df = get_final_score_dfs(data)
    sent_df = get_per_sent_df(data)
    return holistic_df, incremental_df, sent_df


def save_df(df:pd.DataFrame, output_name:str) -> None:
    df.to_csv(output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Coherence')
    ## Required parameters 
    parser.add_argument("--output_dir", default="data/", type=str, help="where to save the data")
    parser.add_argument("--data_dir", default="data/", type=str, help="where is the data")

    args = parser.parse_args()

    f = open (args.data_dir + 'TrainData.json', "r")
    data = json.loads(f.read())
    f.close()
    train_data, dev_data = split_data_train_dev(data)
    holistic_df, incremental_df, sent_df = create_df(train_data)
    save_df(holistic_df, args.output_dir + 'train_holistic.csv')
    save_df(incremental_df, args.output_dir + 'train_incremental.csv')
    save_df(sent_df, args.output_dir + 'train_per_sent.csv')

    holistic_df, incremental_df, sent_df = create_df(dev_data)
    save_df(holistic_df, args.output_dir + 'dev_holistic.csv')
    save_df(incremental_df, args.output_dir + 'dev_incremental.csv')
    save_df(sent_df, args.output_dir + 'dev_per_sent.csv')
    
    f = open (args.data_dir + 'TrainData.json', "r")
    test_data  = json.loads(f.read())
    f.close()
    holistic_df, incremental_df, sent_df = create_df(test_data)
    save_df(holistic_df, args.output_dir + 'test_holistic.csv')
    save_df(incremental_df, args.output_dir + 'test_incremental.csv')
    save_df(sent_df, args.output_dir + 'test_per_sent.csv')

    

