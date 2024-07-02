#!/usr/bin/env bash

python main.py --model_name='google-bert/bert-base-uncased' --data_dir data_cohesion/ --output_dir output_cohesion/ --classification_label sent_cohesion
python main.py --model_name='google-bert/bert-base-uncased' --data_dir data_consistency/ --output_dir output_consistency/ --classification_label sent_consistency
python main.py --model_name='google-bert/bert-base-uncased' --data_dir data_relevance/ --output_dir output_relevance/ --classification_label sent_relevance
python main.py --model_name='google-bert/bert-base-uncased' --data_dir data_other/ --output_dir output_other/ --classification_label sent_binary

python main.py --model_name='google-bert/bert-base-uncased' --data_dir data_cohesion_dc/ --output_dir output_cohesion_dc/ --classification_label sent_cohesion
python main.py --model_name='google-bert/bert-base-uncased' --data_dir data_consistency_dc/ --output_dir output_consistency_dc/ --classification_label sent_consistency
python main.py --model_name='google-bert/bert-base-uncased' --data_dir data_relevance_dc/ --output_dir output_relevance_dc/ --classification_label sent_relevance

python main.py --model_name='google-bert/bert-base-uncased' --data_dir data_coherence_combined/ --output_dir output_coherence_combined/ --classification_label sent_binary
python main.py --model_name='google-bert/bert-base-uncased' --data_dir data_cohesion_combined/ --output_dir output_cohesion_combined/ --classification_label sent_cohesion
python main.py --model_name='google-bert/bert-base-uncased' --data_dir data_consistency_combined/ --output_dir output_consistency_combined/ --classification_label sent_consistency
python main.py --model_name='google-bert/bert-base-uncased' --data_dir data_relevance_combined/ --output_dir output_relevance_combined/ --classification_label sent_relevance


python main.py --model_name='google-bert/bert-large-uncased' --data_dir data_cohesion/ --output_dir output_cohesion/ --classification_label sent_cohesion
python main.py --model_name='google-bert/bert-large-uncased' --data_dir data_consistency/ --output_dir output_consistency/ --classification_label sent_consistency
python main.py --model_name='google-bert/bert-large-uncased' --data_dir data_relevance/ --output_dir output_relevance/ --classification_label sent_relevance
python main.py --model_name='google-bert/bert-large-uncased' --data_dir data_other/ --output_dir output_other/ --classification_label sent_binary

python main.py --model_name='google-bert/bert-large-uncased' --data_dir data_cohesion_dc/ --output_dir output_cohesion_dc/ --classification_label sent_cohesion
python main.py --model_name='google-bert/bert-large-uncased' --data_dir data_consistency_dc/ --output_dir output_consistency_dc/ --classification_label sent_consistency
python main.py --model_name='google-bert/bert-large-uncased' --data_dir data_relevance_dc/ --output_dir output_relevance_dc/ --classification_label sent_relevance

python main.py --model_name='google-bert/bert-large-uncased' --data_dir data_coherence_combined/ --output_dir output_coherence_combined/ --classification_label sent_binary
python main.py --model_name='google-bert/bert-large-uncased' --data_dir data_cohesion_combined/ --output_dir output_cohesion_combined/ --classification_label sent_cohesion
python main.py --model_name='google-bert/bert-large-uncased' --data_dir data_consistency_combined/ --output_dir output_consistency_combined/ --classification_label sent_consistency
python main.py --model_name='google-bert/bert-large-uncased' --data_dir data_relevance_combined/ --output_dir output_relevance_combined/ --classification_label sent_relevance


python get_results.py --model_name='google-bert/bert-base-uncased' --data_path output_cohesion/ --coherence_type sent_cohesion
python get_results.py --model_name='google-bert/bert-base-uncased' --data_path output_consistency/ --coherence_type sent_consistency
python get_results.py --model_name='google-bert/bert-base-uncased' --data_path output_relevance/ --coherence_type sent_relevance
python get_results.py --model_name='google-bert/bert-base-uncased' --data_path output_other/ --coherence_type sent_binary

python get_results.py --model_name='google-bert/bert-base-uncased' --data_path output_cohesion_dc/ --coherence_type sent_cohesion
python get_results.py --model_name='google-bert/bert-base-uncased' --data_path output_consistency_dc/ --coherence_type sent_consistency
python get_results.py --model_name='google-bert/bert-base-uncased' --data_path output_relevance_dc/ --coherence_type sent_relevance

python get_results.py --model_name='google-bert/bert-base-uncased' --data_path output_coherence_combined/ --coherence_type sent_binary
python get_results.py --model_name='google-bert/bert-base-uncased' --data_path output_cohesion_combined/ --coherence_type sent_cohesion
python get_results.py --model_name='google-bert/bert-base-uncased' --data_path output_consistency_combined/ --coherence_type sent_consistency
python get_results.py --model_name='google-bert/bert-base-uncased' --data_path output_relevance_combined/ --coherence_type sent_relevance


python get_results.py --model_name='google-bert/bert-large-uncased' --data_path output_cohesion/ --coherence_type sent_cohesion
python get_results.py --model_name='google-bert/bert-large-uncased' --data_path output_consistency/ --coherence_type sent_consistency
python get_results.py --model_name='google-bert/bert-large-uncased' --data_path output_relevance/ --coherence_type sent_relevance
python get_results.py --model_name='google-bert/bert-large-uncased' --data_path output_other/ --coherence_type sent_binary

python get_results.py --model_name='google-bert/bert-large-uncased' --data_path output_cohesion_dc/ --coherence_type sent_cohesion
python get_results.py --model_name='google-bert/bert-large-uncased' --data_path output_consistency_dc/ --coherence_type sent_consistency
python get_results.py --model_name='google-bert/bert-large-uncased' --data_path output_relevance_dc/ --coherence_type sent_relevance

python get_results.py --model_name='google-bert/bert-large-uncased' --data_path output_coherence_combined/ --coherence_type sent_binary
python get_results.py --model_name='google-bert/bert-large-uncased' --data_path output_cohesion_combined/ --coherence_type sent_cohesion
python get_results.py --model_name='google-bert/bert-large-uncased' --data_path output_consistency_combined/ --coherence_type sent_consistency
python get_results.py --model_name='google-bert/bert-large-uncased' --data_path output_relevance_combined/ --coherence_type sent_relevance