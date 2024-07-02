import logging
import datasets

from get_dataset_for_bert import load_and_cache_coherence_dataset_into_pairs as load_and_cache_coherence_dataset_into_pairs_bert
from get_dataset_for_bert import load_data_for_mtl_model as load_data_for_mtl_model_bert
from get_dataset_for_t5 import load_and_cache_coherence_dataset_into_pairs as load_and_cache_coherence_dataset_into_pairs_t5
from get_dataset_for_gpt import load_and_cache_coherence_dataset_into_pairs as load_and_cache_coherence_dataset_into_pairs_gpt


logger = logging.getLogger(__name__)


def load_datasets(tokenizer, data_args, training_args, model_args):
    mtl_model = data_args.dataset_type == 'mtl'
    if mtl_model: 
        train_dataset, validation_dataset, test_dataset, tasks_info = load_data_for_mtl_model_bert(
            data_args, model_args.encoder_name_or_path)
    else:
        tasks_info = []
        if 't5' in model_args.encoder_name_or_path:
            train_dataset, validation_dataset, test_dataset, num_labels = load_and_cache_coherence_dataset_into_pairs_t5(
                        data_args=data_args, training_args=training_args,
                        tokenizer=tokenizer, model_name=model_args.encoder_name_or_path)
        elif 'gpt' in model_args.encoder_name_or_path:
            train_dataset, validation_dataset, test_dataset, num_labels = load_and_cache_coherence_dataset_into_pairs_gpt(
                        data_args=data_args, training_args=training_args,
                        tokenizer=tokenizer, model_name=model_args.encoder_name_or_path)
        else:
            train_dataset, validation_dataset, test_dataset, num_labels = load_and_cache_coherence_dataset_into_pairs_bert(
                        data_args=data_args, training_args=training_args,
                        tokenizer=tokenizer, model_name=model_args.encoder_name_or_path)
            
    train_dataset.shuffle(seed=123)
    validation_dataset.shuffle(seed=123)

    dataset = datasets.DatasetDict(
        {"train": train_dataset, "validation": validation_dataset ,"test": test_dataset}
    )
    
    return dataset, tasks_info



