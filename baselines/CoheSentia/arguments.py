import logging
from typing import * 
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)



@dataclass
class DataArguments:
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the data downloaded from huggingface.co"},
    )
    label_all_tokens: bool = field(
        default=False, metadata={"help": "Whether in tokens it will label all tokens"}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    data_dir: Optional[str] = field(
        default='data/',
        metadata={"help": "A path to save cached data"})
    
    output_dir: Optional[str] = field(
        default='output/',
        metadata={"help": "A path to save cached data"})
    
    dataset_type: Optional[str] = field(
        default='combined',
        metadata={"help": "type of coherence datasset to train"})
    
    allow_loading: bool = field(
        default=False,
        metadata={"help": "Whether to alllow to load the saved data and save it in the first place"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    num_labels: int = field(
        default=5, 
        metadata={"help": "number of labels to use in generation"},
    )
    num_beams: int = field(
        default=5, 
        metadata={"help": "number of beams to use in generation"},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    encoder_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    t5_use_decoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use t5 decoder or to classify the encoder output myself. in case i it is false, during data processing the labels for bert model will be saved as well"}
    )

