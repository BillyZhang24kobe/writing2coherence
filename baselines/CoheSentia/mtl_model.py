from typing import *
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers import PreTrainedModel, PretrainedConfig

import torch
from binary_sequence_classification import SequenceClassificationHead
from pooler import Pooler
from utils import Task


class MultiTaskModel(PreTrainedModel):
    def __init__(self, encoder_name_or_path: str, tasks: List[Task], config: PretrainedConfig, model_type: str = "bert", encoder_gradient_checkpointing: bool = False):
    # def __init__(self, encoder_name_or_path: str, tasks: List[Task], config_dict: Dict[str, Any], model_type: str = "bert", coherence_add_encoder: bool = False):
        super().__init__(config)
        self.base_model_type = model_type
        # self.encoder = AutoModelForSequenceClassification.from_pretrained(encoder_name_or_path)
        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        if encoder_gradient_checkpointing:
            self.encoder.config.gradient_checkpointing = True
            self.encoder.gradient_checkpointing_enable()
        self.config = config
        self.output_heads = nn.ModuleDict()
        if len(tasks) == 0:
            tasks = [Task(id=0, name='all', num_labels=config.num_labels)]
        for task in tasks:
            decoder = SequenceClassificationHead(self.encoder.config.hidden_size, task.num_labels, config)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.id)] = decoder
        
        if model_type in ["deberta_v2", "deberta_v3"]:
            self.pooler = Pooler(config=config)
        
        # if coherence_add_encoder:
            # TODO: add encoder on top - input is a sequence_output 
            # pass

    @staticmethod
    def _create_output_head(encoder_hidden_size: int, task: Task, config):
        return SequenceClassificationHead(encoder_hidden_size, task.num_labels, config)
        
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            task_ids=None,
            metadata=None,
            **kwargs,
    ):
        """ final_outpus is a dict with keys: 
        1. loss: <torch.float>
        2. logitc: <batch_size, num_label>
        
        for coreference task there are more keys:
            3. predicted_prepositions - <1, batch_size> - the predicted label
            4. best_prepositions -  <1, batch_size> - the predicted label out from all existing example between the pair of nps
            5. logitc_best_preposition - <batch_size, num_label - per task> - the logtic for best_prepositions
        in addition when the output from the encoder doesnt contain only polled_output and sequence_output 
        - the rest of the encoder output is put under the key "extra_items_from_encoder"

        all_labels will be from 3 metrices: 
        1. labels - <batch_size> - what is the real label 
        for coreference task there are 2 more labels:
            2. link_labels - <batch_size> - what is the most common label for this np relation 
            3. extended_preposition_labels - <batch_size, 1, num_labels> - what is the probability for each label between this pair of nps 
        """
        # print(f"start task_ids is {task_ids}")
        # print(f"before squeeze size of input_ids is {input_ids.size()}")
        # shape of input_ids for all tasks: <batch_size, 1, max_seq_len>
        # input_ids=torch.squeeze(input_ids, dim=1)
        # print(f"size of input_ids is {input_ids.size()}")
        # attention_mask = torch.squeeze(attention_mask, dim=1)

        # if token_type_ids is not None:
        #     token_type_ids = torch.squeeze(token_type_ids, dim=1)
    
        if self.base_model_type in ["bert_base", "bert_large"]:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        elif self.base_model_type in ["deberta_v2", "deberta_v3"]:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )
        else:  # self.base_model_type in ["t5"]
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        # sequence_output = [batch_size, seq_size, embed_dim]: for each example, for each token, its embedding result
        # pooled_output = [batch_size, embed_dim]: for each example, the CLS token
        if "pooler_output" not in list(outputs.keys()): # in this case i dont get output.pooled_output so i need to create is using self.pooler
            sequence_output = outputs[0]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        else:
            sequence_output, pooled_output = outputs[:2]

        if task_ids is None:
            task_ids = torch.zeros(len(input_ids))
        unique_task_ids_list = torch.unique(task_ids).tolist()
        # print(f"task ids for this batch are {unique_task_ids_list}")

        loss_list = []
        logits = None # <num_examples, num_labels>
        # filtered using TaskId of each task and fed to the appropriate decoder (output_heads)
        for i, unique_task_id in enumerate(unique_task_ids_list):
            # print(f"now decoder for task {unique_task_id}")
            task_id_filter = task_ids == unique_task_id
            output_dict = self.output_heads[str(int(unique_task_id))].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )
            logits = output_dict["logitc"]
            if i == 0:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits), dim=0)
            if labels is not None:
                task_loss = output_dict["loss"]
                loss_list.append(task_loss)

        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (all_logits, outputs[2:]) #- check version
        # outputs = (logits, outputs[2:]) #- check version

        if loss_list:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs  # check version
        
        final_outputs = {}
        for k, v in output_dict.items():
            if k == "loss":
                if labels is not None:
                    final_outputs["loss"] = loss.mean()
            else:
                final_outputs[k] = v
        if len(outputs[2:])>0:
            final_outputs['extra_items_from_encoder'] = outputs[2:]

        # return final_outputs
        return outputs
