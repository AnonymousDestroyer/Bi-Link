from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
from transformers.models.roberta import RobertaModel, RobertaForMaskedLM, RobertaForTokenClassification
from transformers.models.bert import BertModel
from dict_hub import get_tokenizer
from triplet_mask import construct_mask
import torch.nn.functional as F

def build_model(args) -> nn.Module:
    return BiLinker(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor

@dataclass
class SiameseOutput:
    logit_head: torch.tensor
    logit_tail: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_tr_s: torch.tensor
    hr_vector: torch.tensor
    head_vector: torch.tensor
    tr_vector: torch.tensor
    tail_vector: torch.tensor

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.latent_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)      # activation after cls pooling

        return x


class BiLinker(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.config.latent_dim = self.args.latent_dim
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=False)    # temperature 20*sim learnable or not
        # contrastive between hr tr
        self.hr_tr_s = torch.nn.Parameter(torch.tensor(15.0), requires_grad=True)       # comparison between hr and tr are less impt
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        self.tokenizer = get_tokenizer()
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        model = AutoModel.from_pretrained(args.pretrained_model, add_pooling_layer=False)
        if self.args.freeze:
            freezed_modules = [model.encoder.layer[:self.args.freezed_layers]]
            for module in freezed_modules:
                for param in module.parameters():
                    param.requires_grad = False

        self.head_bert = model
        self.tail_bert = deepcopy(self.head_bert)

    @torch.no_grad()
    def predict_ent_embedding(self, head_batch_dict={}, tail_batch_dict={}, direction="forward") -> dict:
        if direction == "forward":
            output_batch_dict = self.tail_bert(**tail_batch_dict,return_dict=True)
            return output_batch_dict.last_hidden_state.mean(1)
        elif direction == 'backward':
            output_batch_dict = self.head_bert(**head_batch_dict, return_dict=True)
            return output_batch_dict.last_hidden_state.mean(1)
        else:
            raise NotImplementedError

    def _ent_rel_encode(self, encoder, ent_rel_batch_dict, ent_batch_dict):

        outputs_ent_rel = encoder(**ent_rel_batch_dict, return_dict=True)
        outputs_ent = encoder(**ent_batch_dict, return_dict=True)
        ent_rel_last_hidden_state = outputs_ent_rel.last_hidden_state
        ent_last_hidden_state = outputs_ent.last_hidden_state
          
        ent_rel_pooled = _pool_output(self.args.pooling, ent_rel_batch_dict['attention_mask'], ent_rel_last_hidden_state)
        ent_pooled = _pool_output(self.args.pooling, ent_batch_dict['attention_mask'], ent_last_hidden_state)
        return {
            "ent_rel_pooled": ent_rel_pooled, 
            "ent_rel_last": ent_rel_last_hidden_state,
            "ent_pooled": ent_pooled,
            "ent_last": ent_last_hidden_state
        }


    def forward(self, hr_batch_dict, head_batch_dict,
                tr_batch_dict, tail_batch_dict,
                template=None, left_template_position_ids=None, right_template_position_ids=None,
                only_ent_embedding=False, direction="forward", **kwargs) -> dict:

        # self._momentum_update_tail_encoder()

        if only_ent_embedding:
            return self.predict_ent_embedding(head_batch_dict=head_batch_dict, tail_batch_dict=tail_batch_dict, direction=direction)

        head_output_dict = self._ent_rel_encode(self.head_bert, hr_batch_dict, head_batch_dict)
        hr_vector, head_vector = head_output_dict['ent_rel_pooled'], head_output_dict['ent_pooled']

        tail_output_dict = self._ent_rel_encode(self.tail_bert, tr_batch_dict, tail_batch_dict)
        tr_vector, tail_vector = tail_output_dict['ent_rel_pooled'], tail_output_dict['ent_pooled']

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tr_vector': tr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector,
                }

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> SiameseOutput:
        head_vector, tail_vector = output_dict['head_vector'], output_dict['tail_vector']
        hr_vector, tr_vector = output_dict['hr_vector'], output_dict['tr_vector']

        batch_size = head_vector.size(0)
        labels = torch.arange(batch_size).to(head_vector.device)

        logit_head = hr_vector.mm(tail_vector.t())
        logit_tail = tr_vector.mm(head_vector.t())
        # this term avoids mode collapse
        # logit_off_diag_hr_tr = (hr_vector.mm(tr_vector.t())).fill_diagonal_(0)    

        if self.training:
            logit_head -= torch.zeros(logit_head.size()).fill_diagonal_(self.add_margin).to(logit_head.device)
            logit_tail -= torch.zeros(logit_tail.size()).fill_diagonal_(self.add_margin).to(logit_tail.device)
        
        triplet_mask = batch_dict.get("triplet_mask", None)
        if triplet_mask is not None:
            logit_head.masked_fill_(~triplet_mask, 1e-4)

        logit_head = logit_head * self.log_inv_t.exp()
        # logit_head = torch.cat([logit_head, logit_off_diag_hr_tr * self.hr_tr_s], dim=-1)
        logit_tail = logit_tail * self.log_inv_t.exp()
        # logit_tail = torch.cat([logit_tail, logit_off_diag_hr_tr.t() * self.hr_tr_s], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            # pointwise multiplication for similarity
            self_neg_logits_head = torch.sum(hr_vector * head_vector,
                                             dim=1) * self.log_inv_t.exp()  # do not repeat head
            self_neg_logits_tail = torch.sum(tr_vector * tail_vector,
                                             dim=1) * self.log_inv_t.exp()  # do not repeat tail
            self_negative_mask_dict = batch_dict['self_negative_mask']
            self_neg_logits_head.masked_fill_(~self_negative_mask_dict['head_mask'], -100)  # masking
            self_neg_logits_tail.masked_fill_(~self_negative_mask_dict['tail_mask'], -100)

            logit_head = torch.cat([logit_head, self_neg_logits_head.unsqueeze(1)], dim=-1)
            logit_tail = torch.cat([logit_tail, self_neg_logits_tail.unsqueeze(1)], dim=-1)
        
        return {'logit_head': logit_head,
                'logit_tail':logit_tail,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_tr_s': self.hr_tr_s.detach(),  
                'hr_vector': hr_vector,
                'tr_vector': tr_vector,
                'head_vector': head_vector,
                'tail_vector': tail_vector
                }

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -100)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()  # update pre-batch with tail vectors
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits


def _pool_output(pooling: str,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = last_hidden_state[0]
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -100
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
