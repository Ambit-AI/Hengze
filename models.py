# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import (
    BertForSequenceClassification, 
    AlbertForSequenceClassification, 
    XLNetForSequenceClassification, 
    RobertaForSequenceClassification, 
    AutoTokenizer
)


class RobertModel(nn.Module):
    def __init__(self, requires_grad = True):
        super(RobertModel, self).__init__()
        self.bert = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels = 3)
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device(0)
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels=1):
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
        
    
    