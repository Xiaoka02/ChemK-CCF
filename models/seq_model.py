# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4


class SeqModel(nn.Module):
    def __init__(self, hidden_size, dropout, device, model_path, max_length, num_heads=8):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.max_length = max_length

        # BERT model
        self.bert = AutoModel.from_pretrained(model_path).to(device)

        # Additional Attention Layer
        self.self_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        ).to(device)

        # Feature transformation
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        ).to(device)

        self.final_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        ).to(device)

    def forward(self, src=None, input_ids=None, attention_mask=None):
        if src is not None:
            input_ids = src

        # Handle the attention_mask dimension
        if attention_mask is not None and attention_mask.dim() == 4:
            attention_mask = attention_mask.squeeze(1).squeeze(1)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state

        hidden = self.output_layer(sequence_output)

        return self.final_layer(hidden)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        assert hidden_size % num_heads == 0

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        return self.out_linear(out)




