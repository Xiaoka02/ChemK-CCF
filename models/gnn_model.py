# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from preprocess.featurization_2D import BatchMolGraph
from preprocess.nn_utils import index_select_ND, get_activation_function


class GNNModel(nn.Module):
    def __init__(self, atom_fdim: int, bond_fdim: int, hidden_size: int,
                 bias: bool, depth: int, dropout: float, activation: str, device: str):
        super(GNNModel, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.device = device
        self.num_heads = 8

        # Atom Encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_fdim, hidden_size, bias=bias),
            nn.LayerNorm(hidden_size),
            get_activation_function(activation),
            nn.Dropout(dropout)
        ).to(device)

        # Bond Encoder
        self.bond_encoder = nn.Sequential(
            nn.Linear(bond_fdim, hidden_size, bias=bias),
            nn.LayerNorm(hidden_size),
            get_activation_function(activation),
            nn.Dropout(dropout)
        ).to(device)

        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_size, self.num_heads, dropout).to(device) for _ in range(depth)])

        # Global context
        self.global_context = GlobalContext(hidden_size).to(device)

        # Gate fusion
        self.gated_fusion = GatedFusion(hidden_size).to(device)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            get_activation_function(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        ).to(device)

    def forward(self, mol_graph: BatchMolGraph, features_batch=None, batch_mask_gnn=None) -> torch.FloatTensor:

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb = (
            f_atoms.to(self.device),
            f_bonds.to(self.device),
            a2b.to(self.device),
            b2a.to(self.device),
            b2revb.to(self.device),
        )

        x = self.atom_encoder(f_atoms)  # [num_atoms, hidden_size]
        edge_feat = self.bond_encoder(f_bonds)  # [num_bonds, hidden_size]

        if features_batch is not None:
            if not isinstance(features_batch, torch.Tensor):
                features_batch = torch.tensor(features_batch).to(self.device)
            features_batch = features_batch.to(self.device)
            if features_batch.dim() == 2:
                if features_batch.size(-1) == x.size(-1):
                    x = x + features_batch
                else:
                    raise ValueError(f"Feature dimensions don't match: {features_batch.size(-1)} vs {x.size(-1)}")

        all_layer_features = [x]

        for layer in self.transformer_layers:

            neighbors = index_select_ND(x, a2b)  # [num_atoms, max_num_bonds, hidden_size]
            neighbor_edge_feat = index_select_ND(edge_feat, a2b)

            global_context = self.global_context(x, a_scope)

            combined_features = self.gated_fusion(x, global_context)

            x = layer(combined_features, neighbors, neighbor_edge_feat)

            edge_feat = edge_feat + F.dropout(index_select_ND(x, b2a), p=self.dropout, training=self.training)

            all_layer_features.append(x)

        output = self.output_layer(torch.cat([
            all_layer_features[-1],
            all_layer_features[0],
        ], dim=1))

        if batch_mask_gnn is not None:
            batch_mask_gnn = batch_mask_gnn.to(self.device)
            if output.size(0) > batch_mask_gnn.size(0):
                output = output[:batch_mask_gnn.size(0)]
            if batch_mask_gnn.dim() == 1:
                batch_mask_gnn = batch_mask_gnn.unsqueeze(-1).expand(-1, output.size(1))
            output = output * batch_mask_gnn

        return output


class GraphTransformerLayer(nn.Module):
    """Graph Transformer Layer"""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super(GraphTransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)

        self.neighbor_attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_heads)
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor, neighbor_edge_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_atoms, hidden_size]
            neighbors:[num_atoms, max_num_bonds, hidden_size]
            neighbor_edge_feat: [num_atoms, max_num_bonds, hidden_size]
        Returns:
            Updated node characteristics
        """

        attn_output, _ = self.self_attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = x + self.dropout(attn_output.squeeze(0))
        x = self.norm1(x)

        neighbor_scores = self.neighbor_attn(
            torch.cat([
                x.unsqueeze(1).expand(-1, neighbors.size(1), -1),
                neighbors
            ], dim=-1)
        )  # [num_atoms, max_num_bonds, num_heads]

        neighbor_weights = F.softmax(neighbor_scores, dim=1)
        neighbor_weights = neighbor_weights.mean(dim=-1, keepdim=True)

        neighbor_message = (neighbors * neighbor_weights * neighbor_edge_feat).sum(dim=1)
        x = x + self.dropout(neighbor_message)

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class GlobalContext(nn.Module):
    """global context module"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x: torch.Tensor, scope: List[Tuple[int, int]]) -> torch.Tensor:
        weights = self.attention(x)
        weights = F.softmax(weights, dim=0)

        global_context = (x * weights).sum(dim=0, keepdim=True)
        global_context = self.transform(global_context)

        expanded_context = global_context.expand(x.size(0), -1)

        return expanded_context


class GatedFusion(nn.Module):
    """Gate Fusion Module"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([x1, x2], dim=-1))
        return x1 * (1 - gate) + x2 * gate