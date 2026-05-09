# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from models.gnn_block import GraphNorm, MeanPool, GraphPool
from models.compound_encoder import (
    AtomEmbedding, BondEmbedding, BondFloatRBF,
    BondAngleFloatRBF, DirectionalMessagePassing
)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from rdkit.Chem import Draw


class GeoGNNBlock(nn.Module):
    """GeoGNN Block with 3D spatial information"""

    def __init__(self, embed_dim, dropout_rate, last_act, device):
        super(GeoGNNBlock, self).__init__()
        self.embed_dim = embed_dim
        self.last_act = last_act

        # Base GNN layer
        self.gnn = GINEConv(nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim)
        )).to(device)

        # 3D spatial information processing
        self.spatial_encoder = DirectionalMessagePassing(embed_dim, device)

        # Attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4).to(device)

        # Gate
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        ).to(device)

        self.norm1 = nn.LayerNorm(embed_dim).to(device)
        self.norm2 = nn.LayerNorm(embed_dim).to(device)
        self.graph_norm = GraphNorm(device=device)

        if last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, graph, node_hidden, edge_hidden, node_id, edge_id, pos=None):
        """Forward pass with 3D spatial information"""

        identity = node_hidden

        if edge_hidden.size(0) != graph.edge_index.size(1):
            min_size = min(edge_hidden.size(0), graph.edge_index.size(1))
            edge_hidden = edge_hidden[:min_size]
            graph.edge_index = graph.edge_index[:, :min_size]

        gnn_out = self.gnn(node_hidden, graph.edge_index, edge_hidden)
        gnn_out = self.norm1(gnn_out)

        # 3D spatial feature processing
        if pos is not None:
            spatial_out = self.spatial_encoder(gnn_out, graph.edge_index, pos)
            fused_features, _ = self.attention(
                gnn_out.unsqueeze(0),
                spatial_out.unsqueeze(0),
                spatial_out.unsqueeze(0)
            )
            fused_features = fused_features.squeeze(0)
        else:
            fused_features = gnn_out

        # Gated residual connection
        gate = self.gate(torch.cat([fused_features, identity], dim=-1))
        out = gate * fused_features + (1 - gate) * identity

        out = self.norm2(out)
        out = self.graph_norm(graph, out, node_id, edge_id)

        if self.last_act:
            out = self.act(out)
        out = self.dropout(out)

        return out


class GeoModel(nn.Module):
    """GeoGNN Model with comprehensive 3D structure handling"""

    def __init__(self, hidden_dim, dropout_rate, layer_num, readout,
                 atom_names, bond_names, bond_float_names, bond_angle_float_names, device):
        super(GeoModel, self).__init__()

        self.embed_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.layer_num = layer_num
        self.readout = readout
        self.device = device

        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names

        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim, device=device)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim, device=device)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim, device=device)

        self.bond_embedding_list = nn.ModuleList()
        self.bond_float_rbf_list = nn.ModuleList()
        self.bond_angle_float_rbf_list = nn.ModuleList()
        self.atom_bond_block_list = nn.ModuleList()
        self.bond_angle_block_list = nn.ModuleList()

        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(
                BondEmbedding(self.bond_names, self.embed_dim, device=device))
            self.bond_float_rbf_list.append(
                BondFloatRBF(self.bond_float_names, self.embed_dim, device=device))
            self.bond_angle_float_rbf_list.append(
                BondAngleFloatRBF(self.bond_angle_float_names, self.embed_dim, device=device))
            self.atom_bond_block_list.append(
                GeoGNNBlock(
                    self.embed_dim,
                    self.dropout_rate,
                    last_act=(layer_id != self.layer_num - 1),
                    device=device
                ))
            self.bond_angle_block_list.append(
                GeoGNNBlock(
                    self.embed_dim,
                    self.dropout_rate,
                    last_act=(layer_id != self.layer_num - 1),
                    device=device
                ))

        if self.readout == 'mean':
            self.graph_pool = MeanPool()
        else:
            self.graph_pool = GraphPool(pool_type=self.readout)

        self.dihedral_encoder = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        ).to(device)

    @property
    def node_dim(self):
        return self.embed_dim

    @property
    def graph_dim(self):
        return self.embed_dim

    def forward(self, atom_bond_graph, bond_angle_graph, node_id, edge_id, pos=None):
        if hasattr(atom_bond_graph, 'atom_pos'):
            pos = atom_bond_graph.atom_pos
        elif hasattr(atom_bond_graph, 'pos'):
            pos = atom_bond_graph.pos

        node_hidden = self.init_atom_embedding(atom_bond_graph.x.T)
        bond_embed = self.init_bond_embedding(atom_bond_graph.edge_attr.T)
        edge_hidden = bond_embed + self.init_bond_float_rbf(
            atom_bond_graph.edge_attr.T[len(self.bond_names):])

        node_hidden_list = [node_hidden]
        edge_hidden_list = [edge_hidden]

        for layer_id in range(self.layer_num):
            node_hidden = self.atom_bond_block_list[layer_id](
                atom_bond_graph,
                node_hidden_list[layer_id],
                edge_hidden_list[layer_id],
                node_id[0],
                edge_id[0],
                pos
            )

            cur_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph.edge_attr.T)
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](
                atom_bond_graph.edge_attr.T[len(self.bond_names):])
            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angle_graph.edge_attr.T)

            edge_hidden = self.bond_angle_block_list[layer_id](
                bond_angle_graph,
                cur_edge_hidden,
                cur_angle_hidden,
                node_id[1],
                edge_id[1],
                pos
            )

            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)

        return node_hidden_list[-1], edge_hidden_list[-1]
