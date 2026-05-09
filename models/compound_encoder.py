#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn

from utils.compound_tools import CompoundKit
from models.basic_block import RBF


class AtomEmbedding(torch.nn.Module):
    """
    Atom Encoder
    """
    def __init__(self, atom_names, embed_dim, device):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names
        self.embed_list = nn.ModuleList()
        for name in self.atom_names:
            embed = nn.Embedding(CompoundKit.get_atom_feature_size(name) + 5, embed_dim).to(device)
            self.embed_list.append(embed)

        self.pos_encoder = nn.Linear(3, embed_dim).to(device)

    def forward(self, node_features):
        """
        Args:
            node_features(dict of tensor): node features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_names):
            out_embed += self.embed_list[i](node_features[i])
        return out_embed


class AtomFloatEmbedding(torch.nn.Module):
    """
    Atom Float Encoder
    """
    def __init__(self, atom_float_names, embed_dim, rbf_params=None, device=None):
        super(AtomFloatEmbedding, self).__init__()
        self.atom_float_names = atom_float_names

        if rbf_params is None:
            self.rbf_params = {
                'van_der_waals_radis': (torch.arange(1, 3, 0.2), 10.0),  # (centers, gamma)
                'partial_charge': (torch.arange(-1, 4, 0.25), 10.0),  # (centers, gamma)
                'mass': (torch.arange(0, 2, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
            self.linear_list = nn.ModuleList()
            self.rbf_list = nn.ModuleList()
            for name in self.atom_float_names:
                centers, gamma = self.rbf_params[name]
                rbf = RBF(centers, gamma).to(device)
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim).to(device)
                self.linear_list.append(linear)

    def forward(self, feats):
        """
        Args:
            feats(dict of tensor): node float features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_float_names):
            x = feats[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class BondEmbedding(nn.Module):
    """
    Bond Encoder
    """

    def __init__(self, bond_names, embed_dim, device):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names

        self.embed_list = nn.ModuleList()
        for name in self.bond_names:
            embed = nn.Embedding(CompoundKit.get_bond_feature_size(name) + 5, embed_dim).to(device)
            self.embed_list.append(embed)

        self.directional_mp = DirectionalMessagePassing(embed_dim, device)

    def forward(self, edge_features):
        """
        Args:
            edge_features(dict of tensor): edge features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_names):
            out_embed += self.embed_list[i](edge_features[i].long())
        return out_embed


class BondFloatRBF(nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_float_names, embed_dim, rbf_params=None, device=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (torch.arange(0, 2, 0.1).to(device), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma).to(device)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim).to(device)
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[i]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x.float())
        return out_embed


class BondAngleFloatRBF(nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None, device=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (torch.arange(0, np.pi, 0.1).to(device), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_angle_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma).to(device)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim).to(device)
            self.linear_list.append(linear)

    def forward(self, bond_angle_float_features):
        """
        Args:
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            x = bond_angle_float_features[i]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x.float())
        return out_embed


class DirectionalMessagePassing(nn.Module):
    """
    Directional Message Passing module for 3D molecular information
    """

    def __init__(self, embed_dim, device):
        super(DirectionalMessagePassing, self).__init__()
        self.embed_dim = embed_dim

        # Distance Encoding
        self.distance_expansion = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        ).to(device)

        # Directional Encoding
        self.direction_expansion = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        ).to(device)

        # News Update
        self.message_nn = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        ).to(device)

    def forward(self, x, edge_index, pos):
        j, i = edge_index

        # Calculating Direction Vectors and Distances
        dir_vec = pos[i] - pos[j]
        dist = torch.norm(dir_vec, dim=1, keepdim=True)
        dir_vec = dir_vec / (dist + 1e-7)

        # Computing Distance and Direction Embeddings
        dist_embed = self.distance_expansion(dist)
        dir_embed = self.direction_expansion(dir_vec)

        # Message Construction
        message = self.message_nn(torch.cat([
            x[i], x[j], dist_embed * dir_embed
        ], dim=1))

        return message