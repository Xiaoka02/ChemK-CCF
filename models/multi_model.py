#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GlobalAttention
from models.gnn_model import GNNModel
from models.gem_model import GeoModel
from models.seq_model import SeqModel
from models.img_model import MolecularImageModel


class Multi_modal(nn.Module):
    def __init__(self, args, device):
        super(Multi_modal, self).__init__()
        self.args = args
        self.device = device
        self.latent_dim = args.latent_dim
        self.img_hidden_dim = args.img_hidden_dim
        self.batch_size = args.batch_size
        self.graph = args.graph
        self.sequence = args.sequence
        self.geometry = args.geometry
        self.image = args.image

        if self.sequence:
            self.transformer = SeqModel(hidden_size=args.bert_hidden_dim, dropout=args.dropout, device=device,
                                        model_path=args.bert_model_path, max_length=args.max_seq_length,
                                        num_heads=args.bert_num_heads).to(self.device)

        if self.graph:
            self.gnn = GNNModel(
                atom_fdim=args.gnn_atom_dim,
                bond_fdim=args.gnn_bond_dim,
                hidden_size=args.gnn_hidden_dim,
                bias=args.bias,
                depth=args.gnn_num_layers,
                dropout=args.dropout,
                activation=args.gnn_activation,
                device=device
            )

        if self.geometry:
            self.geo_module = GeoModel(hidden_dim=args.geo_hidden_dim,
                                       dropout_rate=args.geo_dropout_rate,
                                       layer_num=args.geo_layer_num,
                                       readout=args.geo_readout,
                                       atom_names=args.atom_names,
                                       bond_names=args.bond_names,
                                       bond_float_names=args.bond_float_names,
                                       bond_angle_float_names=args.bond_angle_float_names,
                                       device=device).to(self.device)

        # image_module
        if self.image:
            self.image_module = MolecularImageModel(channels=args.img_use_channels,
                                                    hidden_dim=args.img_hidden_dim,
                                                    latent_dim=args.latent_dim,
                                                    dropout=args.img_dropout,
                                                    use_layernorm=args.img_use_layernorm,
                                                    pool_type=args.img_pool_type,
                                                    attention_type=args.img_attention,
                                                    device=device).to(self.device)

        if args.pro_num == 3:
            self.pro_seq = nn.Sequential(nn.Linear(256, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_gnn = nn.Sequential(nn.Linear(args.gnn_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_geo = nn.Sequential(nn.Linear(args.geo_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_img = nn.Sequential(nn.Linear(self.img_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
        elif args.pro_num == 1:
            self.pro_seq = nn.Sequential(nn.Linear(256, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_gnn = self.pro_seq
            self.pro_geo = self.pro_seq
            self.pro_img = self.pro_seq

        self.entropy = loss_type[args.task_type]

        if args.pool_type == 'mean':
            self.pool = global_mean_pool
        else:
            self.pool = Global_Attention(args.latent_dim).to(self.device)

        # bert dimension transformation layer
        self.bert_transform = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        ).to(self.device)

        # Feature alignment
        self.num_modalities = sum([self.sequence, self.graph, self.geometry, self.image])
        if self.num_modalities > 1:
            self.fusion_dim = args.latent_dim
            bert_output_dim = args.bert_hidden_dim
            seq_dim = 256 if hasattr(self, 'bert_transform') else bert_output_dim
            self.feature_align = nn.ModuleDict({
                'seq': nn.Linear(seq_dim, self.fusion_dim).to(device) if self.sequence else None,
                'gnn': nn.Linear(args.gnn_hidden_dim, self.fusion_dim).to(device) if self.graph else None,
                'geo': nn.Linear(args.geo_hidden_dim, self.fusion_dim).to(device) if self.geometry else None,
                'img': nn.Linear(args.img_hidden_dim, self.fusion_dim).to(device) if self.image else None
            })
            # Fusion
            if self.args.fusion == 'weight_fusion':
                self.fusion = WeightFusion(
                    seq_hidden_dim=args.seq_hidden_dim,
                    gnn_hidden_dim=args.gnn_hidden_dim,
                    geo_hidden_dim=args.geo_hidden_dim,
                    img_hidden_dim=args.img_hidden_dim,
                    fusion_dim=self.fusion_dim,
                    num_heads=8,
                    dropout=0.1,
                    device=device,
                    args=args
                )
                self.output_layer = nn.Sequential(
                    nn.Linear(self.fusion_dim, self.fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(self.fusion_dim, args.output_dim)
                ).to(self.device)
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(args.latent_dim, args.latent_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.latent_dim, args.output_dim)
            ).to(self.device)

    def forward(self, input_ids, attention_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch, batch_mask_gnn,
                graph_dict, node_id_all, edge_id_all, img_batch=None):
        global molecule_emb
        if img_batch is not None:
            img_batch = img_batch.to(self.device)
        x_list = list()
        cl_list = list()
        modality_types = []
        batch_size = None

        # Sequence
        if self.sequence:
            node_seq_x = self.transformer(
                src=input_ids,
                attention_mask=attention_mask
            )
            if batch_mask_seq is not None:
                if batch_mask_seq.size(1) != node_seq_x.size(1):
                    new_mask = torch.zeros(
                        batch_mask_seq.size(0),
                        node_seq_x.size(1),
                        device=batch_mask_seq.device
                    )
                    new_mask[:, :batch_mask_seq.size(1)] = batch_mask_seq
                    batch_mask_seq = new_mask

                mask_bool = batch_mask_seq.bool()
                valid_tokens = node_seq_x[mask_bool]
            else:
                valid_tokens = torch.mean(node_seq_x, dim=1)

            valid_tokens = self.bert_transform(valid_tokens)
            batch_size = valid_tokens.size(0)

            x_list.append(valid_tokens)
            cl_list.append(self.pro_seq(valid_tokens))
            modality_types.append('seq')

        # Graph
        if self.graph:
            node_gnn_x = self.gnn(gnn_batch_graph, gnn_feature_batch, batch_mask_gnn)
            graph_gnn_x = self.pool(node_gnn_x, batch_mask_gnn)
            if batch_size and graph_gnn_x.size(0) != batch_size:
                if graph_gnn_x.size(0) == 1:
                    graph_gnn_x = graph_gnn_x.expand(batch_size, -1)
                else:
                    graph_gnn_x = graph_gnn_x[:batch_size]

            if self.args.norm:
                x_list.append(F.normalize(graph_gnn_x, p=2, dim=1))
            else:
                x_list.append(graph_gnn_x)
            cl_list.append(self.pro_gnn(graph_gnn_x))
            modality_types.append('gnn')

        # Geometry
        if self.geometry:
            node_repr, edge_repr = self.geo_module(graph_dict[0], graph_dict[1], node_id_all, edge_id_all)
            graph_geo_x = self.pool(node_repr, node_id_all[0])
            if batch_size and graph_geo_x.size(0) != batch_size:
                if graph_geo_x.size(0) == 1:
                    graph_geo_x = graph_geo_x.expand(batch_size, -1)
                else:
                    graph_geo_x = graph_geo_x[:batch_size]
            if self.args.norm:
                x_list.append(F.normalize(graph_geo_x, p=2, dim=1))
            else:
                x_list.append(graph_geo_x)
            cl_list.append(self.pro_geo(graph_geo_x.to(self.device)))
            modality_types.append('geo')

        # Image
        if self.image:
            img_features, functional_groups = self.image_module(img_batch)
            img_features = nn.Linear(img_features.size(1), self.img_hidden_dim).to(self.device)(img_features)
            if batch_size and img_features.size(0) != batch_size:
                if img_features.size(0) == 1:
                    img_features = img_features.expand(batch_size, -1)
                else:
                    img_features = img_features[:batch_size]
            if self.args.norm:
                x_list.append(F.normalize(img_features, p=2, dim=1))
            else:
                x_list.append(img_features)
            cl_list.append(self.pro_img(img_features))
            modality_types.append('img')

        if self.num_modalities == 1:
            molecule_emb = x_list[0]
            if molecule_emb.size(0) == 1:
                correct_batch_size = input_ids.size(0) if input_ids is not None else 32
                molecule_emb = molecule_emb.expand(correct_batch_size, -1)
            pred = self.output_layer(molecule_emb)
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)
            return x_list, pred
        else:
            features_before_fusion = x_list

            # Fusion process
            if self.args.fusion == 'weight_fusion':

                aligned_features = []
                modality_types = []
                feature_idx = 0

                # Sequence
                if self.sequence and len(x_list) > feature_idx:
                    seq_feat = self.feature_align['seq'](x_list[feature_idx])
                    aligned_features.append(seq_feat)
                    modality_types.append('seq')
                    feature_idx += 1

                # Graph
                if self.graph and len(x_list) > feature_idx:
                    gnn_feat = self.feature_align['gnn'](x_list[feature_idx])
                    aligned_features.append(gnn_feat)
                    modality_types.append('gnn')
                    feature_idx += 1

                # Geometry
                if self.geometry and len(x_list) > feature_idx:
                    geo_feat = self.feature_align['geo'](x_list[feature_idx])
                    aligned_features.append(geo_feat)
                    modality_types.append('geo')
                    feature_idx += 1

                # Image
                if self.image and len(x_list) > feature_idx:
                    img_feat = self.feature_align['img'](x_list[feature_idx])
                    aligned_features.append(img_feat)
                    modality_types.append('img')

                batch_size = input_ids.size(0) if input_ids is not None else aligned_features[0].size(0)
                for i in range(len(aligned_features)):
                    if aligned_features[i].size(0) != batch_size:
                        aligned_features[i] = aligned_features[i].expand(batch_size, -1)

                molecule_emb = self.fusion(aligned_features)

                pred = self.output_layer(molecule_emb)

                if pred.dim() > 2:
                    pred = pred.squeeze()
                if pred.dim() == 1:
                    pred = pred.unsqueeze(-1)

            #  # Normalization
            molecule_emb = F.normalize(molecule_emb, p=2, dim=1)
            pred = self.output_layer(molecule_emb)

            return x_list, pred

    def label_loss(self, pred, label, mask):
        assert mask.sum().item() > 0
        loss_mat = self.entropy(pred, label)
        return loss_mat.sum() / mask.sum()

    def cl_loss(self, x1, x2, T=0.1):
        epsilon = 1e-7
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)

        if x1.size(0) != x2.size(0):
            if x1.size(0) == 1:
                x1 = x1.expand(x2.size(0), -1)
            elif x2.size(0) == 1:
                x2 = x2.expand(x1.size(0), -1)
            else:
                min_batch = min(x1.size(0), x2.size(0))
                x1 = x1[:min_batch]
                x2 = x2[:min_batch]

        batch_size, _ = x1.size()
        if torch.isnan(x1).any() or torch.isnan(x2).any():
            return torch.tensor(0.0, device=x1.device)
        x1_abs = x1.norm(dim=1) + epsilon
        x2_abs = x2.norm(dim=1) + epsilon
        # Compute the similarity matrix
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.clamp(sim_matrix, min=-1.0, max=1.0)
        sim_matrix = torch.exp(sim_matrix / T)
        # Getting the similarity of positive sample pairs
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        denominator = sim_matrix.sum(dim=1) - pos_sim + epsilon
        loss = - torch.log(pos_sim / denominator + epsilon)
        valid_mask = ~torch.isnan(loss)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=x1.device)

        return loss[valid_mask].mean()

    def loss_cal(self, x_list, pred, label, mask, alpha=0.08):
        # Labeling losses
        loss1 = self.label_loss(pred, label, mask)

        # Comparative loss
        loss2 = torch.tensor(0.0, device=pred.device)
        valid_pairs = 0
        for i in range(len(x_list)):
            for j in range(i + 1, len(x_list)):
                if x_list[i] is not None and x_list[j] is not None:
                    try:
                        cl_loss = self.cl_loss(x_list[i], x_list[j])
                        if not torch.isnan(cl_loss) and not torch.isinf(cl_loss):
                            loss2 += cl_loss
                            valid_pairs += 1
                    except Exception as e:
                        print(f"Error computing cl_loss for pair ({i},{j}): {e}")
                        continue

        if valid_pairs > 0:
            loss2 = loss2 / valid_pairs
        return loss1 + alpha * loss2, loss1, loss2


loss_type = {'class': nn.BCEWithLogitsLoss(reduction="none"), 'reg': nn.MSELoss(reduction="none")}


class Global_Attention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.at = GlobalAttention(gate_nn=torch.nn.Linear(hidden_size, 1))

    def forward(self, x, batch):
        return self.at(x, batch)


class WeightFusion(nn.Module):
    def __init__(self, seq_hidden_dim, gnn_hidden_dim, geo_hidden_dim, img_hidden_dim,
                 fusion_dim, num_heads=8, dropout=0.1, device=None, args=None):
        """
        Args:
            seq_hidden_dim: int
            gnn_hidden_dim: int
            geo_hidden_dim: int
            img_hidden_dim: int
            fusion_dim: int
            num_heads: int
            dropout: float
            device: torch.device
        """
        super().__init__()

        if fusion_dim is None:
            fusion_dim = max(seq_hidden_dim, gnn_hidden_dim,
                             geo_hidden_dim, img_hidden_dim)

        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.device = device
        self.args = args

        self.modal_dims = {
            'seq': seq_hidden_dim,
            'gnn': gnn_hidden_dim,
            'geo': geo_hidden_dim,
            'img': img_hidden_dim
        }

        # Modal feature projection layer
        self.feature_align = nn.ModuleDict({
            'seq': nn.Linear(seq_hidden_dim, fusion_dim).to(device),
            'gnn': nn.Linear(gnn_hidden_dim, fusion_dim).to(device),
            'geo': nn.Linear(geo_hidden_dim, fusion_dim).to(device),
            'img': nn.Linear(img_hidden_dim, fusion_dim).to(device)
        })

        # Self-attentive layer inside the modal
        self.self_attention = nn.ModuleDict({
            modal: nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ).to(device)
            for modal in ['seq', 'gnn', 'geo', 'img']
        })

        # Cross-attention layers between modes
        self.cross_attention = nn.ModuleDict({
            modal: nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ).to(device)
            for modal in ['seq', 'gnn', 'geo', 'img']
        })

        # Global self-attention after feature fusion
        self.global_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        ).to(device)

        # Normalization
        self.layer_norms = nn.ModuleDict({
            f"{modal}_norm": nn.LayerNorm(fusion_dim).to(device)
            for modal in ['seq', 'gnn', 'geo', 'img']
        })
        self.global_norm = nn.LayerNorm(fusion_dim).to(device)

        # Weight-generating
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim // 2),
            nn.Linear(fusion_dim // 2, 4),
            nn.Softmax(dim=-1)
        ).to(device)

        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim)
        ).to(device)

    def forward(self, aligned_features):
        """
        Args:
            aligned_features: list of tensors
        Returns:
            tensor of shape [batch_size, fusion_dim]
        """
        features_dict = {}
        modality_types = ['seq', 'gnn', 'geo', 'img']

        for i, feat in enumerate(aligned_features):
            if feat is not None:
                if feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)
                features_dict[modality_types[i]] = self.feature_align[modality_types[i]](feat)

        if not features_dict:
            raise ValueError("No valid features provided")

        # Modal internal self-attention
        intra_features = {}
        for modal, feat in features_dict.items():
            if feat.dim() == 2:
                feat = feat.unsqueeze(1)

            attn_output, attn_weights = self.self_attention[modal](
                feat,  # query
                feat,  # key
                feat  # value
            )

            intra_features[modal] = self.layer_norms[f"{modal}_norm"](feat + attn_output)

        # Intermodal cross attention
        cross_features = {}
        modalities = list(features_dict.keys())

        for query_modal in modalities:
            other_modals = [m for m in modalities if m != query_modal]
            if not other_modals:
                cross_features[query_modal] = intra_features[query_modal]
                continue

            other_features = torch.cat([intra_features[m] for m in other_modals],
                                       dim=1)

            query = intra_features[query_modal]

            attn_output, attn_weights = self.cross_attention[query_modal](
                query,
                other_features,
                other_features
            )

            cross_features[query_modal] = self.layer_norms[f"{query_modal}_norm"](query + attn_output)

        #  Global Feature Fusion
        modal_features = [feat.squeeze(1) for feat in cross_features.values()]
        stacked_features = torch.stack(modal_features, dim=1)

        # Generate fusion weights
        fusion_weights = self.fusion(stacked_features.mean(dim=1))

        # Weighted fusion
        weighted_features = []
        for i, feat in enumerate(modal_features):
            weighted_feat = feat * fusion_weights[:, i:i + 1]
            weighted_features.append(weighted_feat)

        fused_feature = sum(weighted_features)
        output = self.output_layer(fused_feature)

        return output

    def intra_modal_attention(self, features_dict):
        """Self-attentive treatment of modal interiors"""
        intra_outputs = {}
        for modal, feat in features_dict.items():
            if feat.dim() == 2:
                feat = feat.unsqueeze(1)

            attn_output, attn_weights = self.self_attention[modal](feat, feat, feat)
            self.attention_weights[f"{modal}_self"] = attn_weights.detach().clone()

            intra_outputs[modal] = self.layer_norms[f"{modal}_norm"](feat + attn_output)

        return intra_outputs

    def cross_modal_attention(self, features_dict):
        """Intermodal cross-attention processing"""
        cross_outputs = {}
        modalities = list(features_dict.keys())

        for query_modal in modalities:
            other_modals = [m for m in modalities if m != query_modal]
            if not other_modals:
                cross_outputs[query_modal] = features_dict[query_modal]
                continue

            other_features = torch.cat([features_dict[m] for m in other_modals], dim=1)

            query = features_dict[query_modal]
            attn_output, attn_weights = self.cross_attention[query_modal](
                query, other_features, other_features
            )
            self.attention_weights[f"{query_modal}_cross"] = attn_weights.detach().clone()

            cross_outputs[query_modal] = self.layer_norms[f"{query_modal}_norm"](
                query + attn_output
            )
        return cross_outputs



