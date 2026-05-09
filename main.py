# -*- coding: utf-8 -*-
import os
import copy
import torch
import logging
import warnings
import numpy as np
import pandas as pd
import statistics
from datetime import datetime
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.optim import Adam
from parser_args import get_args
from torchvision import transforms
from torch.utils.data import BatchSampler, RandomSampler, DataLoader

from preprocess.data import StandardScaler
from preprocess.featurization_2D import BatchMolGraph, MolGraph, get_atom_fdim, get_bond_fdim
from preprocess.utils import get_class_sizes
from preprocess.data_3d import DataProcessor
from preprocess.bert_date import MolecularDataProcessor
from preprocess.data_image import (load_or_generate_images, generate_labels_file, ImageDataset,
                                   load_filenames_and_labels, process_dataset)

from utils.dataset import get_data, split_data, MoleculeDataset, InMemoryDataset
from utils.evaluate import eval_rocauc, eval_rmse
from rdkit import Chem
from models.multi_model import Multi_modal
from collections import defaultdict
from glob import glob

PAD = 0


def create_virtual_geometry_data(num_atoms, device):
    """
    Create virtual geometry data for fallback when geometry generation fails.
    Args:
        num_atoms: Number of atoms
        device: Device
    Returns:
        tuple: (geo_gen, node_id_all, edge_id_all)
    """
    from torch_geometric.data import Batch, Data

    virtual_atomic_nums = torch.full((num_atoms,), 6, dtype=torch.long).to(device)
    virtual_chiral_tags = torch.zeros(num_atoms, dtype=torch.long).to(device)
    virtual_degrees = torch.full((num_atoms,), 2, dtype=torch.long).to(device)
    virtual_formal_charges = torch.zeros(num_atoms, dtype=torch.long).to(device)
    virtual_hybridizations = torch.full((num_atoms,), 2, dtype=torch.long).to(device)
    virtual_implicit_valences = torch.full((num_atoms,), 3, dtype=torch.long).to(device)
    virtual_is_aromatic = torch.zeros(num_atoms, dtype=torch.long).to(device)
    virtual_total_numHs = torch.ones(num_atoms, dtype=torch.long).to(device)

    virtual_atom_features = torch.stack([
        virtual_atomic_nums, virtual_chiral_tags, virtual_degrees,
        virtual_formal_charges, virtual_hybridizations, virtual_implicit_valences,
        virtual_is_aromatic, virtual_total_numHs
    ], dim=1)

    virtual_atom_graph = Data(
        x=virtual_atom_features,
        edge_index=torch.zeros((2, 1), dtype=torch.long).to(device),
        edge_attr=torch.ones((1, 4), dtype=torch.float).to(device)
    )
    virtual_bond_graph = Data(
        x=torch.ones((1, 4), dtype=torch.float).to(device),
        edge_index=torch.zeros((2, 1), dtype=torch.long).to(device),
        edge_attr=torch.ones((1, 1), dtype=torch.float).to(device)
    )

    geo_gen = (Batch.from_data_list([virtual_atom_graph]),
               Batch.from_data_list([virtual_bond_graph]))

    node_id_all = [torch.zeros(1, dtype=torch.long).to(device),
                   torch.zeros(1, dtype=torch.long).to(device)]
    edge_id_all = [torch.zeros(1, dtype=torch.long).to(device),
                   torch.zeros(1, dtype=torch.long).to(device)]

    return geo_gen, node_id_all, edge_id_all


UNK = 1
EOS = 2
SOS = 3
MASK = 4
warnings.filterwarnings('ignore')

IMAGEMOL_CONFIG = {
    'size': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}


def prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, image_data, device):
    # Sequence data processing
    input_ids = seq_data[idx].to(device)
    attention_mask = seq_mask[idx].to(device)

    # 2D molecular map data processing
    mol_batch = MoleculeDataset([gnn_data[i] for i in idx])
    smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()

    mol_graphs = []
    for smiles in smiles_batch:
        try:
            mol_graph = MolGraph(smiles, args)
            mol_graphs.append(mol_graph)
        except Exception as e:
            print(f"Error processing molecule {smiles}: {str(e)}")
            continue
    gnn_batch = BatchMolGraph(mol_graphs, args)

    # 3D geometry data processing
    geo_gen = geo_data.get_batch(idx)
    edge_batch1, edge_batch2 = [], []
    node_id_all = [geo_gen[0].batch, geo_gen[1].batch]
    for i in range(geo_gen[0].num_graphs):
        edge_batch1.append(torch.ones(geo_gen[0][i].edge_index.shape[1], dtype=torch.long).to(device) * i)
        edge_batch2.append(torch.ones(geo_gen[1][i].edge_index.shape[1], dtype=torch.long).to(device) * i)
    edge_id_all = [torch.cat(edge_batch1), torch.cat(edge_batch2)]

    # Image data processing
    if isinstance(image_data[0][0], torch.Tensor):
        image_batch = torch.stack([image_data[i][0] for i in idx]).to(device)
    else:
        image_transform = transforms.Compose([
            transforms.Resize((IMAGEMOL_CONFIG['size'], IMAGEMOL_CONFIG['size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGEMOL_CONFIG['mean'],
                std=IMAGEMOL_CONFIG['std']
            )
        ])
        image_batch = torch.stack([image_transform(image_data[i][0]) for i in idx]).to(device)

    # Target value processing
    # Handle targets that may be scalar or list
    if len(target_batch) > 0:
        first_target = target_batch[0]
        if isinstance(first_target, (list, tuple)):
            # Original format: list of lists
            mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]).to(device)
            targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]).to(device)
        else:
            # New format: scalar list
            mask = torch.Tensor([[x is not None] for x in target_batch]).to(device)
            targets = torch.Tensor([[0 if x is None else x] for x in target_batch]).to(device)
    else:
        mask = torch.Tensor([]).to(device)
        targets = torch.Tensor([]).to(device)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'gnn_batch': gnn_batch,
        'features_batch': features_batch,
        'geo_gen': geo_gen,
        'node_id_all': node_id_all,
        'edge_id_all': edge_id_all,
        'image_batch': image_batch,
        'mask': mask,
        'targets': targets,
        'smiles': smiles_batch
    }


def train(args, model, optimizer, train_idx_loader, seq_data, seq_mask, datas, data_3d, image_data, device, epoch,
          base_viz_dir=None):
    model.train()

    total_all_loss = 0
    total_lab_loss = 0
    total_cl_loss = 0
    batch_count = 0
    vis_epoch_interval = 0

    all_batch_preds = []
    all_batch_labels = []

    for idx in tqdm(train_idx_loader):
        model.zero_grad()
        optimizer.zero_grad()
        batch_data = prepare_data(args, idx, seq_data, seq_mask, datas, data_3d, image_data, device)
        if batch_data is None:
            continue
        x_list, preds = model(
            input_ids=batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            batch_mask_seq=None,
            gnn_batch_graph=batch_data['gnn_batch'],
            gnn_feature_batch=batch_data['features_batch'],
            batch_mask_gnn=None,
            graph_dict=batch_data['geo_gen'],
            node_id_all=batch_data['node_id_all'],
            edge_id_all=batch_data['edge_id_all'],
            img_batch=batch_data['image_batch']
        )
        # Calculated losses
        all_loss, lab_loss, cl_loss = model.loss_cal(x_list, preds, batch_data['targets'], batch_data['mask'])

        total_all_loss = all_loss.item() + total_all_loss
        total_lab_loss = lab_loss.item() + total_lab_loss
        total_cl_loss = cl_loss.item() + total_cl_loss
        batch_count += 1
        vis_epoch_interval += 1

        all_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        all_batch_preds.append(preds.detach())
        all_batch_labels.append(batch_data['targets'])

    all_predictions = torch.cat(all_batch_preds, dim=0)
    all_labels = torch.cat(all_batch_labels, dim=0)

    # Calculate average loss
    avg_all_loss = total_all_loss / batch_count if batch_count > 0 else float('inf')
    avg_lab_loss = total_lab_loss / batch_count if batch_count > 0 else float('inf')
    avg_cl_loss = total_cl_loss / batch_count if batch_count > 0 else float('inf')
    return avg_all_loss, avg_lab_loss, avg_cl_loss, all_predictions, all_labels


@torch.no_grad()
def val(args, model, scaler, val_idx_loader, seq_data, seq_mask, gnn_data, geo_data, image_data, device):
    model.eval()
    total_all_loss = 0
    total_lab_loss = 0
    total_cl_loss = 0
    batch_count = 0

    all_batch_preds = []
    all_batch_labels = []

    for idx in val_idx_loader:
        batch_data = prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, image_data, device)
        if batch_data is None:
            continue

        x_list, preds = model(
            input_ids=batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            batch_mask_seq=None,
            gnn_batch_graph=batch_data['gnn_batch'],
            gnn_feature_batch=batch_data['features_batch'],
            batch_mask_gnn=None,
            graph_dict=batch_data['geo_gen'],
            node_id_all=batch_data['node_id_all'],
            edge_id_all=batch_data['edge_id_all'],
            img_batch=batch_data['image_batch']
        )

        if scaler is not None and args.task_type == 'reg':
            preds_np = preds.detach().cpu().numpy().astype(np.float64)
            # Manual inverse normalization: x * std + mean
            preds_inv = preds_np * scaler['std'] + scaler['mean']
            preds = torch.tensor(preds_inv).to(device)

            targets_np = batch_data['targets'].detach().cpu().numpy().astype(np.float64)
            targets_inv = targets_np * scaler['std'] + scaler['mean']
            batch_data['targets'] = torch.tensor(targets_inv).to(device)

        all_loss, lab_loss, cl_loss = model.loss_cal(x_list, preds,
                                                     batch_data['targets'], batch_data['mask'], args.cl_loss)

        total_all_loss = all_loss.item() + total_all_loss
        total_lab_loss = lab_loss.item() + total_lab_loss
        total_cl_loss = cl_loss.item() + total_cl_loss
        batch_count += 1

        all_batch_preds.append(preds.detach())
        all_batch_labels.append(batch_data['targets'])

    all_predictions = torch.cat(all_batch_preds, dim=0)
    all_labels = torch.cat(all_batch_labels, dim=0)

    y_true = all_labels.detach().cpu().numpy()
    y_pred = all_predictions.detach().cpu().numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    avg_all_loss = total_all_loss / batch_count if batch_count > 0 else float('inf')
    avg_lab_loss = total_lab_loss / batch_count if batch_count > 0 else float('inf')
    avg_cl_loss = total_cl_loss / batch_count if batch_count > 0 else float('inf')

    if args.task_type == 'class':
        result = eval_rocauc(input_dict)['rocauc']
    else:
        result = eval_rmse(input_dict)['rmse']

    return result, avg_all_loss, avg_lab_loss, avg_cl_loss, all_predictions, all_labels


@torch.no_grad()
def test(args, model, scaler, test_idx_loader, seq_data, seq_mask, gnn_data, geo_data, image_data, device):
    y_true = []
    y_pred = []
    for idx in test_idx_loader:
        batch_data = prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, image_data, device)
        if batch_data is None:
            continue

        x_list, preds = model(
            input_ids=batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            batch_mask_seq=None,
            gnn_batch_graph=batch_data['gnn_batch'],
            gnn_feature_batch=batch_data['features_batch'],
            batch_mask_gnn=None,
            graph_dict=batch_data['geo_gen'],
            node_id_all=batch_data['node_id_all'],
            edge_id_all=batch_data['edge_id_all'],
            img_batch=batch_data['image_batch']
        )
        if scaler is not None and args.task_type == 'reg':
            preds_np = preds.detach().cpu().numpy().astype(np.float64)
            # Manual inverse normalization: x * std + mean
            preds_inv = preds_np * scaler['std'] + scaler['mean']
            preds = torch.tensor(preds_inv).to(device)

            # Also inverse normalize targets to ensure RMSE is computed on the same scale
            targets_np = batch_data['targets'].detach().cpu().numpy().astype(np.float64)
            targets_inv = targets_np * scaler['std'] + scaler['mean']
            batch_data['targets'] = torch.tensor(targets_inv).to(device)

        mask = batch_data['mask'].bool()
        y_true.append(batch_data['targets'][mask])
        y_pred.append(preds[mask])

    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if args.task_type == 'class':
        result = eval_rocauc(input_dict)['rocauc']
    else:
        result = eval_rmse(input_dict)['rmse']
    return result


@torch.no_grad()
def causal_analysis_inference(args, model, scaler, intervention_df, seq_data, seq_mask, gnn_data, geo_data, image_data,
                              device, task_names=None, logger=None):
    """
    Perform inference analysis on causal intervention dataset.
    Compute prediction difference between original and intervened molecules.
    Args:
        args: Configuration
        model: Trained model
        scaler: Data scaler (for regression)
        intervention_df: Intervention dataset DataFrame
        seq_data, seq_mask, gnn_data, geo_data, image_data: Various modal data
        device: Compute device
    Returns:
        DataFrame containing inference results
    """
    model.eval()

    results = []

    print(f"Starting causal analysis inference on {len(intervention_df)} intervention samples...")

    # Create molecule index mapping (for finding corresponding preprocessed data)
    smiles_to_idx = {}
    for i, smiles in enumerate(gnn_data.smiles()):
        smiles_to_idx[smiles] = i

    for sample_idx, row in tqdm(intervention_df.iterrows(), total=len(intervention_df), desc="Causal Analysis"):
        original_smiles_str = row['original_smiles']
        intervened_smiles_str = row['intervened_smiles']
        strategy = row['intervention_strategy']

        try:
            # Find original molecule index in preprocessed data
            if original_smiles_str not in smiles_to_idx:
                print(f"Warning: Original SMILES '{original_smiles_str}' not in preprocessed data")
                continue
            original_idx = smiles_to_idx[original_smiles_str]

            # Inference on original molecule
            batch_data_original = prepare_data(args, [original_idx], seq_data, seq_mask, gnn_data, geo_data, image_data,
                                               device)
            if batch_data_original is None:
                print(f"Skipping original molecule inference: {original_smiles_str}")
                continue

            _, preds_orig = model(
                input_ids=batch_data_original['input_ids'],
                attention_mask=batch_data_original['attention_mask'],
                batch_mask_seq=None,
                gnn_batch_graph=batch_data_original['gnn_batch'],
                gnn_feature_batch=batch_data_original['features_batch'],
                batch_mask_gnn=None,
                graph_dict=batch_data_original['geo_gen'],
                node_id_all=batch_data_original['node_id_all'],
                edge_id_all=batch_data_original['edge_id_all'],
                img_batch=batch_data_original['image_batch']
            )

            # Process predictions: use scaler for regression, sigmoid for classification
            if args.task_type == 'reg':
                if scaler is not None:
                    preds_orig_np = preds_orig.detach().cpu().numpy().astype(np.float64)
                    # Manual inverse normalization: x * std + mean
                    preds_orig_inv = preds_orig_np * scaler['std'] + scaler['mean']
                    preds_orig = torch.tensor(preds_orig_inv).to(device)
            else:
                preds_orig = torch.sigmoid(preds_orig)

            # Inference on intervened molecule
            try:
                # Create temporary dataset for intervened molecule and perform full preprocessing
                from preprocess.data import MoleculeDatapoint
                from preprocess.bert_date import MolecularDataProcessor
                from utils.dataset import MoleculeDataset

                # Create datapoint for intervened molecule
                intervened_datapoint = MoleculeDatapoint(
                    line=[intervened_smiles_str, '0'],
                    args=args,
                    use_compound_names=False
                )

                # Sequence data processing (using original sequence processor)
                seq_processor = MolecularDataProcessor(args)
                intervened_input_ids, intervened_attention_mask = seq_processor.process_sequence_batch(
                    [intervened_smiles_str])
                if intervened_attention_mask.dim() == 2:
                    intervened_attention_mask = intervened_attention_mask.unsqueeze(1).unsqueeze(2)

                # Graph data processing
                from preprocess.featurization_2D import MolGraph
                from preprocess.featurization_2D import BatchMolGraph

                intervened_mol_graphs = []
                intervened_mol_graph = MolGraph(intervened_smiles_str, args)
                intervened_mol_graphs.append(intervened_mol_graph)
                intervened_gnn_batch = BatchMolGraph(intervened_mol_graphs, args)
                # Move all BatchMolGraph tensors to device
                intervened_gnn_batch.f_atoms = intervened_gnn_batch.f_atoms.to(device)
                intervened_gnn_batch.f_bonds = intervened_gnn_batch.f_bonds.to(device)
                intervened_gnn_batch.a2b = intervened_gnn_batch.a2b.to(device)
                intervened_gnn_batch.b2a = intervened_gnn_batch.b2a.to(device)
                intervened_gnn_batch.b2revb = intervened_gnn_batch.b2revb.to(device)
                intervened_gnn_batch.bonds = intervened_gnn_batch.bonds.to(device)
                intervened_gnn_batch.batch = intervened_gnn_batch.batch.to(device)

                # Geometry data processing
                from torch_geometric.data import Data, Batch
                from utils.compound_tools import mol_to_geognn_graph_data_MMFF3d

                intervened_mol = Chem.MolFromSmiles(intervened_smiles_str)
                if intervened_mol is None:
                    print(f"Cannot parse intervened molecule SMILES: {intervened_smiles_str}")
                    continue

                try:
                    # Generate 3D geometry data
                    geo_data_dict = mol_to_geognn_graph_data_MMFF3d(intervened_mol)

                    if geo_data_dict is None:
                        print(f"Cannot generate intervened molecule geometry, using virtual data: {intervened_smiles_str}")
                        # Create virtual data as fallback
                        num_atoms = intervened_mol.GetNumAtoms()
                        intervened_geo_gen, intervened_node_id_all, intervened_edge_id_all = create_virtual_geometry_data(
                            num_atoms, device)
                    else:
                        # Use real geometry data
                        atom_names = ["atomic_num", "chiral_tag", "degree",
                                      "formal_charge", "hybridization", "implicit_valence",
                                      "is_aromatic", "total_numHs"]
                        bond_names = ["bond_dir", "bond_type", "is_in_ring"]
                        bond_float_names = ["bond_length"]
                        bond_angle_float_names = ['bond_angle']

                        # Create atom-bond graph
                        ab_g = Data(
                            edge_index=torch.LongTensor(geo_data_dict['edges']).T.to(device),
                            x=torch.LongTensor(np.stack([geo_data_dict[name] for name in atom_names])).T.to(device),
                            edge_attr=torch.FloatTensor(
                                np.stack([geo_data_dict[name] for name in bond_names + bond_float_names])).T.to(device)
                        )

                        # Create bond-angle graph
                        ba_g = Data(
                            edge_index=torch.LongTensor(geo_data_dict['BondAngleGraph_edges']).T.to(device),
                            x=torch.FloatTensor(
                                np.stack([geo_data_dict[name] for name in bond_names + bond_float_names])).T.to(device),
                            edge_attr=torch.FloatTensor(
                                np.stack([geo_data_dict[name] for name in bond_angle_float_names])).T.to(device)
                        )

                        intervened_geo_gen = (Batch.from_data_list([ab_g]), Batch.from_data_list([ba_g]))

                        # Set node and edge IDs
                        intervened_node_id_all = [intervened_geo_gen[0].batch, intervened_geo_gen[1].batch]
                        intervened_edge_id_all = []
                        for i in range(intervened_geo_gen[0].num_graphs):
                            edge_batch1 = torch.ones(intervened_geo_gen[0][i].edge_index.shape[1], dtype=torch.long).to(
                                device) * i
                            intervened_edge_id_all.append(edge_batch1)
                        for i in range(intervened_geo_gen[1].num_graphs):
                            edge_batch2 = torch.ones(intervened_geo_gen[1][i].edge_index.shape[1], dtype=torch.long).to(
                                device) * i
                            intervened_edge_id_all.append(edge_batch2)

                except Exception as e:
                    print(f"Geometry data generation error: {intervened_smiles_str}, Error: {str(e)}")
                    # Create virtual data as fallback
                    num_atoms = intervened_mol.GetNumAtoms()
                    intervened_geo_gen, intervened_node_id_all, intervened_edge_id_all = create_virtual_geometry_data(
                        num_atoms, device)

                # Image data processing
                try:
                    # Generate molecular image using the same method as the original framework
                    from preprocess.data_image import Smiles2Img

                    image_size = getattr(args, 'image_size', IMAGEMOL_CONFIG['size'])
                    if isinstance(image_size, int):
                        image_size = (image_size, image_size)

                    # Generate molecular image
                    intervened_image_pil = Smiles2Img(
                        intervened_smiles_str,
                        size=image_size,
                        savePath=None,  # Do not save to file
                        quality=getattr(args, 'image_quality', 95)
                    )

                    if intervened_image_pil is not None:
                        image_transform = transforms.Compose([
                            transforms.Resize((IMAGEMOL_CONFIG['size'], IMAGEMOL_CONFIG['size'])),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=IMAGEMOL_CONFIG['mean'],
                                std=IMAGEMOL_CONFIG['std']
                            )
                        ])
                        intervened_image_tensor = image_transform(intervened_image_pil)
                        intervened_image_batch = intervened_image_tensor.unsqueeze(0).to(device)
                    else:
                        # If image generation fails, use virtual image
                        print(f"Warning: Cannot generate intervened molecule image, using virtual image: {intervened_smiles_str}")
                        intervened_image_batch = torch.zeros((1, 3, IMAGEMOL_CONFIG['size'], IMAGEMOL_CONFIG['size']),
                                                             dtype=torch.float).to(device)
                except Exception as e:
                    print(f"Image generation error: {intervened_smiles_str}, Error: {str(e)}")
                    intervened_image_batch = torch.zeros((1, 3, IMAGEMOL_CONFIG['size'], IMAGEMOL_CONFIG['size']),
                                                         dtype=torch.float).to(device)

                # Prepare complete batch data
                intervened_batch_data = {
                    'input_ids': intervened_input_ids.to(device),
                    'attention_mask': intervened_attention_mask.to(device),
                    'gnn_batch': intervened_gnn_batch,
                    'features_batch': None,
                    'geo_gen': intervened_geo_gen,
                    'node_id_all': intervened_node_id_all,
                    'edge_id_all': intervened_edge_id_all,
                    'image_batch': intervened_image_batch,
                    'mask': torch.tensor([[True]], dtype=torch.bool).to(device),
                    'targets': torch.tensor([[0.0]], dtype=torch.float).to(device),
                    'smiles': [intervened_smiles_str]
                }

                # Inference on intervened molecule
                _, preds_int = model(
                    input_ids=intervened_batch_data['input_ids'],
                    attention_mask=intervened_batch_data['attention_mask'],
                    batch_mask_seq=None,
                    gnn_batch_graph=intervened_batch_data['gnn_batch'],
                    gnn_feature_batch=intervened_batch_data['features_batch'],
                    batch_mask_gnn=None,
                    graph_dict=intervened_batch_data['geo_gen'],
                    node_id_all=intervened_batch_data['node_id_all'],
                    edge_id_all=intervened_batch_data['edge_id_all'],
                    img_batch=intervened_batch_data['image_batch']
                )

                # Process predictions: use scaler for regression, sigmoid for classification
                if args.task_type == 'reg':
                    if scaler is not None:
                        preds_int_np = preds_int.detach().cpu().numpy().astype(np.float64)
                        # Manual inverse normalization: x * std + mean
                        preds_int_inv = preds_int_np * scaler['std'] + scaler['mean']
                        preds_int = torch.tensor(preds_int_inv).to(device)
                else:
                    preds_int = torch.sigmoid(preds_int)

                # Calculate prediction difference
                # Handle multi-task prediction results
                if preds_orig.dim() > 0 and preds_orig.numel() > 1:
                    # Multi-task case: calculate various metrics for prediction vector
                    if preds_orig.dim() == 1:
                        num_tasks = preds_orig.numel()
                    else:
                        num_tasks = preds_orig.shape[-1]  # Last dimension is task count
                    # Calculate prediction difference for each subtask
                    task_results = {}
                    if task_names and len(task_names) == num_tasks:
                        task_name_list = task_names
                    else:
                        task_name_list = [f'task_{i}' for i in range(num_tasks)]
                        print(f"Processing multi-task dataset with {num_tasks} subtasks: {', '.join(task_name_list)}")
                        logger.info(f"Processing multi-task dataset with {num_tasks} subtasks: {', '.join(task_name_list)}")

                    if args.task_type == 'reg':
                        # Regression task: calculate prediction difference for each subtask
                        for task_idx, task_name in enumerate(task_name_list):
                            if preds_orig.dim() == 1:
                                orig_val = preds_orig[task_idx].item()
                                int_val = preds_int[task_idx].item()
                            else:
                                orig_val = preds_orig[0, task_idx].item()
                                int_val = preds_int[0, task_idx].item()
                            diff_val = int_val - orig_val
                            task_results[task_name] = {
                                'original': orig_val,
                                'intervened': int_val,
                                'difference': diff_val,
                                'abs_difference': abs(diff_val)
                            }

                        # Overall statistics
                        pred_diff = (preds_int.mean() - preds_orig.mean()).item()
                        abs_pred_diff = abs(pred_diff)

                        result = {
                            'original_smiles': original_smiles_str,
                            'intervened_smiles': intervened_smiles_str,
                            'intervention_strategy': strategy,
                            'original_molecule_idx': original_idx,
                            'original_prediction': preds_orig.mean().item(),  # Mean prediction
                            'intervened_prediction': preds_int.mean().item(),
                            'prediction_difference': pred_diff,
                            'abs_prediction_difference': abs_pred_diff,
                            'original_prediction_vector': preds_orig.cpu().numpy().tolist(),  # Full prediction vector
                            'intervened_prediction_vector': preds_int.cpu().numpy().tolist(),
                            'prediction_vector_difference': (preds_int - preds_orig).cpu().numpy().tolist(),
                            'per_task_results': task_results  # Detailed results for each subtask
                        }
                    else:
                        # Classification task: calculate probability difference for each subtask
                        for task_idx, task_name in enumerate(task_name_list):
                            if preds_orig.dim() == 1:
                                orig_prob = preds_orig[task_idx].item()
                                int_prob = preds_int[task_idx].item()
                            else:
                                orig_prob = preds_orig[0, task_idx].item()
                                int_prob = preds_int[0, task_idx].item()
                            prob_diff = int_prob - orig_prob
                            task_results[task_name] = {
                                'original_probability': orig_prob,
                                'intervened_probability': int_prob,
                                'probability_difference': prob_diff,
                                'abs_probability_difference': abs(prob_diff)
                            }

                        # Overall statistics
                        pred_diff = (preds_int.mean() - preds_orig.mean()).item()
                        abs_pred_diff = abs(pred_diff)

                        # Calculate prediction probability change statistics
                        prob_changes = preds_int - preds_orig
                        max_abs_change = prob_changes.abs().max().item()
                        mean_abs_change = prob_changes.abs().mean().item()

                        result = {
                            'original_smiles': original_smiles_str,
                            'intervened_smiles': intervened_smiles_str,
                            'intervention_strategy': strategy,
                            'original_molecule_idx': original_idx,
                            'original_prediction': preds_orig.mean().item(),  # Mean probability
                            'intervened_prediction': preds_int.mean().item(),
                            'prediction_difference': pred_diff,
                            'abs_prediction_difference': abs_pred_diff,
                            'max_abs_probability_change': max_abs_change,  # Max absolute probability change
                            'mean_abs_probability_change': mean_abs_change,  # Mean absolute probability change
                            'original_probability_vector': preds_orig.cpu().numpy().tolist(),
                            'intervened_probability_vector': preds_int.cpu().numpy().tolist(),
                            'probability_difference_vector': prob_changes.cpu().numpy().tolist(),
                            'per_task_results': task_results  # Detailed results for each subtask
                        }

                        # Add probability-related fields
                        result['original_probability'] = preds_orig.mean().item()
                        result['intervened_probability'] = preds_int.mean().item()
                        result['probability_difference'] = pred_diff
                else:
                    # Single task case: directly convert to scalar
                    pred_diff = preds_int.item() - preds_orig.item()

                    result = {
                        'original_smiles': original_smiles_str,
                        'intervened_smiles': intervened_smiles_str,
                        'intervention_strategy': strategy,
                        'original_molecule_idx': original_idx,
                        'original_prediction': preds_orig.item(),
                        'intervened_prediction': preds_int.item(),
                        'prediction_difference': pred_diff,
                        'abs_prediction_difference': abs(pred_diff)
                    }

                    # Add probability fields for classification task
                    if args.task_type == 'class':
                        result['original_probability'] = preds_orig.item()
                        result['intervened_probability'] = preds_int.item()
                        result['probability_difference'] = pred_diff

                # Add chemical property information
                if 'original_logp' in row:
                    result['original_logp'] = row['original_logp']
                    result['intervened_logp'] = row['intervened_logp']
                    result['logp_difference'] = row['intervened_logp'] - row['original_logp']

                results.append(result)

            except Exception as e:
                print(f"Intervened molecule inference failed: {intervened_smiles_str}, Error: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        except Exception as e:
            print(f"Sample {sample_idx} inference failed: {str(e)}")
            continue

    results_df = pd.DataFrame(results)
    print(f"Causal analysis inference completed, processed {len(results_df)} samples")
    return results_df


def bootstrap_confidence_interval(ate_values, n_bootstrap=10000, alpha=0.05):
    """
    Calculate confidence interval for ATE mean using bootstrap method.
    Args:
        ate_values (list): List of ATE values from independent runs
        n_bootstrap (int): Number of bootstrap resampling iterations
        alpha (float): Significance level

    Returns:
        tuple: (lower_ci, upper_ci, is_significant)
    """
    if len(ate_values) < 2:
        # Insufficient data for bootstrap
        mean_val = np.mean(ate_values) if ate_values else 0.0
        return mean_val, mean_val, False

    ate_values = np.array(ate_values)
    n = len(ate_values)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Sampling with replacement
        resample = np.random.choice(ate_values, size=n, replace=True)
        bootstrap_means.append(np.mean(resample))

    # Calculate percentiles
    lower_ci = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_ci = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    # Determine significance: CI does not contain 0
    is_significant = lower_ci > 0 or upper_ci < 0

    return lower_ci, upper_ci, is_significant


def setup_logger(logs_file):
    logger = logging.getLogger()
    # Ensure the log file is written using UTF-8 to avoid encoding errors (e.g., subscript numbers)
    handler = logging.FileHandler(logs_file, encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def generate_random_seed():
    random_seed = random.randint(0, 10000)
    print(f"Generated random seed: {random_seed}")
    return random_seed


def main(args):
    from collections import defaultdict
    global smiles, image_filenames, image_labels, save_model, seed
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up log directories and files (keep existing folder naming)
    logs_dir = f'./LOG/{args.dataset}/{args.lr}_{args.epochs}_{args.batch_size}_{args.fusion}/'
    os.makedirs(logs_dir, exist_ok=True)
    # Determine active modalities and include them in the log file name
    active_mods = []
    try:
        if getattr(args, 'sequence', False):
            active_mods.append('sequence')
        if getattr(args, 'graph', False):
            active_mods.append('graph')
        if getattr(args, 'geometry', False):
            active_mods.append('geometry')
        if getattr(args, 'image', False):
            active_mods.append('image')
    except Exception:
        active_mods = []
    modal_str = '+'.join(active_mods) if len(active_mods) > 0 else 'none'
    # Add test set ratio to log filename
    test_ratio_str = f"test{args.split_sizes[2]}"
    logs_file = os.path.join(logs_dir, f"{current_time}_{modal_str}_{test_ratio_str}.log")
    logger = setup_logger(logs_file)
    logger.info(f"Starting training with parameters: {vars(args)}")

    if args.fixed_seed:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Using fixed seed: {seed}")

    best_overall_result = None
    best_overall_test = None
    best_overall_epoch = 0
    all_test_results = []

    # container to collect per-run ATEs for each intervention strategy
    per_strategy_ates = defaultdict(list)
    # container to collect per-run ATEs for each task within each strategy
    per_task_ates = defaultdict(lambda: defaultdict(list))

    # collect saved model paths for the current invocation (to avoid picking up old checkpoints)
    saved_model_files_current_run = []

    # Clean up old model files from previous runs (optional)
    import glob
    old_model_files = glob.glob(f'./saved_models/{args.dataset}/*.pt')
    if old_model_files:
        print(f"Cleaning up {len(old_model_files)} old model files...")
        for old_file in old_model_files:
            try:
                os.remove(old_file)
            except Exception as e:
                print(f"Failed to delete old file {old_file}: {e}")
        print("Old model files cleaned up")

    train_losses = []
    val_losses = []
    all_predictions = []
    all_labels = []
    # per-run causal results collected immediately after each run
    per_run_results = []

    # Predefine display order and names once
    strategy_display_order = [
        'aromatic_chloro', 'fluoro_aromatic', 'carboxylic_acid',
        'hydroxyl', 'sulfonamide', 'amide'
    ]
    strategy_display_names = {
        'aromatic_chloro': 'Ar-Cl→H',
        'fluoro_aromatic': 'Ar-F→H',
        'carboxylic_acid': '-COOH→H',
        'hydroxyl': '-OH→H',
        'sulfonamide': '-SO₂NH→H',
        'amide': '-CONH→H'
    }

    for run in range(args.num_runs):
        # Ensure each run is completely independent - clean up any historical model state
        if 'save_model' in locals():
            del save_model

        if not args.fixed_seed:
            seed = generate_random_seed()

        # Set random seeds for this run
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        logger.info(f"-----------------------Run {run + 1}/{args.num_runs}-----------------------------")
        logger.info(f"seed:[{seed}]")

        # Device initialization
        device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.cuda else 'cpu')
        torch.cuda.empty_cache()
        logger.info(f"Device set to: {device}")

        # Record hyperparameters
        hyperparams_log = (f"lr: {args.lr}, cl_loss: {args.cl_loss}, cl_loss_num: {args.cl_loss_num}, "
                           f"pro_num: {args.pro_num}, pool_type: {args.pool_type}, "
                           f"gnn_hidden_dim: {args.gnn_hidden_dim}, batch_size: {args.batch_size}, "
                           f"norm: {args.norm}, fusion: {args.fusion}")
        logger.info(hyperparams_log)

        # Data loaded
        data_path = f'data/{args.dataset}/{args.dataset}.csv'
        datas, args.seq_len, task_names = get_data(args, path=data_path)

        # For multi-task datasets like sider, ensure num_tasks is correctly set
        actual_num_tasks = max(datas.num_tasks(), len(task_names))
        print(f"INFO: Detected {actual_num_tasks} tasks for dataset {args.dataset}")
        args.output_dim = args.num_tasks = actual_num_tasks

        # Processing the SMILES sequence
        logger.debug("Processing SMILES sequences...")
        smiles = datas.smiles()
        logger.debug(f"Original dataset size: {len(datas)}, SMILES count: {len(smiles)}")

        processor = MolecularDataProcessor(args)
        input_ids, attention_mask = processor.process_sequence_batch(smiles)
        logger.debug(f"Sequence processed size: input_ids.shape={input_ids.shape}, attention_mask.shape={attention_mask.shape}")

        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        seq_data = input_ids
        seq_mask = attention_mask

        # Check data consistency
        if len(seq_data) != len(datas):
            logger.warning(f"Data length mismatch! seq_data: {len(seq_data)}, datas: {len(datas)}")
            valid_indices = list(range(min(len(seq_data), len(datas))))
            logger.debug(f"Using valid index range: 0-{len(valid_indices)-1}")
        else:
            logger.debug("Data length consistent, check passed")

        # Processing 2D map data
        logger.debug("Processing 2D molecular data...")
        args.gnn_atom_dim = get_atom_fdim()
        args.gnn_bond_dim = get_bond_fdim()

        logger.debug(f"Atom feature dimension: {args.gnn_atom_dim}")
        logger.debug(f"Bond feature dimension: {args.gnn_bond_dim}")

        mol_graphs = []
        for smi in datas.smiles():
            mol_graph = MolGraph(smi, args)
            mol_graphs.append(mol_graph)

        # Processing 3D geometry data
        logger.debug("Loading 3D molecular data...")
        npz_data_path = os.path.join('data', args.dataset)
        try:
            data_processor = DataProcessor(args, device)

            if args.process_3d or not data_processor.check_processed_data():
                logger.debug("Processing 3D data...")
                data_3d = data_processor.process_3d_data()
                data_3d.get_data(device)
            else:
                logger.debug("Loading processed 3D data...")
                data_3d = InMemoryDataset(
                    npz_data_path=os.path.join(npz_data_path)
                )
                data_3d.get_data(device)
        except Exception as e:
            logger.error(f"Error processing 3D data:{str(e)}")
            raise

        # Processing image data
        logger.debug("Processing image data...")
        smiles = datas.smiles()
        load_or_generate_images(args, smiles)
        process_dataset(args)
        generate_labels_file(args)

        image_folder = f'./data/{args.dataset}/image'
        image_labels_csv = f'./data/{args.dataset}/processed/{args.dataset}_label.csv'
        image_filenames, image_labels = load_filenames_and_labels(args, image_folder, image_labels_csv)

        image_transform = transforms.Compose([
            transforms.Resize((IMAGEMOL_CONFIG['size'], IMAGEMOL_CONFIG['size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGEMOL_CONFIG['mean'],
                std=IMAGEMOL_CONFIG['std']
            )
        ])

        image_data = ImageDataset(
            filenames=image_filenames,
            labels=image_labels,
            img_transformer=image_transform,
            args=args
        )

        # Data segmentation - split data first
        train_data, val_data, test_data = split_data(data=datas, split_type=args.split_type, sizes=args.split_sizes,
                                                     seed=seed, args=args)
        train_idx = [data.idx for data in train_data]
        val_idx = [data.idx for data in val_data]
        test_idx = [data.idx for data in test_data]

        # Data consistency check
        lengths = {
            'Original Dataset': len(datas),
            'Sequence Data': len(seq_data),
            '2D Graph Data': len(mol_graphs),
            '3D Geometry Data': len(data_3d),
            'Image Data': len(image_data)
        }

        logger.debug("Modal data sizes:")
        for name, length in lengths.items():
            logger.debug(f"  {name}: {length}")

        if len(set(lengths.values())) == 1:
            max_valid_idx = len(datas) - 1
        else:
            min_length = min(lengths.values())
            max_valid_idx = min_length - 1
            logger.warning(f"Using minimum length {min_length} as valid data range")

        train_idx = [idx for idx in train_idx if idx <= max_valid_idx]
        val_idx = [idx for idx in val_idx if idx <= max_valid_idx]
        test_idx = [idx for idx in test_idx if idx <= max_valid_idx]

        logger.debug(f"Filtered data sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        logger.debug(f"Max valid index: {max_valid_idx}")

        if not train_idx:
            raise ValueError("Training set is empty! Please check data processing.")
        if not val_idx:
            raise ValueError("Validation set is empty! Please check data processing.")
        if not test_idx:
            raise ValueError("Test set is empty! Please check data processing.")

        train_sampler = RandomSampler(train_idx)
        val_sampler = BatchSampler(val_idx, batch_size=args.batch_size, drop_last=True)
        test_sampler = BatchSampler(test_idx, batch_size=args.batch_size, drop_last=False)
        train_idx_loader = DataLoader(train_idx, batch_size=args.batch_size, sampler=train_sampler)

        # Task information processing
        scaler = None  # Initialize scaler first
        if args.task_type == 'class':
            class_sizes = get_class_sizes(datas)
            for i, task_class_sizes in enumerate(class_sizes):
                print(f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
        elif args.task_type == 'reg':
            all_targets = datas.targets()
            all_targets = np.array(all_targets, dtype=np.float64)

            valid_mask = ~np.isnan(all_targets)
            if not np.any(valid_mask):
                scaler = None
                print("Warning: All target values are NaN; standardization cannot be performed")
            else:
                valid_targets = all_targets[valid_mask]
                mean_val = np.mean(valid_targets)
                std_val = np.std(valid_targets, ddof=1) if len(valid_targets) > 1 else 1.0

                #(x - mean) / std
                scaled_targets = np.full_like(all_targets, np.nan)
                scaled_targets[valid_mask] = (valid_targets - mean_val) / std_val
                scaled_targets = scaled_targets.tolist()

                scaler = {'mean': mean_val, 'std': std_val}

            for i, target in enumerate(scaled_targets):
                if not np.isnan(target):
                    datas[i].set_targets(target)

        model = Multi_modal(args, device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")

        optimizer = Adam(params=model.parameters(), lr=args.init_lr, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.final_lr)

        ids = list(range(len(train_data)))
        print('Training model ...')
        best_result = None
        best_test = None
        best_epoch = 0
        save_model = None

        for epoch in range(args.epochs):
            np.random.shuffle(ids)
            args.current_epoch = epoch

            train_all_loss, train_label_loss, train_cl_loss, train_preds, train_labels = train(
                args, model, optimizer, train_idx_loader, seq_data, seq_mask, datas,
                data_3d, image_data, device, epoch
            )
            train_losses.append(train_all_loss)

            model.eval()
            val_result, val_all_loss, val_label_loss, val_cl_loss, val_preds, val_labels = val(
                args, model, scaler, val_sampler, seq_data,
                seq_mask, datas, data_3d, image_data, device
            )
            val_losses.append(val_all_loss)

            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

            all_predictions.extend(train_preds.cpu().numpy())
            all_labels.extend(train_labels.cpu().numpy())

            if best_result is None:
                best_result = val_result
                save_model = copy.deepcopy(model)
            else:
                if args.task_type == 'class' and val_result > best_result:
                        best_result = val_result
                        save_model = copy.deepcopy(model)
                elif args.task_type == 'reg' and val_result < best_result:
                        best_result = val_result
                        save_model = copy.deepcopy(model)

            result = test(args, save_model, scaler, test_sampler, seq_data, seq_mask, datas, data_3d, image_data, device)

            if best_test is None:
                best_test = result
                best_epoch = epoch
                best_run_seed = seed
            else:
                if (args.task_type == 'class' and result > best_test) or (args.task_type == 'reg' and result < best_test):
                    best_test = result
                    best_epoch = epoch
                    best_run_seed = seed

            logger.info(
                f"Epoch: {epoch + 1}, lr: {current_lr:.6f}, train_all_loss: {train_all_loss:.4f}, "
                f"val_all_loss: {val_all_loss:.4f}, Best Test Result: {best_test:.4f}"
            )

            torch.cuda.empty_cache()

        logger.info(f"Final Test Result: {best_test:.4f} at epoch {best_epoch + 1}")
        all_test_results.append(best_test)

        if best_overall_result is None:
            best_overall_result = best_result
            best_overall_test = best_test
            best_overall_epoch = best_epoch
        else:
            if (args.task_type == 'class' and best_overall_result < best_result) or \
                    (args.task_type == 'reg' and best_overall_result > best_result):
                best_overall_result = best_result
                best_overall_test = best_test
                best_overall_epoch = best_epoch

        # Save best model of this run
        os.makedirs(f'./saved_models/{args.dataset}', exist_ok=True)
        test_value_str = f"{best_test:.4f}"
        model_filename = f"{test_value_str}_run_{run}_best_model.pt"
        save_path = os.path.join(f'./saved_models/{args.dataset}', model_filename)
        torch.save({
            'model_state_dict': save_model.state_dict() if save_model is not None else model.state_dict(),
            'args': args,
            'run': run,
            'best_val_result': best_result,
            'best_test_result': best_test,
            'best_epoch': best_epoch
        }, save_path)
        saved_model_files_current_run.append(save_path)

        # Immediately run causal inference for this run using its best model
        try:
            from Causal_Intervention import CausalInterventionGenerator
            intervention_generator = CausalInterventionGenerator(chemical_validity_threshold=getattr(args, 'intervention_threshold', 0.7))
        except Exception:
            from Causal_Intervention import CausalInterventionGenerator
            intervention_generator = CausalInterventionGenerator(getattr(args, 'chemical_threshold', 0.7))

        dataset_path = f'data/{args.dataset}/{args.dataset}.csv'
        output_dir = f'data/{args.dataset}'
        try:
            intervention_df_run = intervention_generator.generate_intervention_dataset(
            dataset_path=dataset_path,
            output_dir=output_dir,
            strategies=getattr(args, 'intervention_strategies', None),
                max_samples=getattr(args, 'intervention_max_samples', None),
                test_indices=test_idx
            )
            logger.info(f"Run {run}: Generated intervention dataset with {len(intervention_df_run)} samples")
            if len(intervention_df_run) == 0:
                logger.warning(f"Run {run}: No intervention samples generated - check if test set contains suitable molecules for intervention")
        except Exception as e:
            logger.warning(f"Failed to generate intervention dataset for run {run}: {e}")
            intervention_df_run = pd.DataFrame()

        if len(intervention_df_run) > 0:
            logger.info(f"Run {run}: Starting causal analysis inference on {len(intervention_df_run)} intervention samples")
            try:
                df_run = causal_analysis_inference(args, save_model, scaler, intervention_df_run, seq_data, seq_mask, datas, data_3d, image_data, device, task_names, logger)
                logger.info(f"Run {run}: Causal analysis completed, got {len(df_run)} results")
                if len(df_run) > 0:
                    run_path = os.path.join(output_dir, f"Causal_Analysis_{args.dataset}_run{run}.csv")
                    df_run.to_csv(run_path, index=False)
                    logger.info(f"Saved per-run causal results to {run_path}")
                    per_run_results.append(df_run)
                    # collect ATEs per strategy for this run
                    for strategy, grp in df_run.groupby('intervention_strategy'):
                        if args.task_type == 'class':
                            ate_val = grp['probability_difference'].mean()
                        else:
                            ate_val = grp['prediction_difference'].mean()
                        per_strategy_ates[strategy].append(float(ate_val))
                        logger.info(f"Run {run}: Strategy {strategy} - ATE = {ate_val:.4f} from {len(grp)} samples")

                        # collect per-task ATEs for multi-task datasets
                        if 'per_task_results' in grp.columns and len(grp) > 0:
                            # Check if first row has per_task_results
                            first_row_tasks = grp['per_task_results'].iloc[0]
                            if isinstance(first_row_tasks, dict):
                                # Get all task names from the first row
                                task_names_in_data = list(first_row_tasks.keys())
                                for task_name in task_names_in_data:
                                    task_ates_for_strategy = []
                                    for _, row in grp.iterrows():
                                        if isinstance(row['per_task_results'], dict) and task_name in row['per_task_results']:
                                            if args.task_type == 'class':
                                                task_ate_val = row['per_task_results'][task_name]['probability_difference']
                                            else:
                                                task_ate_val = row['per_task_results'][task_name]['difference']
                                            task_ates_for_strategy.append(float(task_ate_val))
                                    if task_ates_for_strategy:
                                        # Calculate mean ATE for this task across all samples in this strategy
                                        mean_task_ate = np.mean(task_ates_for_strategy)
                                        per_task_ates[strategy][task_name].append(float(mean_task_ate))
                else:
                    logger.warning(f"Run {run}: Causal analysis returned no results")
            except Exception as e:
                logger.error(f"Run {run}: Causal analysis inference failed: {e}")
        else:
            logger.warning(f"Run {run}: Skipping causal analysis - no intervention data generated")

    # Calculate mean and standard deviation
    average_test_result = sum(all_test_results) / len(all_test_results) if all_test_results else 0.0
    std_dev_test_result = statistics.stdev(all_test_results) if len(all_test_results) > 1 else 0.0

    # Output all test results
    logger.info("All Test Results: " + ", ".join(f"{result:.4f}" for result in all_test_results))
    logger.info(f"Average Test Result across all runs: {average_test_result:.4f}")
    logger.info(f"Standard Deviation of Test Results: {std_dev_test_result:.4f}")

    # Record best model run information
    if 'best_run_seed' in locals():
        logger.info(f"Best model from run (seed={best_run_seed})")
    else:
        logger.info("Best model information not recorded")

    # Aggregate cross-run ATE for each strategy
    if per_strategy_ates:
        print("\nCross-run ATE Summary (per strategy):")
        logger.info("Cross-run ATE summary (per strategy):")

        if len(per_run_results) > 0:
            try:
                _df_runs_concat = pd.concat(per_run_results, ignore_index=True)
                _strategy_counts = _df_runs_concat['intervention_strategy'].value_counts().to_dict()
            except Exception:
                _strategy_counts = {}
        else:
            _strategy_counts = {}

        for strategy in strategy_display_order:
            ates = per_strategy_ates.get(strategy, [])
            display_name = strategy_display_names.get(strategy, strategy)
            if len(ates) == 0:
                print(f"{display_name}: No data")
                logger.info(f"{display_name}: no data")
                continue
            mean_ate = float(np.mean(ates))
            std_ate = float(np.std(ates, ddof=1)) if len(ates) > 1 else 0.0
            k_runs = len(ates)
            k_samples = int(_strategy_counts.get(strategy, 0))

            # Bootstrap confidence interval calculation
            lower_ci, upper_ci, is_significant = bootstrap_confidence_interval(ates, n_bootstrap=10000, alpha=0.05)
            significance_marker = "***" if is_significant else ""

            print(f"{display_name}: mean={mean_ate:.4f}, std={std_ate:.4f}, 95%CI=[{lower_ci:.4f}, {upper_ci:.4f}]{significance_marker}, K_runs={k_runs}, K_samples={k_samples}")
            logger.info(f"{display_name}: mean={mean_ate:.4f}, std={std_ate:.4f}, 95%CI=[{lower_ci:.4f}, {upper_ci:.4f}]{significance_marker}, K_runs={k_runs}, K_samples={k_samples}")

            # Additional significance output
            if is_significant:
                direction = "positive" if mean_ate > 0 else "negative"
                logger.info(f"{display_name}: Causal effect is significant ({direction} effect, 95%CI does not contain 0)")
            else:
                logger.info(f"{display_name}: Causal effect is not significant (95%CI contains 0)")

            # Output per-task ATE statistics (if available)
            if strategy in per_task_ates and per_task_ates[strategy]:
                print(f"\n{display_name} Per-task ATE Details:")
                logger.info(f"{display_name} per-task ATE details:")
                for task_name, task_ates in per_task_ates[strategy].items():
                    if len(task_ates) > 0:
                        task_mean_ate = float(np.mean(task_ates))
                        task_std_ate = float(np.std(task_ates, ddof=1)) if len(task_ates) > 1 else 0.0
                        task_lower_ci, task_upper_ci, task_is_significant = bootstrap_confidence_interval(task_ates, n_bootstrap=10000, alpha=0.05)
                        task_significance_marker = "***" if task_is_significant else ""

                        print(f"  {task_name}: mean={task_mean_ate:.4f}, std={task_std_ate:.4f}, 95%CI=[{task_lower_ci:.4f}, {task_upper_ci:.4f}]{task_significance_marker}")
                        logger.info(f"  {task_name}: mean={task_mean_ate:.4f}, std={task_std_ate:.4f}, 95%CI=[{task_lower_ci:.4f}, {task_upper_ci:.4f}]{task_significance_marker}")

                        if task_is_significant:
                            task_direction = "positive" if task_mean_ate > 0 else "negative"
                            logger.info(f"  {task_name}: Causal effect is significant ({task_direction} effect, 95%CI does not contain 0)")
                        else:
                            logger.info(f"  {task_name}: Causal effect is not significant (95%CI contains 0)")

                # Separate output for each subtask significance summary
                significant_tasks = []
                for task_name, task_ates in per_task_ates[strategy].items():
                    if len(task_ates) > 0:
                        task_mean_ate = float(np.mean(task_ates))
                        task_lower_ci, task_upper_ci, task_is_significant = bootstrap_confidence_interval(task_ates, n_bootstrap=10000, alpha=0.05)
                        if task_is_significant:
                            task_direction = "positive" if task_mean_ate > 0 else "negative"
                            significant_tasks.append(f"{task_name}({task_direction})")

                if significant_tasks:
                    print(f"  {display_name} Significant tasks: {', '.join(significant_tasks)}")
                    logger.info(f"  {display_name} Significant tasks: {', '.join(significant_tasks)}")

    # Causal intervention analysis
    if args.do_causal_intervention:
        print("\nStarting causal intervention analysis...")
        # If per-run results already exist, merge and use directly
        if len(per_run_results) > 0:
            causal_results_df = pd.concat(per_run_results, ignore_index=True)
            causal_output_path = os.path.join(f"data/{args.dataset}", f'Causal_Analysis_{args.dataset}.csv')
            causal_results_df.to_csv(causal_output_path, index=False)
            logger.info(f"Merged causal analysis results saved to: {causal_output_path}")
            skip_generation = True
        else:
            skip_generation = False

        if not skip_generation:
            try:
                from Causal_Intervention import CausalInterventionGenerator
                intervention_generator = CausalInterventionGenerator(
                    chemical_validity_threshold=getattr(args, 'chemical_threshold', 0.7)
                )

                dataset_path = f'data/{args.dataset}/{args.dataset}.csv'
                output_dir = f'data/{args.dataset}'

                intervention_datas, _, _ = get_data(args, path=dataset_path)
                causal_seed = best_run_seed if 'best_run_seed' in locals() else seed

                _, _, intervention_test_data = split_data(
                    data=intervention_datas,
                    split_type=args.split_type,
                    sizes=args.split_sizes,
                    seed=causal_seed,
            args=args
        )
                intervention_test_idx = [data.idx for data in intervention_test_data]

                intervention_df = intervention_generator.generate_intervention_dataset(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    strategies=getattr(args, 'intervention_strategies', None),
                    max_samples=getattr(args, 'intervention_max_samples', None),
                    test_indices=intervention_test_idx
                )

                if len(intervention_df) == 0:
                    print("No valid causal intervention samples generated")
                else:
                    print("\nStarting causal analysis inference...")
                    logger.info("Starting causal analysis inference...")
                    global_causal_results_df = causal_analysis_inference(
                        args, save_model, scaler, intervention_df, seq_data, seq_mask, datas, data_3d, image_data, device, task_names, logger
                    )
                    if len(global_causal_results_df) > 0:
                        global_output_path = os.path.join(output_dir, f'Causal_Analysis_{args.dataset}.csv')
                        global_causal_results_df.to_csv(global_output_path, index=False)
                        print(f"Causal analysis results saved to: {global_output_path}")
                        logger.info(f"Causal analysis results saved to: {global_output_path}")
                    else:
                        logger.warning("Causal analysis produced no valid results")

            except Exception as e:
                logger.error(f"Causal intervention analysis failed: {e}", exc_info=True)

        # If merged causal_results_df or global_causal_results_df exists, show statistics and significant samples
        if ('causal_results_df' in locals() and not causal_results_df.empty) or \
           ('global_causal_results_df' in locals() and len(global_causal_results_df) > 0):
            if 'causal_results_df' in locals() and not causal_results_df.empty:
                df = causal_results_df.copy()
            else:
                df = global_causal_results_df.copy()
            print(f"\nCausal Analysis Statistics (Total {len(df)} samples):")
            logger.info(f"Causal analysis statistics (Total {len(df)} samples)")

            if args.task_type == 'class':
                diff_col = 'probability_difference'
                abs_diff_col = 'abs_prediction_difference'
            else:
                diff_col = 'prediction_difference'
                abs_diff_col = 'abs_prediction_difference'

            # per-strategy stats
            try:
                strategy_stats = df.groupby('intervention_strategy').agg({
                    diff_col: ['mean', 'std', 'count'],
                    abs_diff_col: ['mean', 'max']
                }).round(4)

                # Calculate per-task statistics (if per_task_results exists)
                per_task_strategy_stats = {}
                if 'per_task_results' in df.columns:
                    for strategy in df['intervention_strategy'].unique():
                        strategy_data = df[df['intervention_strategy'] == strategy]
                        task_stats = {}
                        for _, row in strategy_data.iterrows():
                            if isinstance(row.get('per_task_results'), dict):
                                for task_name, task_data in row['per_task_results'].items():
                                    if task_name not in task_stats:
                                        task_stats[task_name] = []
                                    if args.task_type == 'class':
                                        task_stats[task_name].append(task_data.get('probability_difference', 0))
                                    else:
                                        task_stats[task_name].append(task_data.get('difference', 0))

                        per_task_strategy_stats[strategy] = {}
                        for task_name, diffs in task_stats.items():
                            if diffs:
                                per_task_strategy_stats[strategy][task_name] = {
                                    'mean': np.mean(diffs),
                                    'std': np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0,
                                    'count': len(diffs),
                                    'max_abs': max(abs(x) for x in diffs)
                                }
            except Exception:
                strategy_stats = pd.DataFrame()
                per_task_strategy_stats = {}

            # Show sample details for each strategy
            for strategy in strategy_display_order:
                strategy_samples = df[df['intervention_strategy'] == strategy]
                display_name = strategy_display_names.get(strategy, strategy)
                print(f"\nChecking strategy {strategy}: samples={len(strategy_samples)}")
                if len(strategy_samples) == 0:
                    print(f"{display_name}: No data")
                    logger.info(f"{display_name} - No data")
                    continue

                print(f"{display_name} Intervention Samples:")
                logger.info(f"{display_name} Intervention Samples:")

                count = 0
                for _, row in strategy_samples.iterrows():
                    mol_idx = int(row.get('original_molecule_idx', -1))
                    orig_smiles = row.get('original_smiles', 'N/A')
                    interv_smiles = row.get('intervened_smiles', 'N/A')

                    if args.task_type == 'class':
                        prob_orig = row.get('original_probability', np.nan)
                        prob_int = row.get('intervened_probability', np.nan)
                        prob_diff = row.get('probability_difference', np.nan)

                        # Only log SMILES to logger, keep console output concise
                        logger.info(f"Original molecule index {mol_idx}: {orig_smiles}")
                        logger.info(f"Intervened molecule: {interv_smiles}")
                        logger.info(f"Prediction change: mean prob {prob_diff:.4f} ({prob_orig:.4f} -> {prob_int:.4f})")

                        print(f"Idx {mol_idx:4d}: mean prob orig={prob_orig:.4f}, interv={prob_int:.4f}, change={prob_diff:.4f}")

                        # Show per-task probability changes
                        if 'per_task_results' in row and isinstance(row['per_task_results'], dict):
                            print(f"Per-task probability details:")
                            logger.info(f"Per-task probability details:")
                            for task_name, task_data in row['per_task_results'].items():
                                task_orig = task_data.get('original_probability', np.nan)
                                task_int = task_data.get('intervened_probability', np.nan)
                                task_diff = task_data.get('probability_difference', np.nan)
                                print(f"{task_name}: {task_orig:.4f} -> {task_int:.4f} (change={task_diff:.4f})")
                                logger.info(f"{task_name}: {task_orig:.4f} -> {task_int:.4f} (change={task_diff:.4f})")
                    else:
                        orig = row.get('original_prediction', np.nan)
                        interv = row.get('intervened_prediction', np.nan)
                        diff = row.get('prediction_difference', np.nan)

                        # Only log SMILES to logger, keep console output concise
                        logger.info(f"Original molecule index {mol_idx}: {orig_smiles}")
                        logger.info(f"Intervened molecule: {interv_smiles}")
                        logger.info(f"Prediction change: mean value {diff:.4f} ({orig:.4f} -> {interv:.4f})")

                        print(f"Idx {mol_idx:4d}: mean pred orig={orig:.4f}, interv={interv:.4f}, change={diff:.4f}")

                        # Show per-task prediction changes
                        if 'per_task_results' in row and isinstance(row['per_task_results'], dict):
                            print(f"Per-task prediction details:")
                            logger.info(f"Per-task prediction details:")
                            for task_name, task_data in row['per_task_results'].items():
                                task_orig = task_data.get('original', np.nan)
                                task_int = task_data.get('intervened', np.nan)
                                task_diff = task_data.get('difference', np.nan)
                                print(f"{task_name}: {task_orig:.4f} -> {task_int:.4f} (change={task_diff:.4f})")
                                logger.info(f"{task_name}: {task_orig:.4f} -> {task_int:.4f} (change={task_diff:.4f})")
                    count += 1
                    if count >= 10:  # Limit to first 10 samples
                        break
                if len(strategy_samples) > 5:
                    print(f"... and {len(strategy_samples) - 10} more samples")
                    logger.info(f"... and {len(strategy_samples) - 10} more samples")

            # Show per-task statistics (if available)
            if per_task_strategy_stats:
                print("Per-task Statistics:")
                logger.info("Per-task Statistics:")
                for strategy in strategy_display_order:
                    if strategy in per_task_strategy_stats:
                        display_name = strategy_display_names.get(strategy, strategy)
                        print(f"\n{display_name} Per-task Statistics:")
                        logger.info(f"{display_name} Per-task Statistics:")
                        for task_name, stats in per_task_strategy_stats[strategy].items():
                            logger.info(f" {task_name}: mean_change={stats['mean']:.4f}, std={stats['std']:.4f}, "
                                       f"count={stats['count']}, max_abs_change={stats['max_abs']:.4f}")

            # overall stats and significant changes (threshold configurable)
            if args.task_type == 'class':
                threshold = getattr(args, 'intervention_threshold', 0.1)
                significant_changes = df[df['probability_difference'].abs() > threshold]
                threshold_desc = f"abs_prob_change>{threshold}"
            else:
                threshold = getattr(args, 'intervention_threshold', 0.1)
                significant_changes = df[df['abs_prediction_difference'] > threshold]
                threshold_desc = f"abs_change>{threshold}"

            if len(significant_changes) > 0:
                print(f"\nSignificant change samples ({threshold_desc}): {len(significant_changes)} samples")
                logger.info(f"Significant change samples ({threshold_desc}): {len(significant_changes)} samples")
                for strategy in strategy_display_order:
                    strat_sig = significant_changes[significant_changes['intervention_strategy'] == strategy]
                    if len(strat_sig) == 0:
                        continue
                    display_name = strategy_display_names.get(strategy, strategy)
                    strategy_header = f"\n{display_name} Significant Changes:"
                    print(strategy_header)
                    logger.info(strategy_header)
                    # sort by absolute change
                    sort_col = 'abs_prediction_difference' if abs_diff_col in df.columns else diff_col
                    sorted_samples = strat_sig.sort_values(by=sort_col, ascending=False)
                    for _, row in sorted_samples.iterrows():
                        mol_idx = int(row.get('original_molecule_idx', -1))
                        orig_smiles = row.get('original_smiles', 'N/A')
                        interv_smiles = row.get('intervened_smiles', 'N/A')

                        # Log SMILES information
                        smiles_info = f"Original molecule index {mol_idx}: {orig_smiles}"
                        print(smiles_info)
                        logger.info(smiles_info)

                        intervened_info = f"Intervened molecule: {interv_smiles}"
                        print(intervened_info)
                        logger.info(intervened_info)

                        if args.task_type == 'class':
                            sample_info = f"Prediction change: mean prob {row.get('prediction_difference', 0.0):7.4f} " \
                                         f"({row.get('original_probability', np.nan):.4f} -> {row.get('intervened_probability', np.nan):.4f})"
                            print(sample_info)
                            logger.info(sample_info)
                            # Show subtask with maximum change
                            if 'per_task_results' in row and isinstance(row['per_task_results'], dict):
                                max_change_task = max(row['per_task_results'].items(),
                                                    key=lambda x: abs(x[1].get('probability_difference', 0)))
                                task_name, task_data = max_change_task
                                task_info = f"Max change subtask {task_name}: {task_data.get('probability_difference', 0):.4f} " \
                                           f"({task_data.get('original_probability', np.nan):.4f} -> {task_data.get('intervened_probability', np.nan):.4f})"
                                print(task_info)
                                logger.info(task_info)
                        else:
                            sample_info = f"Prediction change: mean value {row.get('prediction_difference', 0.0):7.4f} " \
                                         f"({row.get('original_prediction', np.nan):.4f} -> {row.get('intervened_prediction', np.nan):.4f})"
                            print(sample_info)
                            logger.info(sample_info)
                            # Show subtask with maximum change
                            if 'per_task_results' in row and isinstance(row['per_task_results'], dict):
                                max_change_task = max(row['per_task_results'].items(),
                                                    key=lambda x: abs(x[1].get('difference', 0)))
                                task_name, task_data = max_change_task
                                task_info = f"Max change subtask {task_name}: {task_data.get('difference', 0):.4f} " \
                                           f"({task_data.get('original', np.nan):.4f} -> {task_data.get('intervened', np.nan):.4f})"
                                print(task_info)
                                logger.info(task_info)
                if len(significant_changes) > 10:
                    truncation_info = f"... and {len(significant_changes) - 10} more significant change samples"
                    print(truncation_info)
                    logger.info(truncation_info)
            else:
                print("Causal analysis inference produced no valid results")
                logger.warning("Causal analysis inference produced no valid results")

    # Close logger handlers
    for handler in list(logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)


if __name__ == "__main__":
    arg = get_args()
    main(arg)
