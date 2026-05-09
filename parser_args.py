# -*- coding: utf-8 -*-
import argparse
from preprocess.features_generators import get_available_features_generators
# from preprocess.featurization_2D import ATOM_FDIM, BOND_FDIM

def get_args():
    parser = argparse.ArgumentParser(description='pytorch version of SGG')

    ''' Sequence Settings '''
    parser.add_argument('--sequence', type=bool, default=True)
    parser.add_argument('--seq_hidden_dim', type=int, default=256)
    parser.add_argument('--max_seq_length', type=int, default=96, help='Maximum sequence length for BERT')
    parser.add_argument('--bert_model_path', type=str, default=r'data/bert_pretrained/RoBERTa',
                        help='Path to BERT model')
    parser.add_argument('--bert_num_heads', type=int, default=12, help='Number of attention heads in BERT model')
    parser.add_argument('--bert_hidden_dim', type=int, default=768)

    ''' Graph Settings '''
    parser.add_argument('--graph', type=bool, default=True)
    parser.add_argument('--gnn_atom_dim', type=int, default=148)
    parser.add_argument('--gnn_bond_dim', type=int, default=17)
    parser.add_argument('--gnn_hidden_dim', type=int, default=256)
    parser.add_argument('--gnn_activation', type=str, default='ReLU', help='ReLU,PReLU，tanh，SELU，ELU')
    parser.add_argument('--gnn_num_layers', type=int, default=6)

    ''' Geometric Setting '''
    parser.add_argument('--geometry', type=bool, default=True)
    parser.add_argument('--geo_hidden_dim', type=int, default=256)
    parser.add_argument('--geo_dropout_rate', type=float, default=0.3)
    parser.add_argument('--geo_layer_num', type=int, default=8)
    parser.add_argument('--geo_readout', type=str, default='mean', choices=['mean', 'sum', 'max'])
    parser.add_argument('--atom_names', type=str, nargs='+', default=['atomic_num', 'formal_charge', 'degree',
                                                                      'chiral_tag', 'total_numHs', 'is_aromatic',
                                                                      'hybridization'])
    parser.add_argument('--bond_names', type=str, nargs='+', default=['bond_dir', 'bond_type', 'is_in_ring'])
    parser.add_argument('--bond_float_names', type=str, nargs='+', default=['bond_length'])
    parser.add_argument('--bond_angle_float_names', type=str, nargs='+', default=['bond_angle'])
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--compound_encoder_config", type=str, default="pre3d_configs/geognn_l8.json")
    parser.add_argument('--process_3d', action='store_true', help='Whether to reprocess 3D data')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of processes for data processing')

    ''' Image Setting '''
    parser.add_argument('--image', type=bool, default=True)
    parser.add_argument('--img_hidden_dim', type=int, default=256)
    parser.add_argument('--img_dropout', type=float, default=0.1)
    parser.add_argument('--img_use_channels', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--img_use_layernorm', action='store_true', default=True)
    parser.add_argument('--img_pool_type', type=str, default='adaptive_avg',
                        choices=['adaptive_avg', 'adaptive_max', 'attention'])
    parser.add_argument('--img_attention', type=str, default='both', choices=['none', 'spatial', 'channel', 'both'])

    parser.add_argument('--image_pretrained_path', type=str, default='data/image_pretrained/ImageMol/ImageMol.pth')

    parser.add_argument('--regenerate_images', action='store_true',
                        help='Force regenerate all molecular images')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size of generated molecular images')
    parser.add_argument('--image_quality', type=int, default=95,
                        help='Quality of generated PNG images (1-100)')
    parser.add_argument('--image_batch_size', type=int, default=1000,
                        help='Batch size for image generation to avoid memory issues')

    ''' Classifier '''
    parser.add_argument('--fusion', type=str, default='weight_fusion', choices=['cat', 'weight_fusion', 'weighted_avg'])
    parser.add_argument('--cross_num_heads', type=int, default=8)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=256)

    ''' Training'''
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bias', type=int, default=1)
    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-5, help='Final learning rate')
    parser.add_argument('--cl_loss', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--pro_num', type=int, default=3)
    parser.add_argument('--pool_type', type=str, default='attention')
    parser.add_argument('--cl_loss_num', type=int, default=0)
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--num_runs', type=int, default=5)

    ''' Options '''
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--fixed_seed', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='bace')
    parser.add_argument('--metric', type=str, default='auc',
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy'])
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--vocab_num', type=int, default=0)
    parser.add_argument('--task_type', type=str, default='class', help='class, reg')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'])
    parser.add_argument('--folds_file', type=str, default=None, help='Optional file of fold labels')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.7, 0.1, 0.2],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='Use messages on atoms instead of messages on bonds')
    parser.add_argument('--max_data_size', type=int, help='Maximum number of data points to load')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--val_fold_index', type=int, default=None)
    parser.add_argument('--test_fold_index', type=int, default=None)
    parser.add_argument('--features_generator', type=str, nargs='*', choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')

    # Attention mechanism parameters
    parser.add_argument('--use_local_attention', action='store_true', default=True,
                        help='Whether to use local attention')
    parser.add_argument('--use_global_attention', action='store_true', default=True,
                        help='Whether to use global attention')

    # Causal intervention parameters
    parser.add_argument('--do_causal_intervention', action='store_true', default=True,
                        help='Whether to perform causal intervention analysis after training')
    parser.add_argument('--intervention_strategies', nargs='+',
                        help='List of intervention strategies, defaults to all strategies')
    parser.add_argument('--intervention_threshold', type=float, default=0.1,
                        help='Intervention significance threshold')
    parser.add_argument('--intervention_max_samples', type=int,
                        help='Maximum number of samples to process (for testing)')

    args = parser.parse_args()
    return args
