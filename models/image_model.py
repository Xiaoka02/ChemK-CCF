# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


class MolecularImageModel(nn.Module):
    def __init__(self, channels, hidden_dim, latent_dim, dropout, use_layernorm, pool_type, attention_type, device):
        super(MolecularImageModel, self).__init__()

        self.channels = channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.pool_type = pool_type
        self.attention_type = attention_type
        self.device = device

        # Store attention weights
        self.functional_group_weights = None
        self.molecular_weights = None

        # Convolutional layers
        self.conv_layers = []
        in_channels = 3
        for out_channels in self.channels:
            self.conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1).to(device),
                nn.BatchNorm2d(out_channels).to(device),
                nn.ReLU().to(device),
                nn.MaxPool2d(2).to(device),
                nn.Dropout2d(self.dropout).to(device)
            ])
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*self.conv_layers)

        # Functional groups attention
        self.functional_group_attention = FunctionalGroupAttention(
            self.channels[-1]
        ).to(device)

        # Molecularly characterized attention
        if self.attention_type != 'none':
            self.attention = MolecularAttention(
                self.channels[-1],
                attention_type=self.attention_type
            ).to(device)
        else:
            self.attention = nn.Identity().to(device)

        # Pooling
        if self.pool_type == 'adaptive_avg':
            self.pooling = nn.AdaptiveAvgPool2d(1).to(device)
        elif self.pool_type == 'adaptive_max':
            self.pooling = nn.AdaptiveMaxPool2d(1).to(device)
        elif self.pool_type == 'attention':
            self.pooling = AttentionPooling(self.channels[-1]).to(device)

        # Feature mapping
        self.fc1 = nn.Linear(self.channels[-1], self.hidden_dim).to(device)
        self.dropout1 = nn.Dropout(self.dropout).to(device)
        if self.use_layernorm:
            self.ln1 = nn.LayerNorm(self.hidden_dim).to(device)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(device)
        self.dropout2 = nn.Dropout(self.dropout).to(device)

        self.fc3 = nn.Linear(self.hidden_dim, self.latent_dim).to(device)
        if self.use_layernorm:
            self.ln2 = nn.LayerNorm(self.latent_dim).to(device)

    def forward(self, x):
        # Convolutional feature extraction
        features = self.conv_layers(x)

        # Process functional group attention and molecular attention in parallel
        # Functional group attention branch
        fg_features, functional_groups = self.functional_group_attention(features)
        if hasattr(self.functional_group_attention, 'attention_maps'):
            self.functional_group_weights = self.functional_group_attention.attention_maps

        # Molecular attention branch
        mol_features = self.attention(features)
        if isinstance(self.attention, MolecularAttention):
            self.molecular_weights = self.attention.get_attention_weights()

        # Merge features from two branches
        features = fg_features + mol_features

        # Pooling
        pooled = self.pooling(features)

        # Fully connected layers
        x = self.fc1(pooled.flatten(1))
        x = F.relu(x)
        x = self.dropout1(x)
        if self.use_layernorm:
            x = self.ln1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        if self.use_layernorm:
            x = self.ln2(x)

        return x, functional_groups


class FunctionalGroupAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Chemical bonding parameters (in Å)
        self.bond_lengths = {
            'C-H': 1.09,
            'C-C': 1.54,
            'C=C': 1.34,
            'C≡C': 1.20,
            'C-O': 1.43,
            'C=O': 1.23,
            'O-H': 0.96,
            'N-H': 1.01,
            'C-N': 1.47,
            'C=N': 1.28,
            'C≡N': 1.16,
            'C-S': 1.82,
            'S-H': 1.34,
            'P-O': 1.63,
            'P=O': 1.50
        }

        # Key angle parameters (in degrees)
        self.bond_angles = {
            'C-C-C': 109.5,
            'C=C-C': 120,
            'C-O-H': 109,
            'O=C-O': 123,
            'C-N-H': 109,
            'O=C-N': 120,
            'C-S-H': 96,
            'O=S=O': 119,
            'O=P-O': 117
        }

        # Electronegativity values
        self.electronegativity = {
            'H': 2.20,
            'C': 2.55,
            'N': 3.04,
            'O': 3.44,
            'F': 3.98,
            'Cl': 3.16,
            'Br': 2.96,
            'I': 2.66,
            'S': 2.58,
            'P': 2.19
        }

        # Define functional group templates and their chemical properties
        self.functional_group_patterns = {
            'hydroxyl': {  # -OH
                'composition': ['C', 'O', 'H'],
                'bonds': ['C-O', 'O-H'],
                'angles': ['C-O-H']
            },
            'carboxyl': {  # -COOH
                'composition': ['C', 'O', 'O', 'H'],
                'bonds': ['C=O', 'C-O', 'O-H'],
                'angles': ['O=C-O']
            },
            'carbonyl': {  # C=O
                'composition': ['C', 'O'],
                'bonds': ['C=O'],
                'angles': []
            },
            'amino': {  # -NH2
                'composition': ['C', 'N', 'H', 'H'],
                'bonds': ['C-N', 'N-H', 'N-H'],
                'angles': ['C-N-H']
            },
            'amide': {  # -CONH2
                'composition': ['C', 'O', 'N', 'H', 'H'],
                'bonds': ['C=O', 'C-N', 'N-H', 'N-H'],
                'angles': ['O=C-N']
            },
            'ether': {  # R-O-R
                'composition': ['C', 'O', 'C'],
                'bonds': ['C-O', 'C-O'],
                'angles': ['C-O-C']
            },
            'ester': {  # R-COO-R
                'composition': ['C', 'O', 'O', 'C'],
                'bonds': ['C=O', 'C-O'],
                'angles': ['O=C-O']
            },
            'alkene': {  # C=C
                'composition': ['C', 'C'],
                'bonds': ['C=C'],
                'angles': []
            },
            'alkyne': {  # C≡C
                'composition': ['C', 'C'],
                'bonds': ['C≡C'],
                'angles': []
            },
            'thiol': {  # -SH
                'composition': ['C', 'S', 'H'],
                'bonds': ['C-S', 'S-H'],
                'angles': ['C-S-H']
            },
            'sulfoxide': {  # R-SO-R
                'composition': ['C', 'S', 'O', 'C'],
                'bonds': ['C-S', 'S=O', 'C-S'],
                'angles': ['O=S-C']
            },
            'sulfone': {  # R-SO2-R
                'composition': ['C', 'S', 'O', 'O', 'C'],
                'bonds': ['C-S', 'S=O', 'S=O', 'C-S'],
                'angles': ['O=S=O']
            },
            'phosphate': {  # -PO4
                'composition': ['P', 'O', 'O', 'O', 'O'],
                'bonds': ['P=O', 'P-O', 'P-O', 'P-O'],
                'angles': ['O=P-O']
            },
            'nitrile': {  # -C≡N
                'composition': ['C', 'N'],
                'bonds': ['C≡N'],
                'angles': []
            },
            'nitro': {  # -NO2
                'composition': ['N', 'O', 'O'],
                'bonds': ['N=O', 'N=O'],
                'angles': ['O=N=O']
            }
        }

        # Feature extraction network - using multi-scale features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # Functional group-specific attention heads
        self.functional_heads = nn.ModuleDict()
        for group_name in self.functional_group_patterns.keys():
            self.functional_heads[group_name] = nn.Sequential(
                nn.Conv2d(in_channels * 3, in_channels // 2, 1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, 1, 1)
            )

        # Attention fusion
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(len(self.functional_group_patterns), 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # Initialize functional group templates
        self.initialize_group_templates()

    def initialize_group_templates(self):
        """Initialize feature templates for each functional group"""
        self.group_templates = {}
        for group_name, properties in self.functional_group_patterns.items():
            # 生成特征模板
            template = self.generate_group_template(
                bonds=properties['bonds'],
                composition=properties['composition'],
                angles=properties['angles']
            )
            self.group_templates[group_name] = template

    def generate_group_template(self, bonds, composition, angles):
        """Generate feature template for each functional group"""
        # Create template matching input feature channels
        template = torch.zeros(1, 1, 7, 7)

        # Set center atom - enhance center feature value
        center = 3  # Center position of 7x7 template
        center_value = 2.0 * sum(self.electronegativity.get(atom, 0) for atom in composition) / len(composition)
        template[0, 0, center, center] = center_value

        # Use Gaussian function to enhance bond features
        def gaussian(x, mu=0, sigma=1):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        # Add bond information - using Gaussian distribution
        for bond in bonds:
            if bond in self.bond_lengths:
                # Use bond length as feature, apply Gaussian distribution
                bond_val = 1.0 - (self.bond_lengths[bond] - 0.9) / 1.0
                sigma = 0.5  # Control Gaussian distribution width

                if '=' in bond:  # Double bond
                    for i in range(-1, 2):
                        template[0, 0, center + i, center] += bond_val * gaussian(torch.tensor(float(i)), sigma=sigma)
                elif '≡' in bond:  # Triple bond
                    for i in range(-1, 2):
                        for j in range(2):
                            template[0, 0, center + i, center + j] += bond_val * gaussian(torch.tensor(float(i)),
                                                                                          sigma=sigma)
                else:  # Single bond
                    for j in range(2):
                        template[0, 0, center, center + j] += bond_val * gaussian(torch.tensor(float(j)), sigma=sigma)

        # Add bond angle information - using more precise angle representation
        if angles:
            for angle in angles:
                if angle in self.bond_angles:
                    # Use bond angle as feature
                    angle_val = (self.bond_angles[angle] / 180.0) * 1.5  # Enhance angle feature
                    # Apply angle feature in 3x3 region
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            dist = np.sqrt(i * i + j * j)
                            if dist <= 1.5:  # Only add features in reasonable range
                                template[0, 0, center + i, center + j] += angle_val * gaussian(torch.tensor(dist),
                                                                                               sigma=0.8)

        # Apply Gaussian filter to enhance spatial correlation
        template = gaussian_filter(template.numpy(), sigma=0.5)
        template = torch.from_numpy(template)

        # Enhance contrast
        template = torch.pow(template, 2.0)

        # Normalize template
        if template.max() > 0:
            template = template / template.max()

        return template

    def forward(self, x):
        # Multi-scale feature extraction
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)

        # Merge multi-scale features
        multi_scale_features = torch.cat([feat1, feat2, feat3], dim=1)

        # Calculate attention for each functional group
        group_attentions = []
        for group_name, head in self.functional_heads.items():
            # Get template for current functional group
            template = self.group_templates[group_name].to(x.device)

            # Compute similarity between features and template
            attention = head(multi_scale_features)

            # Apply template attention
            template_attention = F.conv2d(
                x.mean(dim=1, keepdim=True),
                template.to(x.dtype),
                padding=template.size(-1) // 2
            )

            # Combine feature attention and template attention
            combined_attention = attention * F.sigmoid(template_attention)

            # Apply adaptive threshold, filter weak attention
            threshold_value = float(combined_attention.mean() + 0.5 * combined_attention.std())
            combined_attention = torch.where(combined_attention > threshold_value, combined_attention,
                                             torch.zeros_like(combined_attention))

            group_attentions.append(combined_attention)

        # Merge attention from all functional groups
        combined_attention = torch.cat(group_attentions, dim=1)

        # Use attention fusion layer to learn importance of different functional groups
        attention = self.attention_fusion(combined_attention)

        # Store attention maps for visualization
        self.attention_maps = attention

        # Apply attention to features
        enhanced_features = x * attention.expand_as(x)

        return enhanced_features, combined_attention


class MolecularAttention(nn.Module):
    def __init__(self, channels, attention_type='both'):
        super().__init__()
        self.attention_type = attention_type
        self.spatial_weights = None
        self.channel_weights = None

        if attention_type in ['spatial', 'both']:
            # Spatial attention network
            self.spatial_attention = nn.Sequential(
                # First conv block - maintain feature map size
                nn.Conv2d(channels, channels // 2, 3, padding='same'),
                nn.BatchNorm2d(channels // 2),
                nn.ReLU(),
                # Second conv block - increase receptive field
                nn.Conv2d(channels // 2, channels // 2, 3, padding='same'),
                nn.BatchNorm2d(channels // 2),
                nn.ReLU(),
                # Third conv block - further processing
                nn.Conv2d(channels // 2, channels // 4, 3, padding='same'),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(),
                # Final 1x1 conv to generate attention map
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            )

        if attention_type in ['channel', 'both']:
            # Channel attention network
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid()
            )

    def get_attention_weights(self):
        """Get attention weights"""
        if self.attention_type == 'both':
            if self.spatial_weights is not None and self.channel_weights is not None:
                # Get spatial attention
                spatial_weights = self.spatial_weights

                # Get channel attention
                channel_weights = self.channel_weights
                channel_mean = channel_weights.mean(dim=1, keepdim=True)

                # Use multiplication to fuse spatial and channel attention for better molecular structure localization
                combined_weights = spatial_weights * channel_mean

                # Normalize to [0,1] range
                combined_weights = (combined_weights - combined_weights.min()) / (
                            combined_weights.max() - combined_weights.min() + 1e-8)

                return combined_weights

        elif self.attention_type == 'spatial':
            weights = self.spatial_weights
            return (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

        elif self.attention_type == 'channel':
            weights = self.channel_weights.mean(dim=1, keepdim=True)
            return (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

        return None

    def forward(self, x):
        # Save original input size
        orig_size = x.shape[-2:]

        if self.attention_type in ['spatial', 'both']:
            # Compute spatial attention
            self.spatial_weights = self.spatial_attention(x)

            # Ensure attention map aligns with input features
            if self.spatial_weights.shape[-2:] != orig_size:
                self.spatial_weights = F.interpolate(
                    self.spatial_weights,
                    size=orig_size,
                    mode='bilinear',
                    align_corners=True
                )
            x = x * self.spatial_weights

        if self.attention_type in ['channel', 'both']:
            # Compute channel attention
            self.channel_weights = self.channel_attention(x)
            x = x * self.channel_weights.expand_as(x)

        return x


class AttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        return (x * weights).sum(dim=(2, 3), keepdim=True)


