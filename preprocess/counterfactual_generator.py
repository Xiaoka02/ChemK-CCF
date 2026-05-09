#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counterfactual Sample Generator based on STONED algorithm and CHiMoGNN strategy
Complete counterfactual generation system
"""

import os
import pandas as pd
import selfies
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem import rdMolDescriptors
import random
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class STONEDGenerator:
    """STONED algorithm molecule generator"""
    def __init__(self, num_samples=1000, max_mutations=2):
        self.num_samples = num_samples
        self.max_mutations = max_mutations
        # Basic atom types ensuring chemical validity
        self.basic_alphabet = ['B', 'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I']
    
    def generate_counterfactuals(self, original_smiles: str) -> List[str]:
        """Generate counterfactual samples using STONED algorithm"""
        try:
            # 1. Convert to SELFIES
            selfies_string = selfies.encoder(original_smiles)
            if not selfies_string:
                return []
            
            # 2. STONED generation
            cf_selfies_list = self._stoned_generation(selfies_string)
            
            # 3. Convert back to SMILES and validate
            cf_smiles_list = []
            for cf_selfies in cf_selfies_list:
                try:
                    cf_smiles = selfies.decoder(cf_selfies)
                    if cf_smiles and cf_smiles != original_smiles and self._is_valid_molecule(cf_smiles):
                        cf_smiles_list.append(cf_smiles)
                except:
                    continue
            
            return cf_smiles_list
            
        except Exception as e:
            return []
    
    def _stoned_generation(self, selfies_string: str) -> List[str]:
        """Core generation logic of STONED algorithm"""
        generated_selfies = []
        
        for _ in range(self.num_samples):
            try:
                # Randomly select operation type
                operation = random.choice(['insert', 'delete', 'replace'])
                
                if operation == 'insert':
                    modified_selfies = self._insert_token(selfies_string)
                elif operation == 'delete':
                    modified_selfies = self._delete_token(selfies_string)
                else:
                    modified_selfies = self._replace_token(selfies_string)
                
                if modified_selfies and modified_selfies != selfies_string:
                    generated_selfies.append(modified_selfies)
            except:
                continue
        
        return generated_selfies
    
    def _insert_token(self, selfies_string: str) -> str:
        """Insert token"""
        try:
            tokens = selfies_string.split('[')
            if len(tokens) < 2:
                return selfies_string
            
            # Randomly select insertion position
            insert_pos = random.randint(1, len(tokens) - 1)
            # Randomly select token to insert
            new_token = random.choice(self.basic_alphabet)
            # Insert token
            tokens.insert(insert_pos, f'[{new_token}]')
            return ''.join(tokens)
        except:
            return selfies_string
    
    def _delete_token(self, selfies_string: str) -> str:
        """Delete token"""
        try:
            tokens = selfies_string.split('[')
            if len(tokens) <= 2:  # Keep at least one token
                return selfies_string
            
            # Randomly select deletion position (cannot delete the first one)
            delete_pos = random.randint(1, len(tokens) - 1)
            tokens.pop(delete_pos)
            return ''.join(tokens)
        except:
            return selfies_string
    
    def _replace_token(self, selfies_string: str) -> str:
        """Replace token"""
        try:
            tokens = selfies_string.split('[')
            if len(tokens) < 2:
                return selfies_string
            
            # Randomly select replacement position
            replace_pos = random.randint(1, len(tokens) - 1)
            
            # Randomly select new token
            new_token = random.choice(self.basic_alphabet)
            
            # Replace token
            tokens[replace_pos] = f'[{new_token}]'
            return ''.join(tokens)
        except:
            return selfies_string
    
    def _is_valid_molecule(self, smiles: str) -> bool:
        """Validate if molecule is chemically valid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Basic chemical validity check
            Chem.SanitizeMol(mol)
            
            # Drug-like validity check
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            
            # Basic validity range
            if mw < 50 or mw > 900:
                return False
            if logp < -3 or logp > 7:
                return False
            if hbd > 10 or hba > 15:
                return False
            
            return True
        except:
            return False


class CHiMoGNNLabelStrategy:
    """CHiMoGNN label determination strategy"""
    
    def __init__(self):
        self.similarity_threshold = 0.5
        self.label_change_threshold = 0.8
    
    def determine_labels(self, original_smiles: str, original_label: float,
                         counterfactuals: List[str]) -> List[float]:
        """Determine labels for counterfactual samples based on CHiMoGNN strategy"""
        cf_labels = []
        
        for cf_smiles in counterfactuals:
            try:
                # 1. Analyze structural changes
                structural_changes = self._analyze_structural_changes(original_smiles, cf_smiles)
                # 2. Determine if mainly affects spurious part
                if self._mainly_affects_spurious_part(structural_changes):
                    # Keep original label
                    cf_labels.append(original_label)
                else:
                    # Use descriptor-based prediction
                    cf_label = self._predict_label_from_descriptors(original_smiles, cf_smiles, original_label)
                    cf_labels.append(cf_label)
                    
            except Exception as e:
                # Keep original label on error
                cf_labels.append(original_label)
        return cf_labels
    
    def _analyze_structural_changes(self, original_smiles: str, cf_smiles: str) -> Dict[str, float]:
        """Analyze structural changes"""
        try:
            # Calculate molecular descriptors
            original_desc = self._calculate_descriptors(original_smiles)
            cf_desc = self._calculate_descriptors(cf_smiles)
            
            if not original_desc or not cf_desc:
                return {}
            # Calculate changes
            changes = {}
            for key in original_desc:
                if key in cf_desc:
                    changes[f"{key}_change"] = cf_desc[key] - original_desc[key]
            return changes
        except:
            return {}
    
    def _mainly_affects_spurious_part(self, structural_changes: Dict[str, float]) -> bool:
        """Determine if mainly affects spurious part"""
        if not structural_changes:
            return True
        # Spurious feature indicators
        spurious_indicators = [
            'molecular_weight_change',
            'logp_change',
            'tpsa_change',
            'hbd_change',
            'hba_change'
        ]
        
        # Calculate proportion of spurious feature changes
        spurious_changes = sum(1 for change in structural_changes.keys()
                               if any(indicator in change for indicator in spurious_indicators))
        total_changes = len(structural_changes)
        
        if total_changes == 0:
            return True
        
        return spurious_changes / total_changes > 0.6
    
    def _predict_label_from_descriptors(self, original_smiles: str, cf_smiles: str,
                                        original_label: float) -> float:
        """Predict label based on descriptor changes"""
        try:
            # Calculate descriptor changes
            original_desc = self._calculate_descriptors(original_smiles)
            cf_desc = self._calculate_descriptors(cf_smiles)
            if not original_desc or not cf_desc:
                return original_label
            
            # Predict label change based on descriptor changes
            label_change = 0.0
            
            # Molecular weight change effect
            mw_change = cf_desc.get('molecular_weight', 0) - original_desc.get('molecular_weight', 0)
            if abs(mw_change) > 50:
                label_change += 0.1 if mw_change > 0 else -0.1
            
            # Lipophilicity change effect
            logp_change = cf_desc.get('logp', 0) - original_desc.get('logp', 0)
            if abs(logp_change) > 1.0:
                label_change += 0.2 if logp_change > 0 else -0.2
            
            # Polar surface area change effect
            tpsa_change = cf_desc.get('tpsa', 0) - original_desc.get('tpsa', 0)
            if abs(tpsa_change) > 20:
                label_change += 0.1 if tpsa_change > 0 else -0.1
            
            # Hydrogen bond change effect
            hbd_change = cf_desc.get('hbd', 0) - original_desc.get('hbd', 0)
            hba_change = cf_desc.get('hba', 0) - original_desc.get('hba', 0)
            
            if abs(hbd_change) > 2:
                label_change += 0.15 if hbd_change > 0 else -0.15
            if abs(hba_change) > 2:
                label_change += 0.1 if hba_change > 0 else -0.1
            
            # Calculate new label
            new_label = original_label + label_change
            
            # Ensure label is within reasonable range
            if isinstance(original_label, int):  # Classification task
                new_label = 1 if new_label > 0.5 else 0
            else:
                return new_label
        except:
            return original_label
    
    def _calculate_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular descriptors"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            return {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hbd': Lipinski.NumHDonors(mol),
                'hba': Lipinski.NumHAcceptors(mol),
                'qed': QED.qed(mol)
            }
        except:
            return {}


class QualityFilter:
    """Quality filter"""
    
    def __init__(self):
        self.similarity_threshold = 0.3  # Lower similarity threshold
        self.label_change_threshold = 0.8
    
    def filter_chemically_valid(self, original_smiles: str, counterfactuals: List[str]) -> List[str]:
        """Filter chemically valid counterfactual samples"""
        valid_cfs = []
        
        for cf in counterfactuals:
            if self._is_chemically_valid(cf):
                similarity = self._calculate_similarity(original_smiles, cf)
                if similarity > self.similarity_threshold:
                    valid_cfs.append(cf)
        
        return valid_cfs
    
    def filter_high_quality(self, original_smiles: str, original_label: float,
                            counterfactuals: List[str], labels: List[float]) -> Tuple[List[str], List[float]]:
        """Filter high quality counterfactual samples"""
        high_quality_cfs = []
        high_quality_labels = []
        
        for cf, label in zip(counterfactuals, labels):
            label_change = abs(label - original_label)
            if label_change <= self.label_change_threshold:
                if self._provides_meaningful_explanation(original_smiles, cf, label_change):
                    high_quality_cfs.append(cf)
                    high_quality_labels.append(label)
        
        return high_quality_cfs, high_quality_labels
    
    def _is_chemically_valid(self, smiles: str) -> bool:
        """Check chemical validity"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            Chem.SanitizeMol(mol)
            return True
        except:
            return False
    
    def _calculate_similarity(self, smiles1: str, smiles2: str) -> float:
        """Calculate molecular similarity"""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                return 0.0
            
            # Use Tanimoto similarity
            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2)
            
            from rdkit import DataStructs
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            return similarity
        except:
            return 0.0
    
    def _provides_meaningful_explanation(self, original_smiles: str, cf_smiles: str,
                                         label_change: float) -> bool:
        """Check if provides meaningful explanation"""
        try:
            # Analyze structural changes
            original_mol = Chem.MolFromSmiles(original_smiles)
            cf_mol = Chem.MolFromSmiles(cf_smiles)
            
            if original_mol is None or cf_mol is None:
                return False
            
            # Calculate atom count change
            atom_change = cf_mol.GetNumAtoms() - original_mol.GetNumAtoms()
            
            # Check for reasonable structural changes
            if abs(atom_change) > 10:  # Relax atom count change limit
                return False
            
            # Check if label change is reasonable
            if abs(label_change) > 0.8:  # Relax label change limit
                return False
            
            return True
        except:
            return False


class CounterfactualGenerator:
    """Integrated counterfactual generator"""
    
    def __init__(self, num_samples=1000, max_mutations=2):
        self.stoned_generator = STONEDGenerator(num_samples, max_mutations)
        self.label_strategy = CHiMoGNNLabelStrategy()
        self.quality_filter = QualityFilter()
    
    def generate_complete_counterfactuals(self, original_smiles: str, original_label: float) -> Tuple[List[str], List[float]]:
        """Generate complete counterfactual samples (with labels)"""
        try:
            # 1. STONED generate counterfactual samples
            raw_counterfactuals = self.stoned_generator.generate_counterfactuals(original_smiles)
            
            if not raw_counterfactuals:
                return [], []
            
            # 2. Initial quality filtering
            valid_counterfactuals = self.quality_filter.filter_chemically_valid(
                original_smiles, raw_counterfactuals
            )
            
            if not valid_counterfactuals:
                return [], []
            
            # 3. Determine labels
            cf_labels = self.label_strategy.determine_labels(
                original_smiles, original_label, valid_counterfactuals
            )
            
            # 4. Final quality filtering
            final_cfs, final_labels = self.quality_filter.filter_high_quality(
                original_smiles, original_label, valid_counterfactuals, cf_labels
            )
            
            return final_cfs, final_labels
            
        except Exception as e:
            return [], []


class DatasetProcessor:
    """Dataset counterfactual processor"""
    
    def __init__(self, data_dir="data", num_samples=1000, max_mutations=2,
                 num_counterfactuals_per_sample=3):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.max_mutations = max_mutations
        self.num_counterfactuals_per_sample = num_counterfactuals_per_sample
        self.generator = CounterfactualGenerator(num_samples, max_mutations)
        
        # Supported datasets
        self.datasets = [
            'bace', 'bbbp', 'clintox', 'esol', 'freesolv',
            'hiv', 'Lipophilicity', 'sider', 'tox21', 'toxcast'
        ]
    
    def process_dataset(self, dataset_name: str, max_samples: Optional[int] = None) -> bool:
        """Process single dataset"""
        try:
            print(f"\n{'=' * 50}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'=' * 50}")
            
            # 1. Load dataset
            df = self._load_dataset(dataset_name)
            if df is None:
                print(f"Failed to load dataset {dataset_name}")
                return False
            
            # Limit sample count (for testing)
            if max_samples:
                df = df.head(max_samples)
            
            print(f"Loaded {len(df)} samples from {dataset_name}")
            
            # 2. Determine SMILES column and label column
            smiles_col = self._find_smiles_column(df)
            label_col = self._find_label_column(df)
            
            if not smiles_col or not label_col:
                print(f"Cannot find SMILES or label column in {dataset_name}")
                return False
            
            print(f"Using SMILES column: {smiles_col}")
            print(f"Using label column: {label_col}")
            
            # 3. Generate counterfactual samples
            counterfactual_data = []
            successful_samples = 0
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
                original_smiles = row[smiles_col]
                original_label = row[label_col]
                
                # Skip invalid SMILES
                if pd.isna(original_smiles) or not isinstance(original_smiles, str):
                    continue
                
                # Generate counterfactual samples
                counterfactuals, labels = self.generator.generate_complete_counterfactuals(
                    original_smiles, original_label
                )
                
                if counterfactuals:  # If counterfactuals were generated
                    successful_samples += 1
                    
                    # Limit number of counterfactuals per sample
                    if len(counterfactuals) > self.num_counterfactuals_per_sample:
                        # Random selection
                        selected_indices = random.sample(
                            range(len(counterfactuals)),
                            self.num_counterfactuals_per_sample
                        )
                        counterfactuals = [counterfactuals[i] for i in selected_indices]
                        labels = [labels[i] for i in selected_indices]
                    
                    # Save counterfactual data
                    for cf_smiles, cf_label in zip(counterfactuals, labels):
                        new_row = row.copy()
                        new_row[smiles_col] = cf_smiles
                        new_row['original_smiles'] = original_smiles
                        new_row['original_label'] = original_label
                        new_row['counterfactual_label'] = cf_label
                        new_row['label_change'] = cf_label - original_label
                        new_row['modification_type'] = 'STONED_generated'
                        new_row['modification_description'] = 'Generated using STONED algorithm with CHiMoGNN labeling'
                        
                        counterfactual_data.append(new_row)
            
            # 4. Create counterfactual dataset
            if counterfactual_data:
                cf_df = pd.DataFrame(counterfactual_data)
                print(f"Generated {len(cf_df)} counterfactual samples for {dataset_name}")
                print(f"Successfully processed {successful_samples}/{len(df)} original samples")
                
                # 5. Save dataset
                self._save_counterfactual_dataset(dataset_name, cf_df)
                return True
            else:
                print(f"No counterfactuals generated for {dataset_name}")
                return False
                
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            return False
    
    def process_all_datasets(self, max_samples_per_dataset: Optional[int] = None):
        """Process all datasets"""
        print("Starting counterfactual generation for all datasets...")
        print(f"Number of counterfactuals per sample: {self.num_counterfactuals_per_sample}")
        if max_samples_per_dataset:
            print(f"Max samples per dataset: {max_samples_per_dataset}")
        
        results = {}
        
        for dataset_name in self.datasets:
            try:
                success = self.process_dataset(dataset_name, max_samples_per_dataset)
                results[dataset_name] = {
                    'success': success,
                    'error': None if success else 'Processing failed'
                }
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                results[dataset_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Print summary
        print("\n" + "=" * 50)
        print("COUNTERFACTUAL GENERATION SUMMARY")
        print("=" * 50)
        
        for dataset_name, result in results.items():
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            error = f" ({result['error']})" if result['error'] else ""
            print(f"{dataset_name}: {status}{error}")
        
        return results
    
    def _load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load dataset"""
        # Try multiple possible paths
        possible_paths = [
            os.path.join(self.data_dir, dataset_name, f'{dataset_name}.csv'),  # Original path
            os.path.join('..', self.data_dir, dataset_name, f'{dataset_name}.csv'),  # Parent directory
            os.path.join('data', dataset_name, f'{dataset_name}.csv'),  # Direct data path
            os.path.join('../data', dataset_name, f'{dataset_name}.csv'),  # Parent data path
        ]
        
        for data_path in possible_paths:
            if os.path.exists(data_path):
                try:
                    df = pd.read_csv(data_path)
                    print(f"Successfully loaded dataset from: {data_path}")
                    return df
                except Exception as e:
                    print(f"Error loading {data_path}: {e}")
                    continue
        
        # If all paths fail, print all attempted paths
        print(f"Dataset {dataset_name} not found. Tried paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return None
    
    def _find_smiles_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find SMILES column"""
        possible_names = ['smiles', 'SMILES', 'canonical_smiles', 'mol', 'molecule']
        
        for col in df.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                return col
        
        # If not found, return first column
        return df.columns[0] if len(df.columns) > 0 else None
    
    def _find_label_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find label column"""
        possible_names = ['Class', 'class', 'label', 'Label', 'target', 'Target', 'y']
        
        for col in df.columns:
            if col in possible_names:
                return col
        
        # If not found, return last column
        return df.columns[-1] if len(df.columns) > 1 else None
    
    def _save_counterfactual_dataset(self, dataset_name: str, cf_df: pd.DataFrame):
        """Save counterfactual dataset"""
        # Use same path logic as loading dataset
        possible_output_paths = [
            os.path.join(self.data_dir, dataset_name, f'{dataset_name}_cofa.csv'),
            os.path.join('..', self.data_dir, dataset_name, f'{dataset_name}_cofa.csv'),
            os.path.join('data', dataset_name, f'{dataset_name}_cofa.csv'),
            os.path.join('../data', dataset_name, f'{dataset_name}_cofa.csv'),
        ]
        
        for output_path in possible_output_paths:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cf_df.to_csv(output_path, index=False)
                print(f"Saved counterfactual dataset to: {output_path}")
                return
            except Exception as e:
                print(f"Error saving to {output_path}: {e}")
                continue
        
        print(f"Failed to save {dataset_name}_cofa.csv to any location")


def main():
    parser = argparse.ArgumentParser(description='Generate counterfactual datasets using STONED + CHiMoGNN')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory path')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate per molecule')
    parser.add_argument('--max_mutations', type=int, default=2,
                        help='Maximum number of mutations in STONED')
    parser.add_argument('--num_counterfactuals', type=int, default=3,
                        help='Number of counterfactuals per sample')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per dataset (for testing)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to process (default: all)')
    
    args = parser.parse_args()
    
    # Create processor
    processor = DatasetProcessor(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        max_mutations=args.max_mutations,
        num_counterfactuals_per_sample=args.num_counterfactuals
    )
    
    # Set datasets to process
    if args.datasets:
        processor.datasets = args.datasets
    
    # Process datasets
    if len(processor.datasets) == 1:
        # Process single dataset
        success = processor.process_dataset(processor.datasets[0], args.max_samples)
        if success:
            print(f"\n✓ Successfully processed {processor.datasets[0]}")
        else:
            print(f"\n✗ Failed to process {processor.datasets[0]}")
    else:
        # Process all datasets
        results = processor.process_all_datasets(args.max_samples)
        
        # Count results
        success_count = sum(1 for result in results.values() if result['success'])
        total_count = len(results)
        
        print(f"\nFinal Results: {success_count}/{total_count} datasets processed successfully")
        
        if success_count > 0:
            print(f"Counterfactual datasets saved in respective dataset directories with '_cofa' suffix.")


if __name__ == "__main__":
    main()
