#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal Intervention Dataset Generator
Generates counterfactual samples for molecular intervention analysis.

Main Functions:
1. Domain-knowledge-guided causal interventions (SMARTS-based substructure matching)
2. Generate intervened molecular SMILES strings
3. Chemical validity verification
4. Intervention dataset generation and storage
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Lipinski
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
import logging
import argparse
from typing import List, Dict, Tuple, Optional
import warnings
import json

# Suppress common RDKit warnings
warnings.filterwarnings("ignore", message="not removing hydrogen atom without neighbors")
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')

# Set RDKit log level to reduce warning output
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Configure logger (module-level, does not change global root log level)
logger = logging.getLogger(__name__)


class CausalInterventionGenerator:
    """
    Causal Intervention Dataset Generator
    Generates counterfactual samples for molecular intervention analysis.
    """
    def __init__(self, chemical_validity_threshold: float = 0.7):
        """
        Initialize causal intervention generator.
        Args:
            chemical_validity_threshold: Chemical validity threshold (QED)
        """
        # Define domain-knowledge-guided intervention strategies
        self.intervention_strategies = self._define_intervention_strategies()

        # Set chemical validity threshold
        self.chemical_validity_threshold = chemical_validity_threshold

        # Initialization complete
        logger.debug("Causal intervention dataset generator initialized")

    def _define_intervention_strategies(self) -> Dict[str, Dict]:
        """
        Define domain-knowledge-guided intervention strategies
        using SMARTS pattern matching to identify and remove specific chemical groups.
        """
        strategies = {
            # Aromatic chloro group - common in drug molecules, affects activity
            'aromatic_chloro': {
                'name': 'Aromatic Chloro Replacement',
                'smarts_pattern': 'c[Cl]',  # Chlorine attached to aromatic carbon (canonical form)
                'description': 'Replace chlorine on aromatic ring with hydrogen, observe impact on prediction',
                'chemical_impact': 'Chloro groups modulate molecular lipophilicity, replacement with H may alter ADME properties',
                'terminal_only': False  # Halogens are naturally terminal, no additional restriction needed
            },

            # Carboxylic acid group - affects molecular acidity
            'carboxylic_acid': {
                'name': 'Carboxylic Acid Replacement',
                'smarts_pattern': 'C(=O)[O;H,-]',  # Compatible with neutral and ionic carboxylic acids
                'description': 'Replace carboxylic acid group with hydrogen, observe impact on prediction',
                'chemical_impact': 'Carboxylic acid affects molecular acidity and water solubility',
                'terminal_only': True  # Only match terminal carboxylic acids
            },

            # Sulfonamide group - common in drugs
            'sulfonamide': {
                'name': 'Sulfonamide Replacement',
                'smarts_pattern': 'S(=O)(=O)[N;H1,H2]',  # Ensure nitrogen has hydrogen, avoid matching complex structures
                'description': 'Replace sulfonamide group with hydrogen, observe impact on prediction',
                'chemical_impact': 'Sulfonamide affects molecular polarity and hydrogen bonding properties',
                'terminal_only': False  # SMARTS is already specific enough
            },

            # Fluoro aromatic group - affects metabolic stability
            'fluoro_aromatic': {
                'name': 'Fluoro Aromatic Replacement',
                'smarts_pattern': 'c[F]',  # Fluorine attached to aromatic carbon (canonical form)
                'description': 'Replace fluorine on aromatic ring with hydrogen, observe impact on prediction',
                'chemical_impact': 'Fluoro groups affect molecular metabolic stability and lipophilicity',
                'terminal_only': False  # Halogens are naturally terminal, no additional restriction needed
            },

            # Hydroxyl group - affects molecular polarity (only alcohol hydroxyl, avoid matching carboxylic acids)
            'hydroxyl': {
                'name': 'Alcohol Hydroxyl Replacement',
                'smarts_pattern': '[C;X4][OH]',  # Precisely match alcohol hydroxyl on sp3 carbon
                'description': 'Replace alcohol hydroxyl with hydrogen, observe impact on prediction',
                'chemical_impact': 'Hydroxyl affects molecular polarity and hydrogen bonding capacity',
                'terminal_only': False  # SMARTS is already specific enough
            },

            # Amide group - affects molecular conformation (avoid matching peptide bonds)
            'amide': {
                'name': 'Amide Replacement',
                'smarts_pattern': 'C(=O)[N;H1,H2]',  # Ensure nitrogen has hydrogen, avoid matching peptide bonds
                'description': 'Replace amide group with hydrogen, observe impact on prediction',
                'chemical_impact': 'Amide affects molecular conformation and hydrogen bonding properties',
                'terminal_only': False  # SMARTS is already specific enough
            }
        }
        return strategies

    def perform_intervention(self, original_mol: Chem.Mol, strategy_name: str) -> Optional[Chem.Mol]:
        """
        Perform causal intervention on molecule - using safe replacement mechanism to ensure molecular integrity.
        Args:
            original_mol: Original molecule object
            strategy_name: Intervention strategy name

        Returns:
            Intervened molecule object, or None if intervention fails
        """
        if strategy_name not in self.intervention_strategies:
            logger.warning(f"Unknown intervention strategy: {strategy_name}")
            return None

        strategy = self.intervention_strategies[strategy_name]
        smarts_pattern = strategy['smarts_pattern']
        try:
            # Create SMARTS pattern
            pattern = Chem.MolFromSmarts(smarts_pattern)
            if pattern is None:
                logger.warning(f"Invalid SMARTS pattern: {smarts_pattern}")
                return None

            # Search for matching substructures in molecule (implementing minimum edit principle: one site per intervention)
            matches = original_mol.GetSubstructMatches(pattern)

            if not matches:
                logger.debug(f"No matching substructure found in molecule: {smarts_pattern}")
                return None

            # Implement minimum edit principle: intervene only one site at a time
            # Check if strategy requires only terminal groups
            terminal_only = self.intervention_strategies[strategy_name].get('terminal_only', False)

            if terminal_only:
                # Filter out terminal groups (connected atoms <= 1)
                terminal_matches = []
                for match in matches:
                    if self._is_terminal_group(original_mol, match, strategy_name):
                        terminal_matches.append(match)

                if not terminal_matches:
                    logger.debug(f"No eligible terminal group found: {smarts_pattern}")
                    return None

                matches = terminal_matches[:1]  # Only use first terminal match, implementing minimum edit
            else:
                # For non-terminal-restricted strategies, also process only the first match
                matches = matches[:1]

            # Perform intervention on selected match
            for match in matches:
                try:
                    intervened_mol = self._perform_safe_intervention(original_mol, match, strategy_name, smarts_pattern)
                    if intervened_mol is not None:
                        return intervened_mol
                except Exception as e:
                    logger.debug(f"Intervention match {match} failed: {str(e)}")
                    continue

            return None

        except Exception as e:
            logger.error(f"Intervention operation failed: {str(e)}")
            return None

    def _perform_safe_intervention(self, original_mol: Chem.Mol, match: Tuple, strategy_name: str, smarts_pattern: str) -> Optional[Chem.Mol]:
        """
        Perform safe molecular intervention to ensure molecular integrity.
        Args:
            original_mol: Original molecule object
            match: Matched atom index tuple
            strategy_name: Intervention strategy name
            smarts_pattern: SMARTS pattern string

        Returns:
            Intervened molecule object
        """
        try:
            # Create molecule copy
            mol_copy = Chem.Mol(original_mol)

            # Special handling for different strategies
            if strategy_name in ['aromatic_chloro', 'fluoro_aromatic']:
                # Aromatic halogens need special handling
                match_atoms = [atom_idx for atom_idx in match]  # Convert to list
                halogen_idx = match_atoms[1] if len(match_atoms) > 1 else match_atoms[0]
                intervened_mol = self._replace_aromatic_halogen(mol_copy, strategy_name, halogen_idx)
            else:
                # Use unified substructure replacement method
                intervened_mol = self._perform_substructure_replacement(mol_copy, smarts_pattern, strategy_name)

            if intervened_mol is None:
                return None

            # Validate intervened molecule integrity
            if not self._validate_intervention_result(intervened_mol, original_mol):
                return None

            return intervened_mol

        except Exception as e:
            logger.debug(f"Safe intervention failed: {str(e)}")
            return None

    def _perform_substructure_replacement(self, mol: Chem.Mol, smarts_pattern: str, strategy_name: str) -> Optional[Chem.Mol]:
        """
        Perform substructure replacement intervention - using precise replacement strategy.

        Args:
            mol: Molecule object
            smarts_pattern: SMARTS pattern
            strategy_name: Intervention strategy name

        Returns:
            Intervened molecule object
        """
        try:
            # Use ReplaceSubstructs for substructure replacement
            pattern = Chem.MolFromSmarts(smarts_pattern)
            if pattern is None:
                logger.warning(f"Cannot create SMARTS pattern: {smarts_pattern}")
                return None

            replacement = Chem.MolFromSmiles('[H]')
            result_mols = Chem.ReplaceSubstructs(mol, pattern, replacement)

            if not result_mols:
                return None

            result_mol = result_mols[0]
            Chem.SanitizeMol(result_mol)
            return result_mol

        except Exception as e:
            logger.debug(f"Substructure replacement failed: {str(e)}")
            return None

    def _is_terminal_group(self, mol: Chem.Mol, match: Tuple, strategy_name: str) -> bool:
        """
        Determine if matched group is a terminal group (avoid intervening in scaffold structure).
        Dynamically identify by atom type, not relying on fixed indices.

        Args:
            mol: Molecule object
            match: Matched atom index tuple
            strategy_name: Intervention strategy name

        Returns:
            Whether it is a terminal group
        """
        try:
            if strategy_name in ['aromatic_chloro', 'fluoro_aromatic']:
                # Halogens are naturally terminal, SMARTS is already specific
                return True

            elif strategy_name == 'hydroxyl':
                # Precise filtering implemented in _replace_hydroxyl_group, pass through here
                return True

            elif strategy_name == 'carboxylic_acid':
                # Find carbonyl carbon (connected to double-bond oxygen)
                carbonyl_idx = None
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetAtomicNum() == 6:  # Carbon atom
                        # Check if has double-bond oxygen
                        has_double_oxygen = False
                        for neighbor in atom.GetNeighbors():
                            if (neighbor.GetAtomicNum() == 8 and
                                mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE):
                                has_double_oxygen = True
                                break
                        if has_double_oxygen:
                            carbonyl_idx = atom_idx
                            break

                if carbonyl_idx is None:
                    return False

                carbonyl_atom = mol.GetAtomWithIdx(carbonyl_idx)
                # Check carbonyl carbon connectivity (should connect to only one carbon chain)
                carbon_neighbors = [atom for atom in carbonyl_atom.GetNeighbors()
                                  if atom.GetAtomicNum() == 6]

                return len(carbon_neighbors) <= 1  # Terminal carboxylic acid connects to only one carbon chain

            elif strategy_name in ['amide', 'sulfonamide']:
                # Find nitrogen atom
                nitrogen_idx = None
                for atom_idx in match:
                    if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 7:
                        nitrogen_idx = atom_idx
                        break

                if nitrogen_idx is None:
                    return False

                nitrogen_atom = mol.GetAtomWithIdx(nitrogen_idx)
                # Check nitrogen substitution (should have at least one hydrogen, indicating terminal)
                hydrogen_neighbors = [atom for atom in nitrogen_atom.GetNeighbors()
                                    if atom.GetAtomicNum() == 1]

                return len(hydrogen_neighbors) >= 1

            else:
                # Default: all matches are terminal (backward compatibility)
                return True

        except Exception as e:
            logger.debug(f"Terminal group judgment failed: {str(e)}")
            return False

    def _replace_aromatic_halogen(self, mol: Chem.Mol, strategy_name: str, halogen_idx: int = None) -> Optional[Chem.Mol]:
        """
        Special handling for aromatic halogen replacement - directly delete halogen atom,
        RDKit will automatically adjust aromaticity.
        Args:
            mol: Molecule object
            strategy_name: Intervention strategy name
            halogen_idx: Halogen atom index

        Returns:
            Replaced molecule object
        """
        try:
            # Create molecule copy
            mol_copy = Chem.Mol(mol)
            rw_mol = Chem.RWMol(mol_copy)

            # Delete halogen atom
            rw_mol.RemoveAtom(halogen_idx)

            # Convert to regular molecule
            result_mol = rw_mol.GetMol()

            # Recalculate molecular properties and aromaticity
            Chem.SanitizeMol(result_mol)

            return result_mol

        except Exception as e:
            logger.debug(f"Aromatic halogen replacement failed: {str(e)}")
            return None

    def _replace_hydroxyl_group(self, mol: Chem.Mol, smarts_pattern: str) -> Optional[Chem.Mol]:
        """
        Special handling for alcohol hydroxyl replacement - directly delete hydroxyl oxygen and hydrogen atom.

        Args:
            mol: Molecule object
            smarts_pattern: SMARTS pattern

        Returns:
            Replaced molecule object
        """
        try:
            # Find hydroxyl oxygen
            oxygen_idx = None
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 8:  # Oxygen atom
                    # Check if it is hydroxyl (has implicit hydrogen and connects to only one carbon)
                    neighbors = atom.GetNeighbors()
                    implicit_h = atom.GetTotalNumHs()  # Number of implicit hydrogens
                    explicit_h = sum(1 for n in neighbors if n.GetAtomicNum() == 1)  # Explicit hydrogens
                    total_h = implicit_h + explicit_h
                    carbon_count = sum(1 for n in neighbors if n.GetAtomicNum() == 6)

                    # Hydroxyl oxygen should have 1 hydrogen (possibly implicit) and connect to only 1 carbon
                    if total_h == 1 and carbon_count == 1:
                        oxygen_idx = atom.GetIdx()
                        break

            if oxygen_idx is None:
                return None

            # Create molecule copy and delete hydroxyl
            mol_copy = Chem.Mol(mol)
            rw_mol = Chem.RWMol(mol_copy)

            # Find atoms to delete: oxygen and connected hydrogen
            atoms_to_remove = [oxygen_idx]
            oxygen_atom = rw_mol.GetAtomWithIdx(oxygen_idx)
            for neighbor in oxygen_atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:  # Hydrogen atom
                    atoms_to_remove.append(neighbor.GetIdx())

            # Delete atoms in reverse order
            for atom_idx in sorted(atoms_to_remove, reverse=True):
                rw_mol.RemoveAtom(atom_idx)

            # Convert to regular molecule
            result_mol = rw_mol.GetMol()

            # Recalculate molecular properties
            Chem.SanitizeMol(result_mol)

            return result_mol

        except Exception as e:
            logger.debug(f"Hydroxyl replacement failed: {str(e)}")
            return None

    def _validate_intervention_result(self, mol: Chem.Mol, original_mol: Chem.Mol = None) -> bool:
        """
        Validate the reasonableness of intervention results.

        Args:
            mol: Intervened molecule object
            original_mol: Original molecule object (for comparative validation)

        Returns:
            Whether reasonable
        """
        try:
            # Basic validation
            if mol is None or mol.GetNumAtoms() == 0:
                return False

            # Check if single molecule (no '.')
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:
                logger.debug("Intervention result contains multiple fragments, considered invalid")
                return False

            # Check if molecule can be processed normally
            Chem.SanitizeMol(mol)

            # Check valency reasonableness
            try:
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            except:
                pass

            # Check atom valency
            for atom in mol.GetAtoms():
                if not atom.HasProp('_ChiralityPossible'):
                    try:
                        Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum())
                    except:
                        continue

            # Aromaticity validation handled by RDKit's SanitizeMol, no additional checks needed

            # Comparative validation with original molecule
            if original_mol is not None:
                original_atoms = original_mol.GetNumAtoms()
                new_atoms = mol.GetNumAtoms()

                # Atom count change should be reasonable (usually decrease by 1-10 atoms)
                if abs(original_atoms - new_atoms) > 10:
                    logger.debug(f"Atom count change too large: {original_atoms} -> {new_atoms}")
                    return False

                # Molecular weight change should be reasonable
                original_mw = Chem.Descriptors.ExactMolWt(original_mol)
                new_mw = Chem.Descriptors.ExactMolWt(mol)

                # After group replacement, molecular weight should decrease (unless replacing hydrogen)
                if new_mw > original_mw + 50:  # Allow slight increase (e.g., hydrogen replacement)
                    logger.debug(f"Molecular weight change abnormal: {original_mw:.2f} -> {new_mw:.2f}")
                    return False

            # Check basic chemical properties
            validity = self.check_chemical_validity(mol)
            return validity.get('is_valid', False)

        except Exception as e:
            logger.debug(f"Intervention result validation failed: {str(e)}")
            return False

    def generate_intervention_dataset(self, dataset_path: str, output_dir: str,
                                     strategies: List[str] = None, max_samples: int = None,
                                     test_indices: List[int] = None) -> pd.DataFrame:
        """
        Generate causal intervention dataset.

        Args:
            dataset_path: Original dataset path
            output_dir: Output directory
            strategies: List of intervention strategies to use, defaults to all strategies
            max_samples: Maximum number of samples to process, for testing
            test_indices: Test set molecule index list, if provided only intervene test set molecules

        Returns:
            DataFrame containing all intervention samples
        """
        if strategies is None:
            strategies = list(self.intervention_strategies.keys())

        # Read original dataset
        original_df = pd.read_csv(dataset_path)

        # If test set indices provided, only process test set molecules
        if test_indices is not None:
            original_df = original_df.iloc[test_indices].reset_index(drop=True)
            logger.debug(f"Processing only {len(original_df)} molecules in test set")
        else:
            # If max_samples specified, sample
            if max_samples and len(original_df) > max_samples:
                original_df = original_df.sample(n=max_samples, random_state=42)
                logger.info(f"Sampled {max_samples} molecules for processing")

        # Prepare result storage
        intervention_results = []

        logger.debug(f"Starting processing {len(original_df)} molecules")
        logger.debug(f"Using intervention strategies: {strategies}")

        for idx, row in tqdm(original_df.iterrows(), total=len(original_df)):
            smiles = row['smiles']

            try:
                # Parse original molecule
                original_mol = Chem.MolFromSmiles(smiles)
                if original_mol is None:
                    continue

                # Check original molecule chemical properties
                original_validity = self.check_chemical_validity(original_mol)

                for strategy_name in strategies:
                    # Perform intervention
                    intervened_mol = self.perform_intervention(original_mol, strategy_name)

                    if intervened_mol is None:
                        continue

                    # Check intervened molecule chemical validity
                    intervened_validity = self.check_chemical_validity(intervened_mol)

                    if not intervened_validity['is_valid']:
                        continue

                    # Generate intervened SMILES
                    intervened_smiles = Chem.MolToSmiles(intervened_mol, canonical=True)

                    # Record result
                    result = {
                        'original_smiles': smiles,
                        'intervened_smiles': intervened_smiles,
                        'intervention_strategy': strategy_name,
                        # 'strategy_description': self.intervention_strategies[strategy_name]['description'],
                        # 'chemical_impact': self.intervention_strategies[strategy_name]['chemical_impact'],
                        # Chemical properties
                        # 'original_qed': original_validity['qed'],
                        # 'intervened_qed': intervened_validity['qed'],
                        # 'original_mol_weight': original_validity['mol_weight'],
                        # 'intervened_mol_weight': intervened_validity['mol_weight'],
                        'original_logp': original_validity['logp'],
                        'intervened_logp': intervened_validity['logp'],
                        'original_hbd': original_validity['hbd'],
                        'intervened_hbd': intervened_validity['hbd'],
                        'original_hba': original_validity['hba'],
                        'intervened_hba': intervened_validity['hba'],
                        # Validation flag
                        'is_chemically_valid': intervened_validity['is_valid']
                    }

                    intervention_results.append(result)

            except Exception as e:
                logger.error(f"Error processing molecule {smiles}: {str(e)}")
                continue

        # Save results
        results_df = pd.DataFrame(intervention_results)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get dataset name
        dataset_name = os.path.basename(dataset_path).replace('.csv', '')

        # Save intervention dataset
        output_filename = f'Intervention_{dataset_name}.csv'
        output_path = os.path.join(output_dir, output_filename)
        results_df.to_csv(output_path, index=False)

        # Save statistics
        self._save_intervention_statistics(results_df, output_dir, dataset_name)

        # Intervention dataset saved
        logger.debug(f"Intervention dataset saved to: {output_path}")
        logger.debug(f"Generated {len(results_df)} valid intervention samples")

        return results_df

    def check_chemical_validity(self, mol: Chem.Mol) -> Dict:
        """
        Check molecular chemical validity.

        Args:
            mol: Molecule object

        Returns:
            Dictionary containing various chemical properties
        """
        try:
            validity_metrics = {}

            # QED (Quantitative Estimate of Drug-likeness)
            validity_metrics['qed'] = QED.qed(mol)

            # Molecular weight
            validity_metrics['mol_weight'] = Descriptors.ExactMolWt(mol)

            # LogP (Lipophilicity)
            validity_metrics['logp'] = Descriptors.MolLogP(mol)

            # Hydrogen bond donors/acceptors
            validity_metrics['hbd'] = Lipinski.NumHDonors(mol)
            validity_metrics['hba'] = Lipinski.NumHAcceptors(mol)

            # Rotatable bonds
            validity_metrics['rotatable_bonds'] = Lipinski.NumRotatableBonds(mol)

            # Number of rings (use rdMolDescriptors instead of Lipinski)
            validity_metrics['num_rings'] = rdMolDescriptors.CalcNumRings(mol)

            # TPSA (Topological Polar Surface Area)
            validity_metrics['tpsa'] = Descriptors.TPSA(mol)

            # Comprehensive validity score
            # For small molecules (e.g., benzene, simple hydrocarbons), relax requirements
            qed_threshold = self.chemical_validity_threshold
            mol_weight_threshold = 50

            # For aromatic compounds, relax QED requirements
            if validity_metrics['num_rings'] > 0:
                qed_threshold = max(0.3, self.chemical_validity_threshold * 0.6)

            # For simple hydrocarbons, further relax requirements
            if (validity_metrics['num_rings'] == 0 and
                validity_metrics['hbd'] == 0 and
                validity_metrics['hba'] == 0):
                qed_threshold = max(0.2, qed_threshold * 0.5)  # Further relax QED
                mol_weight_threshold = 16  # Allow methane and simple hydrocarbons

            validity_metrics['is_valid'] = (
                validity_metrics['qed'] >= qed_threshold and
                mol_weight_threshold <= validity_metrics['mol_weight'] <= 800 and
                validity_metrics['hbd'] <= 5 and
                validity_metrics['hba'] <= 10
            )

            return validity_metrics

        except Exception as e:
            logger.error(f"Chemical property calculation failed: {str(e)}")
            return {'is_valid': False, 'error': str(e)}

    def _save_intervention_statistics(self, results_df: pd.DataFrame, output_dir: str, dataset_name: str):
        """
        Save intervention statistics.

        Args:
            results_df: Intervention dataset
            output_dir: Output directory
            dataset_name: Dataset name
        """
        try:
            # If no intervention samples, only save basic information
            if len(results_df) == 0:
                stats = {
                    'total_interventions': 0,
                    'unique_original_molecules': 0,
                    'intervention_strategies': {},
                    'chemical_property_stats': {}
                }
            else:
                stats = {
                    'total_interventions': len(results_df),
                    'unique_original_molecules': results_df['original_smiles'].nunique(),
                    'intervention_strategies': {},
                    'chemical_property_stats': {}
                }

                # Strategy statistics (protect against missing columns)
                for strategy in results_df['intervention_strategy'].unique():
                    strategy_data = results_df[results_df['intervention_strategy'] == strategy]

                    avg_qed_change = None
                    if 'intervened_qed' in strategy_data.columns and 'original_qed' in strategy_data.columns:
                        avg_qed_change = (strategy_data['intervened_qed'] - strategy_data['original_qed']).mean()

                    avg_mw_change = None
                    if 'intervened_mol_weight' in strategy_data.columns and 'original_mol_weight' in strategy_data.columns:
                        avg_mw_change = (strategy_data['intervened_mol_weight'] - strategy_data['original_mol_weight']).mean()

                    avg_logp_change = None
                    if 'intervened_logp' in strategy_data.columns and 'original_logp' in strategy_data.columns:
                        avg_logp_change = (strategy_data['intervened_logp'] - strategy_data['original_logp']).mean()

                    chemical_impact = strategy_data['chemical_impact'].iloc[0] if 'chemical_impact' in strategy_data.columns else None

                    stats['intervention_strategies'][strategy] = {
                        'count': len(strategy_data),
                        'avg_qed_change': avg_qed_change,
                        'avg_mol_weight_change': avg_mw_change,
                        'avg_logp_change': avg_logp_change,
                        'chemical_impact': chemical_impact
                    }

                # Chemical property change statistics (calculate if columns exist)
                logp_stats = None
                if 'intervened_logp' in results_df.columns and 'original_logp' in results_df.columns:
                    logp_changes = results_df['intervened_logp'] - results_df['original_logp']
                    logp_stats = {
                        'mean': logp_changes.mean(),
                        'std': logp_changes.std(),
                        'min': logp_changes.min(),
                        'max': logp_changes.max()
                    }

                stats['chemical_property_stats'] = {
                    'logp_change': logp_stats
                }

            # Save statistics
            stats_path = os.path.join(output_dir, f'Intervention_{dataset_name}_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            logger.debug(f"Statistics saved to: {stats_path}")

        except Exception as e:
            # Failed to save statistics (silent record)
            logger.debug(f"Failed to save statistics: {str(e)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Causal Intervention Dataset Generator')
    parser.add_argument('--dataset', type=str, default='bace', help='Dataset name')
    parser.add_argument('--strategies', nargs='+', help='List of intervention strategies')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--max_samples', type=int, help='Maximum number of samples to process (for testing)')
    parser.add_argument('--chemical_threshold', type=float, default=0.7, help='Chemical validity threshold (QED)')

    args = parser.parse_args()

    # Set output directory
    if not args.output_dir:
        args.output_dir = f'data/{args.dataset}'

    # Initialize causal intervention generator
    generator = CausalInterventionGenerator(args.chemical_threshold)

    # Generate intervention dataset
    dataset_path = f'data/{args.dataset}/{args.dataset}.csv'

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file does not exist: {dataset_path}")
        return

    logger.info(f"Processing dataset: {args.dataset}")
    intervention_df = generator.generate_intervention_dataset(
        dataset_path, args.output_dir, args.strategies, args.max_samples
    )

    if len(intervention_df) == 0:
        logger.warning("No valid intervention samples generated")
        return

    # Output summary
    logger.debug("\n" + "="*50)
    logger.debug("Causal Intervention Results Summary:")
    logger.debug(f"- Number of original molecules processed: {intervention_df['original_smiles'].nunique()}")
    logger.debug(f"- Number of intervention samples generated: {len(intervention_df)}")
    logger.debug("="*50)


if __name__ == '__main__':
    main()
