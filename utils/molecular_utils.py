# -*- coding: utf-8 -*-
import torch
from rdkit import Chem


class MolecularProcessor:
    def __init__(self, config):
        self.config = config
        self.atom_features = {
            'atom_type': ['C', 'N', 'O', 'H', 'F', 'Cl', 'Br', 'I', 'P', 'S'],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3
            ]
        }
    
    def get_atom_features(self, atom):
        """Extract atomic features"""
        features = []
        
        # Atomic type
        atom_type = [1 if atom.GetSymbol() == s else 0 for s in self.atom_features['atom_type']]
        features.extend(atom_type)
        
        # Formation charge
        features.append(atom.GetFormalCharge())
        
        # Hybrid type
        hybridization = [1 if atom.GetHybridization() == h else 0 
                        for h in self.atom_features['hybridization']]
        features.extend(hybridization)
        
        # Aromatic
        features.append(int(atom.GetIsAromatic()))
        
        return features
    
    def get_bond_features(self, bond):
        """Extracting chemical bond features"""
        if bond is None:
            return [0] * 4
        
        bond_type = bond.GetBondType()
        features = [
            1 if bond_type == Chem.rdchem.BondType.SINGLE else 0,
            1 if bond_type == Chem.rdchem.BondType.DOUBLE else 0,
            1 if bond_type == Chem.rdchem.BondType.TRIPLE else 0,
            1 if bond_type == Chem.rdchem.BondType.AROMATIC else 0
        ]
        return features
    
    def get_molecular_constraints(self, mol):
        """Get molecular constraints"""
        num_atoms = mol.GetNumAtoms()
        bond_constraints = torch.zeros((num_atoms, num_atoms))
        valence_constraints = torch.zeros(num_atoms)
        
        # Set chemical bond constraints
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_features = self.get_bond_features(bond)
            bond_constraints[i, j] = sum(bond_features)
            bond_constraints[j, i] = sum(bond_features)
        
        # Set atomic valence constraints
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            if symbol in self.config.max_valence:
                valence_constraints[i] = self.config.max_valence[symbol]
        
        return bond_constraints, valence_constraints
    
    def mol_to_graph(self, smiles):
        """Convert SMILES to a graphical representation"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Get atomic features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        
        # Get molecular constraints
        bond_constraints, valence_constraints = self.get_molecular_constraints(mol)
        
        return {
            'atom_features': torch.tensor(atom_features, dtype=torch.float),
            'bond_constraints': bond_constraints,
            'valence_constraints': valence_constraints
        }
    
    def process_batch(self, smiles_list):
        """Processing bulk data"""
        graphs = []
        for smiles in smiles_list:
            graph = self.mol_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)

        max_atoms = max(g['atom_features'].shape[0] for g in graphs)
        max_atoms = min(max_atoms, self.config.max_atoms)
        
        batch_features = []
        batch_bond_constraints = []
        batch_valence_constraints = []
        
        for graph in graphs:
            num_atoms = graph['atom_features'].shape[0]
            if num_atoms > max_atoms:

                features = graph['atom_features'][:max_atoms]
                bond_cons = graph['bond_constraints'][:max_atoms, :max_atoms]
                valence_cons = graph['valence_constraints'][:max_atoms]
            else:

                features = torch.cat([
                    graph['atom_features'],
                    torch.zeros(max_atoms - num_atoms, graph['atom_features'].shape[1])
                ])
                bond_cons = torch.nn.functional.pad(
                    graph['bond_constraints'],
                    (0, max_atoms - num_atoms, 0, max_atoms - num_atoms)
                )
                valence_cons = torch.cat([
                    graph['valence_constraints'],
                    torch.zeros(max_atoms - num_atoms)
                ])
            
            batch_features.append(features)
            batch_bond_constraints.append(bond_cons)
            batch_valence_constraints.append(valence_cons)
        
        return {
            'atom_features': torch.stack(batch_features),
            'bond_constraints': torch.stack(batch_bond_constraints),
            'valence_constraints': torch.stack(batch_valence_constraints)
        } 