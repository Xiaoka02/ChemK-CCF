# -*- coding: utf-8 -*-
from argparse import Namespace
from typing import List, Tuple, Union
from rdkit import Chem
import torch

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
    'is_aromatic': [0, 1],
    'is_in_ring': [0, 1],
    'ring_size': [0, 3, 4, 5, 6, 7, 8],
    'num_radical_electrons': [0, 1, 2],
}

ATOM_FDIM = (
    sum(len(choices) + 1 for k, choices in ATOM_FEATURES.items()
        if k not in ['is_aromatic', 'is_in_ring']) +
    sum(len(choices) for k, choices in ATOM_FEATURES.items()
        if k in ['is_aromatic', 'is_in_ring']) + 1
)
BOND_FDIM = 17

SMILES_TO_GRAPH = {}


def clear_cache():
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim() -> int:
    return ATOM_FDIM


def get_bond_fdim() -> int:
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_features(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    features = []

    # Fundamental characteristic
    features.extend(onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']))
    features.extend(onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']))
    features.extend(onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    features.extend(onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']))
    features.extend(onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']))
    features.extend(onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']))

    features.extend([0, 1] if atom.GetIsAromatic() else [1, 0])  # is_aromatic
    features.extend([0, 1] if atom.IsInRing() else [1, 0])       # is_in_ring

    ring_size = 0
    for size in range(3, 9):
        if atom.IsInRingSize(size):
            ring_size = size
            break
    features.extend(onek_encoding_unk(ring_size, ATOM_FEATURES['ring_size']))
    features.extend(onek_encoding_unk(atom.GetNumRadicalElectrons(), ATOM_FEATURES['num_radical_electrons']))

    features.append(atom.GetMass() * 0.01)

    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    if bond is None:
        return [1] + [0] * (BOND_FDIM - 1)

    bt = bond.GetBondType()
    features = [
        0,
    ]

    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    features.extend(onek_encoding_unk(bt, bond_types))

    features.extend(onek_encoding_unk(int(bond.GetStereo()), list(range(6))))

    features.extend([0, 1] if bond.GetIsConjugated() else [1, 0])

    features.extend([0, 1] if bond.IsInRing() else [1, 0])

    return features


class MolGraph:
    def __init__(self, smiles: str, args: Namespace):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)

        self.num_nodes = self.mol.GetNumAtoms()
        self.num_edges = self.mol.GetNumBonds() * 2

        self.f_atoms = []
        for atom in self.mol.GetAtoms():
            self.f_atoms.append(atom_features(atom))

        self.edge_index = []
        self.f_bonds = []
        self.a2b = []
        self.b2a = []
        self.b2revb = []

        for _ in range(self.num_nodes):
            self.a2b.append([])

        for a1 in range(self.num_nodes):
            for a2 in range(a1 + 1, self.num_nodes):
                bond = self.mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)

                self.f_bonds.extend([f_bond, f_bond])

                b1 = len(self.b2a)
                b2 = b1 + 1

                # a1 -> a2
                self.edge_index.extend([[a1, a2], [a2, a1]])
                self.a2b[a2].append(b1)
                self.b2a.append(a1)

                # a2 -> a1
                self.a2b[a1].append(b2)
                self.b2a.append(a2)

                self.b2revb.extend([b2, b1])


class BatchMolGraph:
    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.num_graphs = len(mol_graphs)

        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()

        self.n_atoms = 1
        self.n_bonds = 1
        self.a_scope = []
        self.b_scope = []

        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        a2b = [[]]
        b2a = [0]
        b2revb = [0]
        bonds = [[0, 0]]

        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.num_nodes):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(len(mol_graph.b2a)):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.num_nodes))
            self.b_scope.append((self.n_bonds, len(mol_graph.f_bonds)))

            self.n_atoms += mol_graph.num_nodes
            self.n_bonds += len(mol_graph.f_bonds)

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a][:self.max_num_bonds] +
                                     [0] * (self.max_num_bonds - len(a2b[a]))
                                     for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.bonds = torch.LongTensor(bonds)

        self.batch = torch.repeat_interleave(
            torch.arange(self.num_graphs),
            torch.tensor([mol_graph.num_nodes for mol_graph in mol_graphs])
        )

    def get_components(self) -> Tuple:
        return (self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb,
                self.a_scope, self.b_scope, self.bonds)


def mol2graph(smiles_batch: List[str], args: Namespace) -> BatchMolGraph:
    mol_graphs = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_GRAPH:
            mol_graph = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles, args)
            if not args.no_cache:
                SMILES_TO_GRAPH[smiles] = mol_graph
        mol_graphs.append(mol_graph)

    return BatchMolGraph(mol_graphs, args)