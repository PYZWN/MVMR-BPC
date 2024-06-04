import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromFASTA(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return np.array(features), np.array(edge_index)



def mol_to_graph(seq):

    # 最大原子数量
    max_atoms = 100 
    mol = Chem.MolFromFASTA(seq)
    num_atoms = mol.GetNumAtoms()

    # Atom features
    atom_features_list = [atom_features(atom) for atom in mol.GetAtoms()]

    # Ensure that the atom features matrix size is max_atoms
    atom_features_matrix = np.zeros((max_atoms, len(atom_features_list[0])), dtype=np.float32)
    for i in range(min(num_atoms, max_atoms)):
        atom_features_matrix[i] = atom_features_list[i]

    atom_features_matrix = np.array(atom_features_matrix, dtype=np.float32)

    # Adjacency matrix
    adj_matrix = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    
    # Ensure that the adjacency matrix does not exceed max_atoms
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        
        # Check if the indices are within the allowed range
        if i < max_atoms and j < max_atoms:
            adj_matrix[i, j] = bond_type
            adj_matrix[j, i] = bond_type

    return atom_features_matrix, adj_matrix


# # seq = "FMVFMVPF"
# mol = Chem.MolFromFASTA("FMVFMVPF")

# features, edge_index = mol_to_graph()
# print(features.shape)
# print(edge_index.shape)

def GenGraphData(root):
    dirs = ["ACP","AMP"]
    feature_dict = {}
    adj_dict = {}
    for dir in dirs:
        print('\n')
        print('now is ', dir)
        file = '{}CD_.txt'.format(dir)
        file_path = os.path.join(root, dir, file)
        with open(file_path) as f:
            for each in f:
                if each == '\n' or each[0] == '>':
                    continue
                else:
                    feature, adj = mol_to_graph(each.rstrip())
                    feature_dict[each.rstrip()] = feature
                    print(feature.shape)
                    adj_dict[each.rstrip()] = adj
    f_save = open('AMPACPgraph.pkl', 'wb')
    pickle.dump(feature_dict, f_save)
    f_save.close()
    f_save = open('AMPACPgraph_adj.pkl', 'wb')
    pickle.dump(adj_dict, f_save)
    f_save.close()
