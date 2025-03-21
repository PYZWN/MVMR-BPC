# -*- coding: utf-8 -*-
# @Author  : zhd
# @FileName: demo.py
# @Software: PyCharm


import os
import time
import numpy as np
from pathlib import Path
from utils import mol_to_graph
dir = 'MKey_Net_DiladCNNBiLSTM_GCN_Attention'
Path(dir).mkdir(exist_ok=True)
t = time.localtime(time.time())
with open(os.path.join(dir, 'time.txt'), 'w') as f:
    f.write('start time: {}m {}d {}h {}m {}s'.format(t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
    f.write('\n')
from sklearn.model_selection import train_test_split



def GetSourceData(root, dir, lb):
    seqs = []
    print('\n')
    print('now is ', dir)
    file = '{}CD_.txt'.format(dir)
    file_path = os.path.join(root, dir, file)

    with open(file_path) as f:
        for each in f:
            if each == '\n' or each[0] == '>':
                continue
            else:
                seqs.append(each.rstrip())

    # data and label
    label = len(seqs) * [lb]
    seqs_train, seqs_test, label_train, label_test = train_test_split(seqs, label, test_size=0.2, random_state=0)
    print('train data:', len(seqs_train))
    print('test data:', len(seqs_test))
    print('train label:', len(label_train))
    print('test_label:', len(label_test))
    print('total numbel:', len(seqs_train)+len(seqs_test))

    return seqs_train, seqs_test, label_train, label_test



def DataClean(data):
    max_len = 0
    for i in range(len(data)):
        st = data[i]
        # get the maximum length of all the sequences
        if(len(st) > max_len): max_len = len(st)

    return data, max_len

# new
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import networkx as nx

def ligand_MACCSKey(seq):
    features = []
    mol = Chem.MolFromFASTA(seq)

    fingerprints = MACCSkeys.GenMACCSKeys(mol)

    for i in range(1, len(fingerprints.ToBitString())):
        features.append(int(fingerprints.ToBitString()[i]))
    return features

from rdkit.Chem import AllChem

def ligand_ECFP(seq, radius=2):
    features = []
    mol = Chem.MolFromFASTA(seq)

    # Generate ECFP
    fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, radius)

    for bit in fingerprints:
        features.append(bit)

    return features

def ligand_DaylightFingerprint(seq):
    features = []
    mol = Chem.MolFromFASTA(seq)

    # 生成 Daylight 指纹
    fingerprints = Chem.RDKFingerprint(mol)

    for bit in fingerprints:
        features.append(bit)

    return features
# def NetPeptide(seq):
#     import pickle
#     f = open("./net_data/net_file_1000.pkl", 'rb')
#     net_feature = pickle.load(f)
#     f.close()
#     return net_feature[seq]

def PadEncode(data, max_len):

    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e = []
    data_key = []
    data_net = []
    data_features_matrix = []
    data_adj_matrix = []
    data_esm = []
    import pickle
    f = open("./net_data/net_file_60.pkl", 'rb')
    net_feature = pickle.load(f)
    g_f = open("./graph_data/graph_feature_file_100.pkl", 'rb')
    graph_features = pickle.load(g_f)
    a_f = open("./graph_data/graph_adj_file_100.pkl", 'rb')
    graph_adjs = pickle.load(a_f)
    e_f = open("sequencefeature.pkl", 'rb')
    esm_feature = pickle.load(e_f)
    for i in range(len(data)):
        ecfp_features = ligand_ECFP(data[i])
        Daylight_features = ligand_DaylightFingerprint(data[i])

        # 点对点结合，直接拼接两个特征向量
        fused_features = [ecfp + dl for ecfp, dl in zip(ecfp_features, Daylight_features)]
        data_key.append(fused_features)
        data_net.append(net_feature[data[i]])
        data_features_matrix.append(graph_features[data[i]])
        data_adj_matrix.append(graph_adjs[data[i]])
        data_esm.append(esm_feature[data[i]])
        length = len(data[i])
        elemt, st = [], data[i]
        for j in st:
            index = amino_acids.index(j)
            elemt.append(index)
        if length < max_len:
            elemt += [0]*(max_len-length)
        data_e.append(elemt)
    f.close()

    return data_e, data_key, data_net, data_features_matrix, data_adj_matrix, data_esm




def GetSequenceData(dirs, root):
    # getting training data and test data
    count, max_length = 0, 0
    tr_data, te_data, tr_label, te_label = [], [], [], []
    for dir in dirs:
        # 1.getting data from file
        tr_x, te_x, tr_y, te_y = GetSourceData(root, dir, count)
        count += 1

        # 2.getting the maximum length of all sequences
        tr_x, len_tr = DataClean(tr_x)
        te_x, len_te = DataClean(te_x)
        if len_tr > max_length: max_length = len_tr
        if len_te > max_length: max_length = len_te

        # 3.dataset
        tr_data += tr_x
        te_data += te_x
        tr_label += tr_y
        te_label += te_y


    # data coding and padding vector to the filling length
    trainSeqdata, trainKeydata, trainNetdata, trainFeaturedata, trainAdjdata, trainesmdata = PadEncode(tr_data, max_length)
    testSeqdata, testKeydata, testNetdata, testFeaturedata, testAdjdata, testesmdata = PadEncode(te_data, max_length)

    # data type conversion
    train_seq_data = np.array(trainSeqdata)
    train_key_data = np.array(trainKeydata)
    train_net_data = np.array(trainNetdata)
    train_net_data = train_net_data.reshape(train_net_data.shape[0], train_net_data.shape[1], train_net_data.shape[2], 1)
    train_feature_data = np.array(trainFeaturedata)
    train_adj_data = np.array(trainAdjdata)
    train_esm_data = np.array(trainesmdata)

    test_seq_data = np.array(testSeqdata)
    test_key_data = np.array(testKeydata)
    test_net_data = np.array(testNetdata)
    test_net_data = test_net_data.reshape(test_net_data.shape[0], test_net_data.shape[1], test_net_data.shape[2], 1)
    test_feature_data = np.array(testFeaturedata)
    test_adj_data = np.array(testAdjdata)
    test_esm_data = np.array(testesmdata)

    train_label = np.array(tr_label)
    test_label = np.array(te_label)

    return [train_seq_data, train_key_data, train_net_data,train_feature_data, train_adj_data, train_esm_data, test_seq_data,test_key_data, test_net_data,test_feature_data, test_adj_data, test_esm_data, train_label, test_label]



def GetData(path):
    dirs = ['AMP', 'ACP', 'ADP', 'AHP', 'AIP'] # functional peptides

    # get sequence data
    sequence_data = GetSequenceData(dirs, path)

    return sequence_data



def TrainAndTest(tr_seq_data, tr_key_data, tr_net_data, tr_feature_data, tr_adj_data, tr_esm_data,tr_label, te_seq_data, te_key_data, te_net_data, te_feature_data, te_adj_data, te_esm_data,te_label):

    from train import train_main # load my training function

    train = [tr_seq_data, tr_key_data, tr_net_data, tr_feature_data, tr_adj_data, tr_esm_data,tr_label]
    test = [te_seq_data, te_key_data, te_net_data, te_feature_data, te_adj_data, te_esm_data,te_label]

    threshold = 0.5
    model_num = 1  # model number 4
    test.append(threshold)
    train_main(train, test, model_num, dir)

    ttt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('finish time: {}m {}d {}h {}m {}s'.format(ttt.tm_mon, ttt.tm_mday, ttt.tm_hour, ttt.tm_min, ttt.tm_sec))

def Test(tr_data, tr_label, te_data, te_label):

    from train import test_main # load my training function

    train = [tr_data, tr_label]
    test = [te_data, te_label]

    threshold = 0.5
    model_num = 1  # model number 3
    test.append(threshold)
    test_main(train, test, model_num, dir)

    ttt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('finish time: {}m {}d {}h {}m {}s'.format(ttt.tm_mon, ttt.tm_mday, ttt.tm_hour, ttt.tm_min, ttt.tm_sec))




def main():
    # I.get sequence data
    path = 'data1' # data path
    sequence_data = GetData(path)


    # sequence data partitioning
    tr_seq_data,tr_key_data,tr_net_data,tr_feature_data,tr_adj_data, tr_esm_data ,te_seq_data,te_key_data,te_net_data,te_feature_data, te_adj_data, te_esm_data, tr_seq_label,te_seq_label = \
        sequence_data[0],sequence_data[1],sequence_data[2],sequence_data[3],sequence_data[4],sequence_data[5],sequence_data[6],sequence_data[7],sequence_data[8],sequence_data[9],sequence_data[10],sequence_data[11],sequence_data[12],sequence_data[13]


    # II.training and testing
    TrainAndTest(tr_seq_data, tr_key_data, tr_net_data, tr_feature_data, tr_adj_data, tr_esm_data, tr_seq_label, te_seq_data, te_key_data, te_net_data, te_feature_data, te_adj_data,te_esm_data, te_seq_label)
    # Test(tr_seq_data, tr_seq_label, te_seq_data, te_seq_label)





if __name__ == '__main__':
    # executing the main function
    main()