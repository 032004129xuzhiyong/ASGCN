# -*- coding:utf-8 -*-
import os.path
import torch
import numpy as np
import scipy.sparse as sp
from scipy import io
from sklearn.neighbors import kneighbors_graph
import random


def load_mat(mat_path, topk=10, train_ratio=0.1):
    """
    load dataset
    :param mat_path: str file path
    :param topk: int default 10
    :param train_ratio: float [0,1]
    :return:
        adjs, inputs, labels, train_bool, val_bool, n_view, n_node, n_feats, n_class
    """
    print('=' * 15, 'Load data ', os.path.basename(mat_path), '=' * 15)
    mat = io.loadmat(mat_path)
    X = mat['X'][0]  # [n_view,n_nodes,n_feature]
    Y = np.squeeze(mat['Y']) - np.min(mat['Y'])  # [n_nodes]

    labels = Y
    n_class = len(np.unique(labels))
    n_view = X.shape[0]
    n_node = X[0].shape[0]
    adjs = []
    inputs = []
    n_feats = []
    for i in range(n_view):
        tempX = X[i]
        if sp.isspmatrix(tempX):
            tempX = tempX.toarray()
        inputs.append(torch.from_numpy(tempX.astype(np.float32)).float())
        adjs.append(get_adj_matrix(tempX, topk).to_dense().float())
        n_feats.append(len(tempX[0]))
    train_bool, val_bool = split_train_and_val_to_get_bool_ind(labels, train_ratio)
    labels = torch.from_numpy(labels)
    train_bool = torch.from_numpy(train_bool)
    val_bool = torch.from_numpy(val_bool)
    print(f'n_view: {n_view} n_node: {n_node} n_feats: {n_feats} n_class: {n_class}')
    print(f'labels: {labels.shape} train_bool: {train_bool.sum()} val_bool: {val_bool.sum()}')
    return adjs, inputs, labels, train_bool, val_bool, n_view, n_node, n_feats, n_class


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_adj_matrix(X_view, topk):
    """
    get adjacent matrix from x_veiw with knn method.
    :param X_view: [n_nodes, n_features]
    :param topk: int
    :return:
        A_: #[n_nodes,n_nodes] 规范化的邻接矩阵
    """
    temp = kneighbors_graph(X_view, topk)
    temp = sp.coo_matrix(temp)
    temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
    temp = normalize(temp + sp.eye(temp.shape[0]))
    A_ = sparse_mx_to_torch_sparse_tensor(temp)
    return A_


def split_train_and_val_to_get_bool_ind(labels:np.ndarray, train_ratio:float):
    """
    generate bool array to show whether to calculate loss
    :param labels: all labels
    :param train_ratio: float [0,1]
    :return:
        train_bool, val_bool
        note: len(train_bool) == len(val_bool)
    """
    class_set = list(np.unique(labels))
    num_class = len(class_set)
    total_num = labels.shape[0]
    need_num = int(total_num * train_ratio)
    num_per_class_ = round(need_num / num_class)  # 平均每个类的个数
    pre_class_num_list = []
    for i in range(num_class - 1):
        temp = random.randint(int(num_per_class_),int(num_per_class_)+1)
        if temp <= 0:
            temp = 1
        pre_class_num_list.append(temp)
    last_class_num = need_num - sum(pre_class_num_list)
    num_per_class = pre_class_num_list + [last_class_num]
    random.shuffle(num_per_class)
    num_per_class = {class_set[idx]: num_per_class[idx] for idx in range(len(class_set))}
    labels_index = list(range(total_num))
    random.shuffle(labels_index)
    labels = labels[labels_index]

    train_bool = np.zeros_like(labels, dtype=bool)
    for idx, label in enumerate(labels):
        if num_per_class[label] > 0:
            num_per_class[label] -= 1
            train_bool[labels_index[idx]] = True
    val_bool = ~train_bool
    return train_bool, val_bool