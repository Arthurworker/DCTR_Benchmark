import torch
import os
import yaml
import stat
import pickle
import numpy as np
import pandas as pd
from os import listdir
from ast import literal_eval
from os.path import isfile, join
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz


def load_pandas(path, name, sep, df_name, value_name, shape=None):
    df = pd.read_csv(path + name, sep=sep)
    rows = df[df_name[0]]
    cols = df[df_name[1]]
    if value_name is not None:
        values = df[value_name]
    else:
        values = [1]*len(rows)

    if shape:
        return csr_matrix((values, (rows, cols)), shape=shape)
    else:
        return csr_matrix((values, (rows, cols)), shape=(rows.max() + 1, cols.max() + 1))


def load_pandas_without_names(path, name, sep, df_name, value_name, shape=None):
    df = pd.read_csv(path + name, sep=sep, header=None, names=df_name)
    rows = df[df_name[0]]
    cols = df[df_name[1]]
    if value_name is not None:
        values = df[value_name]
    else:
        values = [1]*len(rows)

    if shape:
        return csr_matrix((values, (rows, cols)), shape=shape)
    else:
        return csr_matrix((values, (rows, cols)), shape=(rows.max() + 1, cols.max() + 1))


def save_numpy(matrix, path, model):
    save_npz('{0}{1}'.format(path, model), matrix)


def save_array(array, path, model):
    np.save('{0}{1}'.format(path, model), array)


def load_numpy(path, name):
    return load_npz(path+name).tocsr()


def load_dataframe_csv(path, name):
    return pd.read_csv(path+name)


def save_dataframe_csv(df, path, name):
    df.to_csv(path+name, index=False)


def find_best_hyperparameters(folder_path, meatric, scene='r'):
    csv_files = [join(folder_path, f) for f in listdir(folder_path)
                 if isfile(join(folder_path, f)) and f.endswith('tuning_'+scene+'.csv') and not f.startswith('final')]
    best_settings = []
    for record in csv_files:
        df = pd.read_csv(record)
        df[meatric+'_Score'] = df[meatric].map(lambda x: literal_eval(x)[0])
        best_settings.append(df.loc[df[meatric+'_Score'].idxmax()].to_dict())

    df = pd.DataFrame(best_settings).drop(meatric+'_Score', axis=1)

    return df


def find_single_best_hyperparameters(folder_path, meatric):
    df = pd.read_csv(folder_path)
    df[meatric + '_Score'] = df[meatric].map(lambda x: literal_eval(x)[0])
    best_settings = df.loc[df[meatric + '_Score'].idxmax()].to_dict()

    return best_settings


def save_dataframe_latex(df, path, model):
    with open('{0}{1}_parameter_tuning.tex'.format(path, model), 'w') as handle:
        handle.write(df.to_latex(index=False))


def load_csv(path, name, shape=(1010000, 2262292)):
    data = np.genfromtxt(path + name, delimiter=',')
    matrix = csr_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=shape)
    save_npz(path + "rating.npz", matrix)
    return matrix


def save_pickle(path, name, data):
    with open('{0}/{1}.pickle'.format(path, name), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path, name):
    with open('{0}/{1}.pickle'.format(path, name), 'rb') as handle:
        data = pickle.load(handle)

    return data


def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)


def get_file_names(folder_path, extension='.yml'):
    return [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(extension)]


def write_file(folder_path, file_name, content, exe=False):
    full_path = folder_path+'/'+file_name
    with open(full_path, 'w') as the_file:
        the_file.write(content)

    if exe:
        st = os.stat(full_path)
        os.chmod(full_path, st.st_mode | stat.S_IEXEC)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def rating_mat_to_sample(mat):
    row, col = np.nonzero(mat)
    y = mat[row,col]
    y = np.asarray(y, dtype=np.int).flatten()
    y[y == -1] = 0
    x = np.concatenate([row.reshape(-1,1), col.reshape(-1,1)], axis=1)
    return x, y

def shuffle(x, y, seed):
    np.random.seed(seed)
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    return x[idxs], y[idxs]

class McDropout(torch.nn.Module):
    def __init__(self, pdrop = 0.5):
        super(McDropout, self).__init__()
        assert 0 <= pdrop <= 1
        self.pdrop = pdrop

    def forward(self, X):
        mask = (torch.rand(X.shape) > self.pdrop).float()
        return mask * X

