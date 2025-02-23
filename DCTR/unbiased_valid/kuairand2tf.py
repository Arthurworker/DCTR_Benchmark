import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sparse
from utils import trainUtils

from datatransform.transform_coat import DataTransform
import argparse
from pathlib import Path
from sklearn.utils import shuffle
parser = argparse.ArgumentParser(description='Transfrom original data to TFRecord')


parser.add_argument('--label', type=str, default="Label")
parser.add_argument("--store_stat", action="store_true", default=True)
parser.add_argument("--threshold", type=int, default=2)
parser.add_argument('--problem', type=str, default="kuairand")
parser.add_argument("--dataset1", type=Path, default='data/kuairand/s_data.csv')
parser.add_argument("--dataset2", type=Path, default='data/kuairand/r_data.csv')
parser.add_argument("--stats", type=Path, default='data/kuairand/stats_2')
parser.add_argument("--record", type=Path, default='data/kuairand/threshold_2')
parser.add_argument("--record2", type=Path, default='data/kuairand/threshold2_2')
parser.add_argument('--type', type=str, default="tep3")
parser.add_argument("--ratio_tep3", nargs='+', type=float, default=[0.1, 0.1, 0.8])
parser.add_argument("--ratio_trp2", nargs='+', type=float, default=[0.8, 0.2, 0.0])
parser.add_argument("--ratio_tep2", nargs='+', type=float, default=[0.1, 0.0, 0.9])

args = parser.parse_args()


def coat2structure(dataset, data_dir):
    dataloader = trainUtils.getDataLoader(dataset, str(data_dir))
    data_path = 'data/' + dataset

    #
    user_feat_dict, item_feat_dict = dataloader.get_feat_dict("train", batch_size=512)

    with open('data/{}/user_feat_dict.pkl'.format(dataset), 'wb') as f:
        pickle.dump(user_feat_dict, f)

    with open('data/{}/item_feat_dict.pkl'.format(dataset), 'wb') as f:
        pickle.dump(item_feat_dict, f)

    #
    train_data = dataloader.get_train_data("train", batch_size=512)
    users, items, labels = [], [], []
    for _, user_id, item_id, label, _ in train_data:
        users.extend(user_id.squeeze().numpy())
        items.extend(item_id.squeeze().numpy())
        labels.extend(label.squeeze().numpy())

    users, items, labels = np.array(users, dtype=int), np.array(items, dtype=int), np.array(labels)
    labels[labels == 0] = -1
    print('num of train:', np.size(users))
    train = sparse.csr_matrix((labels, (users, items)), shape=(max(users) + 1, max(items) + 1), dtype=np.float32)
    np.save(data_path + '/matrix_train.npy', train.todense())

    utrain_data = dataloader.get_train_data("utrain", batch_size=512)
    users, items, labels = [], [], []
    for _, user_id, item_id, label, _ in utrain_data:
        users.extend(user_id.squeeze().numpy())
        items.extend(item_id.squeeze().numpy())
        labels.extend(label.squeeze().numpy())

    users, items, labels = np.array(users, dtype=int), np.array(items, dtype=int), np.array(labels)
    labels[labels == 0] = -1
    print('num of utrain:', np.size(users))
    utrain = sparse.csr_matrix((labels, (users, items)), shape=(max(users) + 1, max(items) + 1), dtype=np.float32)
    np.save(data_path + '/matrix_utrain.npy', utrain.todense())

    matrix_train = np.load(data_path + '/matrix_train.npy')
    matrix_utrain = np.load(data_path + '/matrix_utrain.npy')
    print('shape of matrix_train:', matrix_train.shape)
    print(sum(sum(matrix_train != 0)))
    print('shape of matrix_utrain:', matrix_utrain.shape)
    print(sum(sum(matrix_utrain != 0)))


class KuairandTransform(DataTransform):
    def __init__(self, dataset_path1, dataset_path2, path, path2, stats_path, min_threshold, label_index, type,
                 ratio_tep3, ratio_trp2, ratio_tep2, store_stat=False, seed=2021):
        super(KuairandTransform, self).__init__(dataset_path1, dataset_path2, stats_path, store_stat=store_stat,
                                                seed=seed)
        self.threshold = min_threshold
        self.label = label_index
        self.id = ["User_id", "Item_id"]
        self.type = type
        self.split_tep3 = ratio_tep3
        self.split_trp2 = ratio_trp2
        self.split_tep2 = ratio_tep2
        self.path = path
        self.path2 = path2
        self.stats_path = stats_path
        self.name = ['Uc1', 'Uc2', 'Uc3', 'Uc4', 'Uc5', 'Uc6', 'Uc7', 'Uc8', 'Uc9', 'Uc10', 'Uc11', 'Uc12', 'Uc13',
                     'Uc14', 'Uc15', 'Uc16', 'Uc17', 'Uc18', 'Uc19', 'Uc20', 'Uc21', 'Uc22', 'Uc23', 'Uc24', 'Uc25',
                     'Uc26', 'Uc27', 'Ui28', 'Ui29', 'Ui30', 'Ui31', 'Ic32', 'Ic33', 'Ic34', 'Ic35', 'Ii36', 'Ii37',
                     'Ii38', 'Ii39', 'Ii40', 'Ii41', 'Ii42', 'Ii43', 'Ii44', 'Ii45', 'Ii46', 'Ii47', 'Ii48', 'Ii49',
                     'Ii50', 'Ii51', 'Ii52', 'Ii53', 'Ii54', 'Ii55', 'Ii56', 'Ii57', 'Ii58', 'Ii59', 'Ii60', 'Ii61',
                     'Ii62', 'Ii63', 'Ii64', 'Ii65', 'Ii66', 'Ii67', 'Ii68', 'Ii69', 'Ii70', 'Ii71', 'Ii72', 'Ii73',
                     'Ii74', 'Ii75', 'Ii76', 'Ii77', 'Ii78', 'Ii79', 'Ii80', 'Ii81', 'Ii82', 'Ii83', 'Ii84', 'Ii85',
                     'Ii86', 'Ii87', 'Ii88', 'User_id', 'Item_id', 'Label']

    def process(self):
        self._read(name=self.name, header=None, sep=",", label_index=self.label, id=self.id)
        if self.store_stat:
            white_list = ['Ui28', 'Ui29', 'Ui30', 'Ui31', 'Ii36', 'Ii37', 'Ii38', 'Ii39', 'Ii40', 'Ii41', 'Ii42',
                          'Ii43', 'Ii44', 'Ii45', 'Ii46', 'Ii47', 'Ii48', 'Ii49', 'Ii50', 'Ii51', 'Ii52', 'Ii53',
                          'Ii54', 'Ii55', 'Ii56', 'Ii57', 'Ii58', 'Ii59', 'Ii60', 'Ii61', 'Ii62', 'Ii63', 'Ii64',
                          'Ii65', 'Ii66', 'Ii67', 'Ii68', 'Ii69', 'Ii70', 'Ii71', 'Ii72', 'Ii73', 'Ii74', 'Ii75',
                          'Ii76', 'Ii77', 'Ii78', 'Ii79', 'Ii80', 'Ii81', 'Ii82', 'Ii83', 'Ii84', 'Ii85', 'Ii86',
                          'Ii87', 'Ii88']
            print('white_list len:', len(white_list))
            self.generate_and_filter(threshold=self.threshold, label_index=self.label, id=self.id,
                                     white_list=white_list)

        if self.type == 'tep3':
            self.traindata = shuffle(self.traindata, random_state=self.seed)
            self.transform_tfrecord(self.traindata, self.path, "train", label_index=self.label, id=self.id, mark=1)
            self.data = self.testdata
            tr, te, val = self.random_split(ratio=self.split_tep3)
            self.transform_tfrecord(tr, self.path, "utrain", label_index=self.label, id=self.id, mark=0)
            self.transform_tfrecord(te, self.path, "test", label_index=self.label, id=self.id, mark=0)
            self.transform_tfrecord(val, self.path, "validation", label_index=self.label, id=self.id, mark=0)
        else:
            self.traindata = shuffle(self.traindata, random_state=self.seed)
            self.data = self.traindata
            tr, te, val = self.random_split(ratio=self.split_trp2)
            self.transform_tfrecord(tr, self.path2, "train", label_index=self.label, id=self.id, mark=1)
            self.transform_tfrecord(val, self.path2, "validation", label_index=self.label, id=self.id, mark=1)

            self.data = self.testdata
            tr, te, val = self.random_split(ratio=self.split_tep2)
            self.transform_tfrecord(tr, self.path2, "utrain", label_index=self.label, id=self.id, mark=0)
            self.transform_tfrecord(te, self.path2, "test", label_index=self.label, id=self.id, mark=0)

    def _process_x(self):
        print(self.data[self.data["Label"] == 1].shape)

        def bucket(value):
            if not pd.isna(value):
                if value > 2:
                    value = int(np.floor(np.log(value) ** 2))
                else:
                    value = int(value)
            return value
        numeric_list = ['Ui28', 'Ui29', 'Ui30', 'Ui31', 'Ii36', 'Ii37', 'Ii38', 'Ii39', 'Ii40', 'Ii41', 'Ii42', 'Ii43',
                        'Ii44', 'Ii45', 'Ii46', 'Ii47', 'Ii48', 'Ii49', 'Ii50', 'Ii51', 'Ii52', 'Ii53', 'Ii54', 'Ii55',
                        'Ii56', 'Ii57', 'Ii58', 'Ii59', 'Ii60', 'Ii61', 'Ii62', 'Ii63', 'Ii64', 'Ii65', 'Ii66', 'Ii67',
                        'Ii68', 'Ii69', 'Ii70', 'Ii71', 'Ii72', 'Ii73', 'Ii74', 'Ii75', 'Ii76', 'Ii77', 'Ii78', 'Ii79',
                        'Ii80', 'Ii81', 'Ii82', 'Ii83', 'Ii84', 'Ii85', 'Ii86', 'Ii87', 'Ii88']
        for col_name in numeric_list:
            self.data[col_name] = self.data[col_name].apply(bucket)

    def _process_y(self):
        pass


if __name__ == "__main__":
    tranformer = KuairandTransform(args.dataset1, args.dataset2, args.record, args.record2, args.stats, args.threshold,
                                   args.label, args.type, args.ratio_tep3, args.ratio_trp2, args.ratio_tep2,
                                   store_stat=args.store_stat)
    tranformer.process()
    coat2structure(args.problem, args.record)
