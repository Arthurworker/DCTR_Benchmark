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
parser.add_argument("--threshold", type=int, default=0)
parser.add_argument('--problem', type=str, default="coat")
parser.add_argument("--dataset1", type=Path, default='data/coat/traindata.csv')
parser.add_argument("--dataset2", type=Path, default='data/coat/testdata.csv')
parser.add_argument("--stats", type=Path, default='data/coat/stats')
parser.add_argument("--record", type=Path, default='data/coat/tfrecord')
parser.add_argument("--record2", type=Path, default='data/coat/tfrecord2')
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


class CoatTransform(DataTransform):
    def __init__(self, dataset_path1, dataset_path2, path, path2, stats_path, min_threshold, label_index, type,
                 ratio_tep3, ratio_trp2, ratio_tep2, store_stat=False, seed=2021):
        super(CoatTransform, self).__init__(dataset_path1, dataset_path2, stats_path, store_stat=store_stat, seed=seed)
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
        self.name = "User_feat_id,User_gender,User_age,User_location,User_fashioninterest,Item_feat_id,Item_gender,Item_jackettype,Item_color,Item_onfrontpage,User_id,Item_id,Label".split(
            ",")

    def process(self):
        self._read(name=self.name, header=None, sep=",", label_index=self.label, id=self.id)
        if self.store_stat:
            self.generate_and_filter(threshold=self.threshold, label_index=self.label, id=self.id)

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

    def _process_y(self):
        self.traindata["Label"] = self.traindata["Label"].apply(lambda x: 0 if x == 0 else 1)
        self.testdata["Label"] = self.testdata["Label"].apply(lambda x: 0 if x == 0 else 1)
        self.data["Label"] = self.data["Label"].apply(lambda x: 0 if x == 0 else 1)


if __name__ == "__main__":
    tranformer = CoatTransform(args.dataset1, args.dataset2, args.record, args.record2, args.stats, args.threshold,
                               args.label, args.type, args.ratio_tep3, args.ratio_trp2, args.ratio_tep2,
                               store_stat=args.store_stat)
    tranformer.process()
    coat2structure(args.problem, args.record)
