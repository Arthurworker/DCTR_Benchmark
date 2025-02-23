import tensorflow as tf
import glob
import torch
import numpy as np
import os

# tf.enable_eager_execution()

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CoatLoader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        # Feature FIELDS: 12
        self.FIELDS = 12
        self.tfrecord_path = tfrecord_path
        self.description = {
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
            "user_id": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "item_id": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "mark": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
        }

    # read val/test data
    def get_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['user_id'], example['item_id'], example['label'], example['mark']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x, uid, iid, y, m in ds:
            x = torch.from_numpy(x.numpy())
            uid = torch.from_numpy(uid.numpy())
            iid = torch.from_numpy(iid.numpy())
            y = torch.from_numpy(y.numpy())
            m = torch.from_numpy(m.numpy())
            yield x, uid, iid, y, m

    # read train data
    def get_train_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['user_id'], example['item_id'], example['label'], example['mark']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        data = []
        for x, uid, iid, y, m in ds:
            x = torch.from_numpy(x.numpy())
            uid = torch.from_numpy(uid.numpy())
            iid = torch.from_numpy(iid.numpy())
            y = torch.from_numpy(y.numpy())
            m = torch.from_numpy(m.numpy())
            data.append([x, uid, iid, y, m])
        return data

    # read train and utrain data
    def get_comb_train_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['user_id'], example['item_id'], example['label'], example['mark']

        files = glob.glob(self.tfrecord_path + '/' + "*{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            shuffle(buffer_size=1000000, seed=2024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        data = []
        for x, uid, iid, y, m in ds:
            x = torch.from_numpy(x.numpy())
            uid = torch.from_numpy(uid.numpy())
            iid = torch.from_numpy(iid.numpy())
            y = torch.from_numpy(y.numpy())
            m = torch.from_numpy(m.numpy())
            data.append([x, uid, iid, y, m])
        return data

    def get_feat_dict(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['user_id'], example['item_id'], example['label'], example['mark']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        user_feat_dict = {}
        item_feat_dict = {}
        for x, uid, iid, y, m in ds:
            x = x.numpy()
            uid = np.squeeze(uid.numpy())
            iid = np.squeeze(iid.numpy())
            y = y.numpy()
            for i in range(len(x)):
                user_feat_dict[uid[i]] = x[i, :5]
                item_feat_dict[iid[i]] = x[i, 5:]

        return user_feat_dict, item_feat_dict


class KuaiRandLoader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 90
        self.tfrecord_path = tfrecord_path
        self.description = {
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
            "user_id": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "item_id": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "mark": tf.io.FixedLenFeature([self.SAMPLES], tf.float32)
        }

    # read val/test data
    def get_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['user_id'], example['item_id'], example['label'], example['mark']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x, uid, iid, y, m in ds:
            x = torch.from_numpy(x.numpy())
            uid = torch.from_numpy(uid.numpy())
            iid = torch.from_numpy(iid.numpy())
            y = torch.from_numpy(y.numpy())
            m = torch.from_numpy(m.numpy())
            yield x, uid, iid, y, m

    #  read train data
    def get_train_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['user_id'], example['item_id'], example['label'], example['mark']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        data = []
        for x, uid, iid, y, m in ds:
            x = torch.from_numpy(x.numpy())
            uid = torch.from_numpy(uid.numpy())
            iid = torch.from_numpy(iid.numpy())
            y = torch.from_numpy(y.numpy())
            m = torch.from_numpy(m.numpy())
            data.append([x, uid, iid, y, m])
        return data

    # read train and utrain data
    def get_comb_train_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['user_id'], example['item_id'], example['label'], example['mark']

        files = glob.glob(self.tfrecord_path + '/' + "*{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            shuffle(buffer_size=1000000, seed=2024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        data = []
        for x, uid, iid, y, m in ds:
            x = torch.from_numpy(x.numpy())
            uid = torch.from_numpy(uid.numpy())
            iid = torch.from_numpy(iid.numpy())
            y = torch.from_numpy(y.numpy())
            m = torch.from_numpy(m.numpy())
            data.append([x, uid, iid, y, m])
        return data

    def get_feat_dict(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['user_id'], example['item_id'], example['label'], example['mark']

        files = glob.glob(self.tfrecord_path + '/' + "{}*".format(data_type))
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        user_feat_dict = {}
        item_feat_dict = {}
        for x, uid, iid, y, m in ds:
            x = x.numpy()
            uid = np.squeeze(uid.numpy())
            iid = np.squeeze(iid.numpy())
            y = y.numpy()
            for i in range(len(x)):
                user_feat_dict[uid[i]] = x[i, :31]
                item_feat_dict[iid[i]] = x[i, 31:]
        return user_feat_dict, item_feat_dict
