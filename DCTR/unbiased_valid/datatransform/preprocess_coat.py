import numpy as np
import os
import pandas as pd


def get_origin_data(path):
    with open(path, "r") as f:
        origin_data = []
        for line in f.readlines():
            origin_data.append(line.split())

        origin_data = np.array(origin_data).astype(int)

    return origin_data


def proess_features(features, features_map):
    num, field_num = features.shape[0], len(features_map)
    new_features = np.zeros((num, field_num), dtype=int)

    for ind in range(num):
        col_ind = 0
        field_ind = 0
        for key, values in features_map.items():
            for i in range(len(values)):
                if features[ind][col_ind + i] == 1:
                    new_features[ind][field_ind] = i

            col_ind += len(values)
            field_ind += 1

    return new_features


def process_save_data1(x_data, save_file_path):
    """

    :param x_data:
    :param save_file_path:
    :return: (u, i, rating)
    """
    # 获取评分不为0的位置
    nonzero_positions = np.where(x_data > 0)
    # 提取非零元素的行索引、列索引和对应的评分值
    user_ids, item_ids, values = nonzero_positions[0], nonzero_positions[1], x_data[nonzero_positions]

    # 将三元组保存为 txt 文件
    rating_num = 0
    with open(save_file_path, "w") as save_file:
        for user_id, item_id, value in zip(user_ids, item_ids, values):
            rating_num += 1
            save_file.write(f"{user_id} {item_id} {value}\n")

    print("rating_num:",  rating_num)


def process_save_data2(x_data, save_file_path, threshold = 4):
    """

    :param x_data:
    :param save_file_path:
    :return: (u, i, label)
    """
    # 获取评分不为0的位置
    nonzero_positions = np.where(x_data > 0)
    # 提取非零元素的行索引、列索引和对应的评分值
    user_ids, item_ids, values = nonzero_positions[0], nonzero_positions[1], x_data[nonzero_positions]

    # 将三元组保存为 txt 文件 隐式反馈
    rating_num = 0
    with open(save_file_path, "w") as save_file:
        for user_id, item_id, value in zip(user_ids, item_ids, values):
            rating_num += 1
            if value >= threshold:
                value = 1
            else:
                value = 0
            save_file.write(f"{user_id} {item_id} {value}\n")

    print("rating_num:", rating_num)


def process_save_data3(x_data, save_file_path, new_user_features, new_item_features, threshold=4):
    nonzero_positions = np.where(x_data > 0)
    user_ids, item_ids, values = nonzero_positions[0], nonzero_positions[1], x_data[nonzero_positions]

    # 将user_id, user_feature, item_id, item_feature, rating保存为 txt 文件 隐式反馈
    rating_num = 0
    with open(save_file_path, "w") as save_file:
        for user_id, item_id, value in zip(user_ids, item_ids, values):
            rating_num += 1
            user_feat = new_user_features[user_id]
            item_feat = new_item_features[item_id]
            if value >= threshold:
                value = 1
            else:
                value = 0
            save_file.write(
                f"{user_id} {user_feat[0]} {user_feat[1]} {user_feat[2]} {user_feat[3]} {item_id} {item_feat[0]} {item_feat[1]} {item_feat[2]} {item_feat[3]} {user_id} {item_id} {value}\n")

    print("rating_num:", rating_num)


def check_conflicts_between_sets(user_path, random_path, uid_ind, iid_ind, remove_type='log'):
    df_user = pd.read_csv(user_path, header=None, delimiter=',')
    df_rand = pd.read_csv(random_path, header=None, delimiter=',')

    if remove_type == 'log':
        df_all = df_user.merge(df_rand.drop_duplicates(), on=[uid_ind, iid_ind], how='left', indicator=True)
        _df_user = df_user[df_all['_merge'] == 'left_only']
        _df_user = _df_user.iloc[:, :]
        if _df_user.shape[0] - df_user.shape[0]:
            print('There is a conflict: the number of redundancies is {0} and new data is saved'.
                  format(_df_user.shape[0] - df_user.shape[0]))
            _df_user.to_csv(user_path, header=None, index=None)
    else:
        df_all = df_rand.merge(df_user.drop_duplicates(), on=[uid_ind, iid_ind], how='left', indicator=True)
        _df_rand = df_rand[df_all['_merge'] == 'left_only']
        _df_rand = _df_rand.iloc[:, :]
        if _df_rand.shape[0] - df_rand.shape[0]:
            print('There is a conflict: the number of redundancies is {0} and new data is saved'.
                  format(_df_rand.shape[0] - df_rand.shape[0]))
            _df_rand.to_csv(random_path, header=None, index=None)


if __name__ == '__main__':
    data_dir = "./data"
    name1 = 'coat'
    data_set_dir = os.path.join(data_dir, name1)
    train_file = os.path.join(data_set_dir, "train.ascii")
    test_file = os.path.join(data_set_dir, "test.ascii")

    name2 = 'user_item_features'
    features_set_dir = os.path.join(data_set_dir, name2)
    user_features_file = os.path.join(features_set_dir, "user_features.ascii")
    item_features_file = os.path.join(features_set_dir, "item_features.ascii")

    # path: 'data/coat/user_item_features/user_features_map.txt'
    user_features_map = {
        'gender': ['men', 'women'],
        'age': ['20-30', '30-40', '40-50', '50-60', 'over 60', 'under 20'],
        'location': ['rural', 'suburban', 'urban'],
        'fashioninterest': ['moderately', 'not at all', 'very']
    }

    # path: 'data/coat/user_item_features/item_features_map.txt'
    item_features_map = {
        'gender': ['men', 'women'],
        'jackettype': ['bomber', 'cropped', 'field', 'fleece', 'insulated', 'motorcycle', 'other', 'packable',
                       'parkas', 'pea', 'rain', 'shells', 'track', 'trench', 'vests', 'waterproof'],
        'color': ['beige', 'black', 'blue', 'brown', 'gray', 'green', 'multi', 'navy', 'olive', 'other', 'pink',
                  'purple', 'red'],
        'onfrontpage': ['yes', 'no']
    }

    x_train = get_origin_data(train_file)
    x_test = get_origin_data(test_file)

    user_features = get_origin_data(user_features_file)
    item_features = get_origin_data(item_features_file)

    print("===>Load from {} data set<===".format(name1))
    print("[train] rating ratio: {:.6f}".format((x_train > 0).sum() / (x_train.shape[0] * x_train.shape[1])))
    print("[test]  rating ratio: {:.6f}".format((x_test > 0).sum() / (x_test.shape[0] * x_test.shape[1])))
    print("train_data_num:", (x_train > 0).sum())
    print(x_train.shape)

    new_user_features = proess_features(user_features, user_features_map)
    print(new_user_features.shape)

    new_item_features = proess_features(item_features, item_features_map)
    print(new_item_features.shape)

    save_file_path = os.path.join(data_set_dir, "implicit_train_merge_feature.txt")
    process_save_data3(x_train, save_file_path, new_user_features, new_item_features)

    save_file_path = os.path.join(data_set_dir, "implicit_test_merge_feature.txt")
    process_save_data3(x_test, save_file_path, new_user_features, new_item_features)

    traindata_file_path = os.path.join(data_set_dir, "implicit_train_merge_feature.txt")
    traindata = pd.read_csv(traindata_file_path, delimiter=' ', header=None)

    traindata_csv_path = os.path.join(data_set_dir, "traindata.csv")
    traindata.to_csv(traindata_csv_path, index=False, header=None)

    testdata_file_path = os.path.join(data_set_dir, "implicit_test_merge_feature.txt")
    testdata = pd.read_csv(testdata_file_path, delimiter=' ', header=None)

    testdata_csv_path = os.path.join(data_set_dir, "testdata.csv")
    testdata.to_csv(testdata_csv_path, index=False, header=None)

    label_ind = traindata.shape[1]-1

    check_conflicts_between_sets(traindata_csv_path, testdata_csv_path, uid_ind=label_ind-2, iid_ind=label_ind-1)


