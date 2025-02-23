import pandas as pd


def get_origin_label(data_path, file_name):
    origin_data = pd.read_csv(data_path + file_name, sep=",")
    print('finish read csv')
    keep_column = ['user_id', 'video_id', 'is_click']
    origin_data = origin_data[keep_column]
    target_column = ['is_click']
    for col in target_column:
        origin_data[col] = origin_data[col].astype(float)

    return origin_data, target_column


def process_user_features(data_path, file_name):
    user_features = pd.read_csv(data_path + file_name)
    user_features = user_features.fillna(0)
    user_features['user_id_feature'] = user_features['user_id'].apply(lambda x: x)
    user_features = user_features.rename(columns={'follow_user_num': 'follow_user_num_uf'})

    key_features = ['user_id']
    numeric_features = 'follow_user_num_uf,fans_user_num,friend_user_num,register_days'.split(',')
    cate_features = 'user_id_feature,user_active_degree,is_lowactive_period,is_live_streamer,is_video_author,' \
                    'follow_user_num_range,fans_user_num_range,friend_user_num_range,register_days_range'.split(',')
    onehot_features = ['onehot_feat' + str(i) for i in range(18)]
    cate_features = cate_features + onehot_features
    # print(numeric_features)
    # print(cate_features)
    all_columns = key_features + numeric_features + cate_features
    user_features = user_features[all_columns]

    return user_features, numeric_features, cate_features, key_features


def process_item_features(data_path, basic_file_name, sta_file_name):
    item_basic_features = pd.read_csv(data_path + basic_file_name)
    item_statistics_features = pd.read_csv(data_path + sta_file_name)
    item_features = pd.merge(item_basic_features, item_statistics_features, on='video_id', how='outer')
    item_features = item_features.fillna(0)
    item_features['video_id_feature'] = item_features['video_id'].apply(lambda x: x)

    key_features = ['video_id']
    cate_features = 'video_id_feature,video_type,upload_type,music_type'.split(',')
    basic_numeric_features = 'video_duration,server_width,server_height'.split(',')
    statistics_numeric_features = [x for x in item_statistics_features.columns if x not in ['video_id', 'counts']]
    numeric_features = basic_numeric_features + statistics_numeric_features

    all_columns = key_features + numeric_features + cate_features
    item_features = item_features[all_columns]

    return item_features, numeric_features, cate_features, key_features


def origin_data_process(data_path, file_name):
    origin_label, target_column = get_origin_label(data_path, file_name)
    print(origin_label.head(5), origin_label.shape)
    user_features, uf_numeric_features, uf_cate_features, user_id = process_user_features(data_path,
                                                                                          "user_features_pure.csv")
    item_features, if_numeric_features, if_cate_features, video_id = process_item_features(data_path,
                                                                                           "video_features_basic_pure.csv",
                                                                                           "video_features_statistic_pure.csv")
    sample_data = pd.merge(origin_label, user_features, on='user_id', how='left')
    sample_data = pd.merge(sample_data, item_features, on='video_id', how='left')

    user_features_columns = uf_cate_features + uf_numeric_features
    item_features_columns = if_cate_features + if_numeric_features
    print('uf_cate_features num:', len(uf_cate_features), 'uf_numeric_features num:', len(uf_numeric_features))
    print('if_cate_features num:', len(if_cate_features), 'if_numeric_features num:', len(if_numeric_features))
    save_columns = user_features_columns + item_features_columns + user_id + video_id + target_column
    result_data = sample_data[save_columns]
    result_data = result_data.fillna(0)
    # result_data = result_data.sample(frac=1)

    return result_data, target_column, user_features_columns, item_features_columns


def check_conflicts_between_sets(user_path, random_path, uid_ind, iid_ind, remove_type='log'):
    df_user = pd.read_csv(user_path, header=None, delimiter=',')
    df_rand = pd.read_csv(random_path, header=None, delimiter=',')

    if remove_type == 'log':
        df_user.drop_duplicates(inplace=True, subset=[uid_ind, iid_ind], ignore_index=True)
        df_rand.drop_duplicates(inplace=True, subset=[uid_ind, iid_ind], ignore_index=True)

        df_all = df_user.merge(df_rand, on=[uid_ind, iid_ind], how='left', indicator=True)
        _df_user = df_user[df_all['_merge'] == 'left_only']
        _df_user = _df_user.iloc[:, :]
        if _df_user.shape[0] - df_user.shape[0]:
            print('There is a conflict: the number of redundancies is {0} and new data is saved'.
                  format(_df_user.shape[0] - df_user.shape[0]))
            _df_user.to_csv(user_path, header=None, index=None)
            df_rand.to_csv(random_path, header=None, index=None)
    else:
        df_user.drop_duplicates(inplace=True, subset=[uid_ind, iid_ind], ignore_index=True)
        df_rand.drop_duplicates(inplace=True, subset=[uid_ind, iid_ind], ignore_index=True)

        df_all = df_rand.merge(df_user, on=[uid_ind, iid_ind],  how='left', indicator=True)
        _df_rand = df_rand[df_all['_merge'] == 'left_only']
        _df_rand = _df_rand.iloc[:, :]
        if _df_rand.shape[0] - df_rand.shape[0]:
            print('There is a conflict: the number of redundancies is {0} and new data is saved'.
                  format(_df_rand.shape[0] - df_rand.shape[0]))
            _df_rand.to_csv(random_path, header=None, index=None)
            df_user.to_csv(user_path, header=None, index=None)


if __name__ == '__main__':
    data_path = './data/kuairand/'
    save_path = './data/kuairand/'
    train_type = '2'

    if train_type == '1':
        file_name = 'log_standard_4_08_to_4_21_pure.csv'
    else:
        file_name = 'log_standard_4_22_to_5_08_pure.csv'
    result_data, target_column, user_features_columns, item_features_columns = origin_data_process(data_path, file_name)
    print(result_data.head(5), result_data.shape)
    print('user_field_num:', len(user_features_columns))
    print('item_field_num:', len(item_features_columns))
    print('field_num:', len(target_column + user_features_columns + item_features_columns))
    print('column_num:', len(target_column + user_features_columns + item_features_columns) + 2)
    result_data.to_csv(save_path+'s_data.csv', header=None, index=False)

    file_name = 'log_random_4_22_to_5_08_pure.csv'
    result_data, target_column, user_features_columns, item_features_columns = origin_data_process(data_path, file_name)
    print(result_data.head(5), result_data.shape)
    print('user_field_num:', len(user_features_columns))
    print('item_field_num:', len(item_features_columns))
    print('field_num:', len(target_column + user_features_columns + item_features_columns))
    print('column_num:', len(target_column + user_features_columns + item_features_columns) + 2)
    result_data.to_csv(save_path + 'r_data.csv', header=None, index=False)

    s_data = pd.read_csv(save_path + 's_data.csv', header=None, delimiter=',')
    r_data = pd.read_csv(save_path + 'r_data.csv', header=None, delimiter=',')
    label_ind = s_data.shape[1] - 1
    check_conflicts_between_sets(save_path + 's_data.csv', save_path + 'r_data.csv', uid_ind=label_ind-2,
                                 iid_ind=label_ind-1)







