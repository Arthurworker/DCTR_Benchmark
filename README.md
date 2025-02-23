### "Does Feature Matter: Towards Debiased Recommendations in Non-simplified Scenarios"

### 1. Environment

python  3.7+
pytorch 1.6+
numpy   1.21+
pandas  1.1.5+
optuna  3.6.1+
tensorflow 2.4.0+

### 2. DataPreprocess

#### 2.1 Datasets

You can download the raw data using these links as follow:

**Coat:** [Link](https://github.com/CrazyDumpling/CDR_CIKM2023/tree/main/data/coat)

**KuaiRand (Pure):** [Link](https://kuairand.com/)

The raw data and related directory structures that need to be prepared are organized as follows:

#### 2.2 Original Data Preprocess

We setup two strategies for the dataset preprocessing:

1. Biased Validation
2. Unbiased Validation

In order to facilitate the processing of the two different dataset partitioning strategies described above, we use the parameter "type" to set the preprocessing strategy of choice. "*tep2*" denotes the Biased Validation and "*tep3*" denotes the Unbiased Validation

You can prepare the coat data in the following code.

```
# Coat data preprocessing cmd
python3 datatransform/preprocess_coat.py 
python3 coat2tf.py --type tep2

# KuaiRand data preprocessing cmd
python3 datatransform/preprocess_kuairand.py
python3 kuairand2tf.py --type tep2
```

The statistics of the dataset are as follows:

| dataset  | users | items | train  | utrain | validation | test   | fileds | user_fileds | item_fileds | userid_filed_index | itemid_filed_index | features |
| -------- | ----- | ----- | ------ | ------ | ---------- | ------ | ------ | ----------- | ----------- | ------------------ | ------------------ | -------- |
| coat     | 290   | 300   | 6960   | 427    | 427        | 3420   | 10     | 5           | 5           | 0                  | 5                  | 637      |
| kuairand | 25877 | 6618  | 295497 | 118592 | 118592     | 948739 | 88     | 31          | 57          | 0                  | 31                 | 41300    |

The data directory structure used for model training is as follows:

```
data\
├─coat
│  ├─stats
│  │  ├─defaults.pkl
│  │  ├─feat_map.pkl
│  │  └─offset.pkl
│  ├─tfrecord
│  │  ├─test_0000.tfrecord
│  │  ├─train_0000.tfrecord
│  │  ├─utrain_0000.tfrecord
│  │  └─validation_0000.tfrecord
│  ├─tfrecord2
│  │  ├─test_0000.tfrecord
│  │  ├─train_0000.tfrecord
│  │  ├─utrain_0000.tfrecord
│  │  └─validation_0000.tfrecord
│  └─user_item_features
│  │  ├─item_features.ascii
│  │  ├─item_features_map.txt
│  │  ├─user_features.ascii
│  │  └─user_features_map.txt
│  ├─implicit_test_merge_feature.txt
│  ├─implicit_train_merge_feature.txt
│  ├─test.ascii
│  └─train.ascii
└─kuairand
    ├─stats_2
    │  ├─defaults.pkl
    │  ├─feat_map.pkl
    │  └─offset.pkl
    ├─threshold_2
    │  ├─test_0000.tfrecord
    │  ├─train_0000.tfrecord
    │  ├─utrain_0000.tfrecord
    │  ├─validation_0000.tfrecord
    ├─threshold2_2
    │  ├─test_0000.tfrecord
    │  ├─train_0000.tfrecord
    │  ├─utrain_0000.tfrecord
    │  ├─validation_0000.tfrecord
    ├─load_data_pure.py
    ├─log_random_4_22_to_5_08_pure.csv
    ├─log_standard_4_08_to_4_21_pure.csv
    ├─log_standard_4_22_to_5_08_pure.csv
    ├─r_data.csv
    ├─s_data.csv
    ├─user_features_pure.csv
    ├─video_features_basic_pure.csv
    └─video_features_statistic_pure.csv
```

#### 2.3 Add popularity features Data Preprocess

After add global user popularity and item popularity as two extra features. We categorize popularity into high, medium and low popularity features based on the frequency statistics of users and items, respectively.

The processing is consistent with the original processing.

The statistics of the new dataset are as follows:

| dataset  | users | items | train  | utrain | validation | test   | fileds | user_fileds | item_fileds | userid_filed_index | itemid_filed_index | features |
| -------- | ----- | ----- | ------ | ------ | ---------- | ------ | ------ | ----------- | ----------- | ------------------ | ------------------ | -------- |
| coat     | 290   | 300   | 6960   | 427    | 427        | 3420   | 10     | 6           | 6           | 0                  | 6                  | 643      |
| kuairand | 25877 | 6618  | 295497 | 118592 | 118592     | 948739 | 88     | 32          | 58          | 0                  | 32                 | 41306    |

### 3. Running Baselines & BIEE

#### 3.1 Running Base Models

```
# Bias base model
python3 tune_parameters.py -dataset 'coat' -model 'dcn' -y 'config/base_ctr.yml' -sr 'bias' -tb 'op_dcn_bias.csv' -cuda 0

# Combine base model
python3 tune_parameters.py -dataset 'coat' -model 'dcn' -y 'config/base_ctr.yml' -sr 'combine' -tb 'op_dcn_combine.csv' -cuda 0

```

#### 3.2 Running Baselines on new dataset

```
# Bias base model
python3 tune_parameters.py -dataset 'coat' -model 'dcn' -y 'config/base_ctr.yml' -sr 'bias' -tb 'op_dcn_bias.csv' -cuda 0

# Combine base model
python3 tune_parameters.py -dataset 'coat' -model 'dcn' -y 'config/base_ctr.yml' -sr 'combine' -tb 'op_dcn_combine.csv' -cuda 0

```

#### 3.3 Running BIEE

```
# BIEE + Bias
python3 tune_parameters.py -dataset 'coat' -model 'dcn' -y 'config/biee_ctr.yml' -sr 'combine' -tb 'op_dcn_combine.csv' -cuda 0

# BIEE + Combine
python3 tune_parameters.py -dataset 'coat' -model 'dcn' -y 'config/biee_ctr.yml' -sr 'combine' -tb 'op_dcn_combine.csv' -cuda 0
```
