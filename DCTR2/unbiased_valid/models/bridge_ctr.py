import torch
import random
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter
import pickle
import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial

import os
from sklearn import metrics
from utils import trainUtils
import warnings

warnings.filterwarnings("ignore")


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Objective:

    def __init__(self, model_opt, dataset, model, optimizer, data_dir, save_dir, seed, device, cuda,
                 epoch=500) -> None:
        """Initialize Class"""
        self.model_opt = model_opt
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.seed = seed
        self.device = device
        self.epoch = epoch
        self.cuda = cuda

        with open('./data/{}/user_feat_dict.pkl'.format(dataset), 'rb') as fi:
            self.user_feat_dict = pickle.load(fi)
        with open('./data/{}/item_feat_dict.pkl'.format(dataset), 'rb') as fi:
            self.item_feat_dict = pickle.load(fi)

        max_user_id = int(max(self.user_feat_dict.keys()))
        max_item_id = int(max(self.item_feat_dict.keys()))
        user_field_num = len(self.user_feat_dict[0])
        item_field_num = len(self.item_feat_dict[0])

        user_feat_list = []
        item_feat_list = []
        for i in range(max_user_id + 1):
            if i in self.user_feat_dict.keys():
                user_feat_list.append(self.user_feat_dict[i])
            else:
                user_feat_list.append(np.zeros(user_field_num, dtype=int))
        for i in range(max_item_id + 1):
            if i in self.item_feat_dict.keys():
                item_feat_list.append(self.item_feat_dict[i])
            else:
                item_feat_list.append(np.zeros(item_field_num, dtype=int))

        self.user_feat_tab = torch.tensor(user_feat_list).to(self.device)
        self.item_feat_tab = torch.tensor(item_feat_list).to(self.device)

        self.all_user_id = torch.tensor(list(self.user_feat_dict.keys())).to(self.device)
        self.all_item_id = torch.tensor(list(self.item_feat_dict.keys())).to(self.device)

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""

        # sample a set of hyperparameters.
        if self.dataset == 'coat':
            lr = trial.suggest_categorical('lr', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
            l2 = trial.suggest_categorical('l2', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
            batch_size = trial.suggest_categorical('bsize', [16, 32, 64])
            alpha1 = trial.suggest_categorical('alpha1', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        else:
            if self.model == 'fm':
                lr = trial.suggest_categorical('lr', [1e-3, 3e-4, 1e-4, 3e-5])
            else:
                lr = trial.suggest_categorical('lr', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
            l2 = trial.suggest_categorical('l2', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
            batch_size = trial.suggest_categorical('bsize', [256, 512, 1024])
            alpha1 = trial.suggest_categorical('alpha1', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])

        setup_seed(self.seed)

        model = BridgeCTR(self.model_opt, self.dataset, self.model, self.optimizer, self.data_dir, self.save_dir,
                          self.user_feat_tab, self.item_feat_tab, self.all_user_id, self.all_item_id,
                          bsize=np.int(batch_size), alpha1=alpha1, lr=lr, l2=l2, cuda=self.cuda).to(self.device)

        best_auc, best_logloss, best_te_auc, best_te_logloss = model.fit(self.epoch, early_stop_cnt=5)
        print('AUC: ', best_auc, 'LOGlOSS: ', best_logloss, 'test_AUC: ', best_te_auc,
              'test_LOGlOSS: ', best_te_logloss)

        return best_auc


class Tuner:
    """Class for tuning hyperparameter of CTR models."""

    def __init__(self):
        """Initialize Class."""

    @staticmethod
    def tune(n_trials, model_opt, dataset, model, optimizer, data_dir, save_dir, seed, device, epoch, cuda):
        """Hyperparameter Tuning by TPE."""
        objective = Objective(model_opt=model_opt, dataset=dataset, model=model, optimizer=optimizer, data_dir=data_dir,
                              save_dir=save_dir, seed=seed, epoch=epoch, device=device, cuda=cuda)
        study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.trials_dataframe(), study.best_params


class BridgeCTR(torch.nn.Module):
    def __init__(self, model_opt, dataset, model, optimizer, data_dir, save_dir, user_feat_tab, item_feat_tab,
                 all_user_id, all_item_id, bsize, alpha1=0.5, lr=1e-3, l2=1e-2, cuda=0):
        super(BridgeCTR, self).__init__()
        self.lr = lr
        self.l2 = l2
        self.bs = bsize
        self.alpha1 = alpha1
        self.dataset = dataset
        self.model = model
        self.model_dir = save_dir
        self.user_feat_tab = user_feat_tab
        self.item_feat_tab = item_feat_tab
        self.all_user_id = all_user_id
        self.all_item_id = all_item_id
        self.field_num = model_opt['field_num']
        self.dim = model_opt['latent_dim']
        self.dataloader = trainUtils.getDataLoader(dataset, data_dir)
        self.device = trainUtils.getDevice(cuda)
        self.network = trainUtils.getModel(model, model_opt).to(self.device)
        self.unif_network = trainUtils.getModel(model, model_opt, dataset, 'unif').to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.optim = trainUtils.getOptim(self.network, optimizer, self.lr, self.l2)
        self.logger = trainUtils.get_log(model)

        # load pretrain network parameters
        self.network.load_state_dict(
            torch.load('pretrain/unif_val/' + self.dataset + '_' + self.model + '_bias_featPlus2' + '.pt'))
        self.unif_network.load_state_dict(
            torch.load('pretrain/unif_val/' + self.dataset + '_' + self.model + '_unif_featPlus2' + '.pt'))

    def train_on_batch(self, label, data, sample_data):
        self.network.train()
        self.optim.zero_grad()

        data, label = data.to(self.device), label.to(self.device)
        sample_data = sample_data.to(self.device)

        logit = self.network(data)
        log_loss = self.criterion(logit, label)
        factual_loss = log_loss

        unif_label = torch.sigmoid(self.unif_network(sample_data))
        counter_factual_loss = self.criterion(self.network(sample_data), unif_label)

        loss = factual_loss + self.alpha1 * counter_factual_loss

        loss.backward()
        self.optim.step()

        return loss.item()

    def eval_on_batch(self, data):
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            logit = self.network(data)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob

    def fit(self, epochs, early_stop_cnt):
        early_stop_step = 0
        best_auc = 0.0
        best_logloss = 10000.0
        best_te_auc = 0.0
        best_te_logloss = 10000.0

        train_data = self.dataloader.get_train_data("train", batch_size=self.bs)
        valid_data = self.dataloader.get_train_data('val', batch_size=self.bs * 10)
        test_data = self.dataloader.get_train_data('test', batch_size=self.bs * 10)

        for _ in tqdm(range(int(epochs))):
            for feature, user_id, item_id, label, mark in train_data:

                # sample user and item
                all_user_len, all_item_len = len(self.all_user_id), len(self.all_item_id)
                min_len = min(all_user_len, all_item_len)
                if min_len >= len(label):
                    sample_user_index = torch.randint(all_user_len, (len(label),))
                    sample_item_index = torch.randint(all_item_len, (len(label),))
                else:
                    sample_user_index = torch.randint(all_user_len, (min_len,))
                    sample_item_index = torch.randint(all_item_len, (min_len,))

                sample_user_id = self.all_user_id[sample_user_index].long()
                sample_item_id = self.all_item_id[sample_item_index].long()

                # get sampled user's and item's feature
                user_feats = self.user_feat_tab[sample_user_id]
                item_feats = self.item_feat_tab[sample_item_id]

                sample_data = torch.cat((user_feats, item_feats), dim=1)
                # print(sample_data.shape)
                _ = self.train_on_batch(label, feature, sample_data)

            val_auc, val_loss = self.evaluate(valid_data)

            if val_auc > best_auc:
                best_auc = val_auc
                best_logloss = val_loss
                best_te_auc, best_te_logloss = self.evaluate(test_data)
                early_stop_step = 0

            else:
                early_stop_step += 1
                if early_stop_step == early_stop_cnt:
                    break

        return best_auc, best_logloss, best_te_auc, best_te_logloss

    def evaluate(self, valid_data: list):
        preds, trues = [], []
        for feature, user_id, item_id, label, mark in valid_data:
            pred = self.eval_on_batch(feature)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)

        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")

        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)

        return auc, loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def bridgectr(model_opt, dataset, model, data_dir, save_dir, seed=0, bsize=256, alpha1=0.5, lr=5e-4, l2=1e-5,
              n_trials=100, epoch=5000, cuda=0, searcher='grid', optimizer='Adam', **unused):
    progress = WorkSplitter()

    progress.section("Bridge-CTR (+2 feat): Set the random seed")
    setup_seed(seed)

    progress.section("Bridge-CTR (+2 feat): Training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if searcher == 'optuna':
        print('######### tuner ###########')
        tuner = Tuner()
        trials, best_params = tuner.tune(n_trials=n_trials, model_opt=model_opt, dataset=dataset, model=model,
                                         optimizer=optimizer, data_dir=data_dir, save_dir=save_dir, seed=seed,
                                         epoch=epoch, device=device, cuda=cuda)
        return trials, best_params

    if searcher == 'grid':
        print('######### trainer ###########')
        print('alpha1', alpha1, 'lr:', lr, 'l2:', l2, 'bs:', bsize)
        model = BridgeCTR(model_opt, dataset, model, optimizer, data_dir, save_dir, bsize, alpha1, lr, l2, cuda
                          ).to(device)

        best_auc, best_logloss, best_te_auc, best_te_logloss = model.fit(epoch, early_stop_cnt=5)
        print('AUC: ', best_auc, 'LOGlOSS: ', best_logloss, 'test_AUC: ', best_te_auc,
              'test_LOGlOSS: ', best_te_logloss)
