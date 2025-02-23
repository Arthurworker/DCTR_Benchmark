import torch
import random
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter
import pickle
import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial
import copy
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

        self.user_feat_id = {'max': 0, 'min': float('inf')}
        self.item_feat_id = {'max': 0, 'min': float('inf')}

        for k, v in self.user_feat_dict.items():
            if max(v) > self.user_feat_id['max']:
                self.user_feat_id['max'] = max(v)
            if min(v) < self.user_feat_id['min']:
                self.user_feat_id['min'] = min(v)
        for k, v in self.item_feat_dict.items():
            if max(v) > self.item_feat_id['max']:
                self.item_feat_id['max'] = max(v)
            if min(v) < self.item_feat_id['min']:
                self.item_feat_id['min'] = min(v)

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""

        # sample a set of hyperparameters.
        if self.dataset == 'coat':
            lr = trial.suggest_categorical('learning_rate', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
            l2 = trial.suggest_categorical('l2_norm', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            alpha1 = trial.suggest_categorical('alpha1', [1e-4, 1e-3, 1e-2, 1e-1, 1])
            # alpha1 = trial.suggest_categorical('alpha1', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        else:
            lr = trial.suggest_categorical('learning_rate', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
            l2 = trial.suggest_categorical('l2_norm', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
            batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
            alpha1 = trial.suggest_categorical('alpha1', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])

        setup_seed(self.seed)

        model = CausECTR(self.model_opt, self.dataset, self.model, self.optimizer, self.data_dir, self.save_dir,
                         self.user_feat_id, self.item_feat_id, bsize=np.int(batch_size), alpha1=alpha1, lr=lr, l2=l2,
                         cuda=self.cuda).to(self.device)

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


class CausECTR(torch.nn.Module):
    def __init__(self, model_opt, dataset, model, optimizer, data_dir, save_dir, user_feat_id, item_feat_id, bsize,
                 alpha1=0.5, lr=1e-3, l2=1e-2, cuda=0):
        super(CausECTR, self).__init__()
        self.lr = lr
        self.l2 = l2
        self.bs = bsize
        self.alpha1 = alpha1
        self.dataset = dataset
        self.model = model
        self.model_dir = save_dir
        self.user_feat_id = user_feat_id
        self.item_feat_id = item_feat_id
        self.feat_num = model_opt['feat_num']
        self.field_num = model_opt['field_num']
        self.dim = model_opt['latent_dim']
        self.model_opt = copy.deepcopy(model_opt)
        self.model_opt['feat_num'] = self.model_opt['feat_num'] + self.item_feat_id['max'] - self.item_feat_id['min'] + 1
        self.dataloader = trainUtils.getDataLoader(dataset, data_dir)
        self.device = trainUtils.getDevice(cuda)
        self.network = trainUtils.getModel(model, self.model_opt).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.optim = trainUtils.getOptim(self.network, optimizer, self.lr, self.l2)
        self.logger = trainUtils.get_log(model)

        if self.dataset == 'coat':
            self.item_field_ind = 5
        else:
            self.item_field_ind = 31

    def train_on_batch(self, label, data, data_train):
        self.network.train()
        self.optim.zero_grad()

        data, label, data_train = data.to(self.device), label.to(self.device), data_train.to(self.device)
        logit = self.network(data)
        log_loss = self.criterion(logit, label)

        item_feature_emb = self.network.getEmb(data_train[:, self.item_field_ind:])
        item_feature_emb2 = self.network.getEmb(data_train[:, self.item_field_ind:] + (
                self.feat_num - self.item_feat_id['min']))
        student_embedding = item_feature_emb
        teacher_embedding = torch.detach(item_feature_emb2)
        reg = torch.sum(torch.abs(student_embedding - teacher_embedding))
        loss = log_loss + self.alpha1 * reg

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

    def fit(self, epochs, early_stop_cnt, pretrain=True):
        early_stop_step = 0
        best_auc = 0.0
        best_logloss = 10000.0
        best_te_auc = 0.0
        best_te_logloss = 10000.0

        train_data = self.dataloader.get_train_data("train", batch_size=self.bs)
        utrain_data = self.dataloader.get_train_data("utrain", batch_size=self.bs)
        utrain_num = len(utrain_data)
        utrain_ind = 0

        valid_data = self.dataloader.get_train_data('val', batch_size=self.bs * 10)
        test_data = self.dataloader.get_train_data('test', batch_size=self.bs * 10)

        for _ in tqdm(range(int(epochs))):
            for feature, user_id, item_id, label, mark in train_data:
                if utrain_ind % utrain_num == 0:
                    utrain_ind = 0
                    random.shuffle(utrain_data)

                feature_unif, user_id_unif, item_id_unif, label_unif, mark_unif = utrain_data[utrain_ind]
                feature_unif,  label_unif = feature_unif.clone(), label_unif.clone()
                feature_unif[:, self.item_field_ind:] += (self.feat_num - self.item_feat_id['min'])

                feature_combine = torch.cat((feature, feature_unif), dim=0)
                label_combine = torch.cat((label, label_unif), dim=0)
                _ = self.train_on_batch(label_combine, feature_combine, feature)
                utrain_ind += 1

            val_auc, val_loss = self.evaluate(valid_data)

            if val_auc > best_auc:
                best_auc = val_auc
                best_logloss = val_loss
                best_te_auc, best_te_logloss = self.evaluate(test_data)
                early_stop_step = 0

                if pretrain:
                    if not os.path.exists(self.model_dir):
                        os.makedirs(self.model_dir)

                    torch.save(self.network.state_dict(), self.model_dir + self.dataset + '_' + self.model + '_cause' +
                                '.pt')

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


def causectr(model_opt, dataset, model, data_dir, save_dir, seed=0, bsize=256, alpha1=1, lr=5e-4, l2=1e-5,
             n_trials=100, epoch=500, cuda=0, searcher='grid', optimizer='Adam', **unused):
    progress = WorkSplitter()

    progress.section("CausE-CTR: Set the random seed")
    setup_seed(seed)

    progress.section("CausE-CTR: Training")
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
        model = CausECTR(model_opt, dataset, model, optimizer, data_dir, save_dir, bsize, alpha1, lr, l2, cuda
                         ).to(device)

        best_auc, best_logloss, best_te_auc, best_te_logloss = model.fit(epoch, early_stop_cnt=5)
        print('AUC: ', best_auc, 'LOGlOSS: ', best_logloss, 'test_AUC: ', best_te_auc,
              'test_LOGlOSS: ', best_te_logloss)
