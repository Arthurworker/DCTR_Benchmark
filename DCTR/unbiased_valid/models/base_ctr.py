import torch
import random
from tqdm import tqdm
from utils.progress import WorkSplitter

import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial

import os
import numpy as np
from sklearn import metrics
from utils import trainUtils
import warnings

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Objective:

    def __init__(self, model_opt, dataset, model, optimizer, data_dir, seed, device, cuda,
                 epoch=500, source=None) -> None:
        """Initialize Class"""
        self.model_opt = model_opt
        self.dataset = dataset
        self.model = model
        self.source = source
        self.optimizer = optimizer
        self.data_dir = data_dir
        self.seed = seed
        self.device = device
        self.epoch = epoch
        self.cuda = cuda

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""

        # sample a set of hyperparameters.
        if self.dataset == 'coat':
            lr = trial.suggest_categorical('lr', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
            l2 = trial.suggest_categorical('l2', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
            batch_size = trial.suggest_categorical('bsize', [16, 32, 64])
        else:
            if self.model == 'fm':
                lr = trial.suggest_categorical('lr', [1e-3, 3e-4, 1e-4, 3e-5])
            else:
                lr = trial.suggest_categorical('lr', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
            l2 = trial.suggest_categorical('l2', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
            batch_size = trial.suggest_categorical('bsize', [256, 512, 1024])

        setup_seed(self.seed)

        model = BASECTR(self.model_opt, self.dataset, self.model, self.optimizer, self.data_dir,
                        bsize=np.int(batch_size), lr=lr, l2=l2, cuda=self.cuda, source=self.source).to(self.device)

        best_auc, best_logloss, best_te_auc, best_te_logloss = model.fit(self.epoch, early_stop_cnt=5)
        print('AUC: ', best_auc, 'LOGlOSS: ', best_logloss, 'test_AUC: ', best_te_auc,
              'test_LOGlOSS: ', best_te_logloss)

        return best_auc


class Tuner:
    """Class for tuning hyperparameter of CTR models."""

    def __init__(self):
        """Initialize Class."""

    @staticmethod
    def tune(n_trials, model_opt, dataset, model, optimizer, data_dir, seed, device, epoch, cuda, source):
        """Hyperparameter Tuning by TPE."""
        objective = Objective(model_opt=model_opt, dataset=dataset, model=model, optimizer=optimizer, data_dir=data_dir,
                              seed=seed, epoch=epoch, device=device, cuda=cuda, source=source)
        study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.trials_dataframe(), study.best_params


class BASECTR(torch.nn.Module):
    def __init__(self, model_opt, dataset, model, optimizer, data_dir, bsize,
                 lr=1e-3, l2=1e-2, cuda=0, source=None):
        super(BASECTR, self).__init__()
        self.lr = lr
        self.l2 = l2
        self.bs = bsize
        self.source = source
        self.dataset = dataset
        self.model = model
        self.field_num = model_opt['field_num']
        self.dim = model_opt['latent_dim']
        self.dataloader = trainUtils.getDataLoader(dataset, data_dir)
        self.device = trainUtils.getDevice(cuda)
        self.network = trainUtils.getModel(model, model_opt, dataset, source).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.optim = trainUtils.getOptim(self.network, optimizer, self.lr, self.l2)
        self.logger = trainUtils.get_log(model)

    def train_on_batch(self, label, data):
        self.network.train()
        self.optim.zero_grad()

        data, label = data.to(self.device), label.to(self.device)
        logit = self.network(data)
        log_loss = self.criterion(logit, label)
        loss = log_loss

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

    def fit(self, epochs, early_stop_cnt, with_test=False, save=False):
        early_stop_step = 0
        best_auc = 0.0
        best_logloss = 10000.0
        best_te_auc = 0.0
        best_te_logloss = 10000.0

        if with_test:
            if self.source == 'unif':
                train_data = self.dataloader.get_train_data("utrain", batch_size=self.bs)
                valid_data = self.dataloader.get_train_data('val', batch_size=self.bs * 10)
                test_data = self.dataloader.get_train_data('test', batch_size=self.bs * 10)

            elif self.source == 'combine':
                train_data = self.dataloader.get_comb_train_data("train", batch_size=self.bs)
                valid_data = self.dataloader.get_train_data('val', batch_size=self.bs * 10)
                test_data = self.dataloader.get_train_data('test', batch_size=self.bs * 10)

            else:
                train_data = self.dataloader.get_train_data("train", batch_size=self.bs)
                valid_data = self.dataloader.get_train_data('val', batch_size=self.bs * 10)
                test_data = self.dataloader.get_train_data('test', batch_size=self.bs * 10)

            for _ in tqdm(range(int(epochs))):
                for feature, user_id, item_id, label, mark in train_data:
                    _ = self.train_on_batch(label, feature)

                val_auc, val_loss = self.evaluate(valid_data)

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_logloss = val_loss
                    best_te_auc, best_te_logloss = self.evaluate(test_data)
                    early_stop_step = 0

                    if save:
                        model_dir = 'save/'
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)

                        torch.save(self.network.state_dict(), model_dir + self.dataset + '_' + self.model + '_' +
                                   self.source + '.pt')

                else:
                    early_stop_step += 1
                    if early_stop_step == early_stop_cnt:
                        break
        else:
            if self.source == 'unif':
                train_data = self.dataloader.get_train_data("utrain", batch_size=self.bs)
                valid_data = self.dataloader.get_train_data('val', batch_size=self.bs * 10)

            elif self.source == 'combine':
                train_data = self.dataloader.get_comb_train_data("train", batch_size=self.bs)
                valid_data = self.dataloader.get_train_data('val', batch_size=self.bs * 10)

            else:
                train_data = self.dataloader.get_train_data("train", batch_size=self.bs)
                valid_data = self.dataloader.get_train_data('val', batch_size=self.bs * 10)

            for _ in tqdm(range(int(epochs))):
                for feature, user_id, item_id, label, mark in train_data:
                    _ = self.train_on_batch(label, feature)

                val_auc, val_loss = self.evaluate(valid_data)

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_logloss = val_loss
                    early_stop_step = 0

                    if save:
                        model_dir = 'pretrain/'
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)

                        torch.save(self.network.state_dict(), model_dir + self.dataset + '_' + self.model + '_' +
                                   self.source + '.pt')

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


def basectr(model_opt, dataset, model, data_dir, source=None, seed=0, bsize=256, lr=5e-4, l2=1e-5, n_trials=50,
            epoch=5000, cuda=0, searcher='grid', optimizer='Adam', with_test=False, save=False, **unused):
    progress = WorkSplitter()

    progress.section("BASE-CTR: Set the random seed")
    setup_seed(seed)

    progress.section("BASE-CTR: Training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if searcher == 'optuna':
        print('######### tuner ###########')
        tuner = Tuner()
        trials, best_params = tuner.tune(n_trials=n_trials, model_opt=model_opt, dataset=dataset, model=model,
                                         optimizer=optimizer, data_dir=data_dir, seed=seed, epoch=epoch, device=device,
                                         cuda=cuda, source=source)
        return trials, best_params

    if searcher == 'grid':
        print('######### trainer ###########')
        print('lr:', lr, 'l2:', l2, 'bs:', bsize)
        model = BASECTR(model_opt, dataset, model, optimizer, data_dir, bsize, lr, l2, cuda, source).to(device)

        best_auc, best_logloss, best_te_auc, best_te_logloss = model.fit(epoch, early_stop_cnt=5, with_test=with_test,
                                                                         save=save)
        print('AUC: ', best_auc, 'LOGlOSS: ', best_logloss, 'test_AUC: ', best_te_auc,
              'test_LOGlOSS: ', best_te_logloss)

        return best_auc, best_logloss, best_te_auc, best_te_logloss

