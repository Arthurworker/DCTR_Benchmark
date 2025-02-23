import torch
import random
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter
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

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""

        # sample a set of hyperparameters.
        if self.dataset == 'coat':
            lr = trial.suggest_categorical('lr', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
            l2 = trial.suggest_categorical('l2', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
            batch_size = trial.suggest_categorical('bsize', [16, 32, 64])
            alpha1 = trial.suggest_uniform('alpha1', 0.0001, 0.1)
            alpha2 = trial.suggest_categorical('alpha2', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        else:
            if self.model == 'fm':
                lr = trial.suggest_categorical('lr', [1e-3, 3e-4, 3e-5])
            else:
                lr = trial.suggest_categorical('lr', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
            l2 = trial.suggest_categorical('l2', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
            batch_size = trial.suggest_categorical('bsize', [256, 512, 1024])
            alpha1 = trial.suggest_uniform('alpha1', 0.0001, 0.1)
            alpha2 = trial.suggest_categorical('alpha2', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        setup_seed(self.seed)

        model = TJRCTR(self.model_opt, self.dataset, self.model, self.optimizer, self.data_dir, self.save_dir,
                       bsize=np.int(batch_size), alpha1=alpha1, alpha2=alpha2, lr=lr, l2=l2, cuda=self.cuda
                       ).to(self.device)

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
        objective = Objective(model_opt=model_opt, dataset=dataset, model=model, optimizer=optimizer,
                              data_dir=data_dir, save_dir=save_dir, seed=seed, epoch=epoch, device=device, cuda=cuda)
        study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.trials_dataframe(), study.best_params


class TJRCTR(torch.nn.Module):
    def __init__(self, model_opt, dataset, model, optimizer, data_dir, save_dir, bsize,
                 alpha1=0.5, alpha2=0.5, lr=1e-3, l2=1e-2, cuda=0):
        super(TJRCTR, self).__init__()
        self.lr = lr
        self.l2 = l2
        self.bs = bsize
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.dataset = dataset
        self.model = model
        self.model_dir = save_dir
        self.field_num = model_opt['field_num']
        self.dim = model_opt['latent_dim']
        self.dataloader = trainUtils.getDataLoader(dataset, data_dir)
        self.device = trainUtils.getDevice(cuda)
        self.network = trainUtils.getModel_TJR2(model, model_opt).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.optim = trainUtils.getOptim(self.network, optimizer, self.lr, self.l2)
        self.logger = trainUtils.get_log(model)

    def train_on_batch(self, label, data, mark):
        self.network.train()
        self.optim.zero_grad()

        data, label, mark = data.to(self.device), label.to(self.device), mark.to(self.device)

        norm_out = self.network.forward(data)
        norm_loss = self.alpha1 * torch.mean(mark * self.criterion(norm_out, label))

        unif_out = self.network.forward(data) - self.network.res_forward(data)
        res_loss = self.alpha2 * torch.mean(mark * self.criterion(
            unif_out, label)) + torch.mean((1 - mark) * self.criterion(unif_out, label))

        loss = norm_loss + res_loss

        loss.backward()
        self.optim.step()

        return loss.item()

    def eval_on_batch(self, data):
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            logit = self.network.predict(data)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob

    def fit(self, epochs, early_stop_cnt):
        early_stop_step = 0
        best_auc = 0.0
        best_logloss = 10000.0
        best_te_auc = 0.0
        best_te_logloss = 10000.0

        train_data = self.dataloader.get_comb_train_data("train", batch_size=self.bs)
        valid_data = self.dataloader.get_train_data('val', batch_size=self.bs * 10)
        test_data = self.dataloader.get_train_data('test', batch_size=self.bs * 10)

        for _ in tqdm(range(int(epochs))):
            for feature, user_id, item_id, label, mark in train_data:

                _ = self.train_on_batch(label, feature, mark)

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


def tjrctr(model_opt, dataset, model, data_dir, save_dir, seed=0, bsize=256, alpha1=0.5, alpha2=0.5, lr=5e-4, l2=1e-5,
           n_trials=150, epoch=5000, cuda=0, searcher='grid', optimizer='Adam', **unused):
    progress = WorkSplitter()

    progress.section("TJR-CTR (+2 feat): Set the random seed")
    setup_seed(seed)

    progress.section("TJR-CTR (+2 feat): Training")
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
        print('alpha1', alpha1, 'alpha2', alpha2, 'lr:', lr, 'l2:', l2, 'bs:', bsize)
        model = TJRCTR(model_opt, dataset, model, optimizer, data_dir, save_dir, bsize, alpha1, alpha2, lr, l2, cuda
                       ).to(device)

        best_auc, best_logloss, best_te_auc, best_te_logloss = model.fit(epoch, early_stop_cnt=5)
        print('AUC: ', best_auc, 'LOGlOSS: ', best_logloss, 'test_AUC: ', best_te_auc,
              'test_LOGlOSS: ', best_te_logloss)
