import torch
import random
from tqdm import tqdm
from utils.progress import WorkSplitter
from scipy.sparse import csr_matrix
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

    def __init__(self, model_opt, dataset, pscore, model, optimizer, data_dir, save_dir, seed, device, cuda,
                 epoch=500) -> None:
        """Initialize Class"""
        self.model_opt = model_opt
        self.dataset = dataset
        self.pscore = pscore
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
            beta1 = trial.suggest_uniform('beta1', 0.01, 0.9)
        else:
            if self.model == 'fm':
                lr = trial.suggest_categorical('lr', [1e-3, 3e-4, 1e-4, 3e-5])
            else:
                lr = trial.suggest_categorical('lr', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
            l2 = trial.suggest_categorical('l2', [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
            batch_size = trial.suggest_categorical('bsize', [256, 512, 1024])
            beta1 = trial.suggest_uniform('beta1', 0.01, 0.9)

        setup_seed(self.seed)

        model = UDIPSCTR(self.model_opt, self.dataset, self.model, self.optimizer, self.data_dir, self.save_dir,
                         bsize=np.int(batch_size), beta1=beta1, lr=lr, l2=l2, cuda=self.cuda).to(self.device)

        best_auc, best_logloss, best_te_auc, best_te_logloss = model.fit(self.epoch, self.pscore, early_stop_cnt=5)
        print('AUC: ', best_auc, 'LOGlOSS: ', best_logloss, 'test_AUC: ', best_te_auc,
              'test_LOGlOSS: ', best_te_logloss)

        return best_auc


class Tuner:
    """Class for tuning hyperparameter of CTR models."""

    def __init__(self):
        """Initialize Class."""

    @staticmethod
    def tune(n_trials, model_opt, dataset, pscore, model, optimizer, data_dir, save_dir, seed, device, epoch, cuda):
        """Hyperparameter Tuning by TPE."""
        objective = Objective(model_opt=model_opt, dataset=dataset, pscore=pscore, model=model, optimizer=optimizer,
                              data_dir=data_dir, save_dir=save_dir, seed=seed, epoch=epoch, device=device, cuda=cuda)
        study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.trials_dataframe(), study.best_params


class UDIPSCTR(torch.nn.Module):
    def __init__(self, model_opt, dataset, model, optimizer, data_dir, save_dir, bsize,
                 beta1=0.5, lr=1e-3, l2=1e-2, cuda=0):
        super(UDIPSCTR, self).__init__()
        self.lr = lr
        self.l2 = l2
        self.bs = bsize
        self.beta1 = beta1
        self.dataset = dataset
        self.model = model
        self.model_dir = save_dir
        self.field_num = model_opt['field_num']
        self.dim = model_opt['latent_dim']
        self.dataloader = trainUtils.getDataLoader(dataset, data_dir)
        self.device = trainUtils.getDevice(cuda)
        self.network = trainUtils.getModel(model, model_opt).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.optim = trainUtils.getOptim(self.network, optimizer, self.lr, self.l2)
        self.logger = trainUtils.get_log(model)
        
    def train_on_batch(self, label, data, ips_weight):
        self.network.train()
        self.optim.zero_grad()

        data, label = data.to(self.device), label.to(self.device)
        logit = self.network(data)
        log_loss = self.criterion(logit, label)
        loss = torch.mean(ips_weight * log_loss)

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

    def fit(self, epochs, pscore, early_stop_cnt, pretrain=False):
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
                uid = user_id.long()
                is_popularity = torch.where(pscore[uid] > self.beta1,
                                            torch.ones_like(uid).to(self.device),
                                            torch.zeros_like(uid).to(self.device))
                ips_weight = is_popularity * (1 / pscore[uid]) + (1 - is_popularity) * torch.ones_like(uid).to(
                    self.device)

                _ = self.train_on_batch(label, feature, ips_weight)

            val_auc, val_loss = self.evaluate(valid_data)

            if val_auc > best_auc:
                best_auc = val_auc
                best_logloss = val_loss
                best_te_auc, best_te_logloss = self.evaluate(test_data)
                early_stop_step = 0

                if pretrain:
                    if not os.path.exists(self.model_dir):
                        os.makedirs(self.model_dir)

                    torch.save(self.network.state_dict(), self.model_dir + self.dataset + '_' + self.model + '_udips' +
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


def udipsctr(model_opt, dataset, model, data_dir, save_dir, seed=0, bsize=256, beta1=0.5, lr=5e-4, l2=1e-5,
             n_trials=100, epoch=500, cuda=0, searcher='grid', optimizer='Adam', **unused):
    progress = WorkSplitter()

    progress.section("UDIPS-CTR: Set the random seed")
    setup_seed(seed)

    progress.section("UDIPS-CTR: Estimate pscore")

    matrix_path = 'data/' + dataset + '/matrix_train.npy'

    matrix_train = np.load(matrix_path)

    temp_matrix_train = csr_matrix(matrix_train.shape)
    temp_matrix_train[matrix_train.nonzero()] = 1

    item_freq = np.sum(temp_matrix_train, axis=0).A1
    pscore_i = item_freq / np.max(item_freq)

    R = temp_matrix_train.toarray()
    numerator = np.sum(R * pscore_i, axis=1)
    denominator = np.sum(R, axis=1)
    pscore_u = numerator / denominator

    pscore_u[denominator == 0] = 1e-8

    progress.section("UDIPS-CTR: Training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pscore_u = torch.from_numpy(pscore_u).to(device)

    if searcher == 'optuna':
        print('######### tuner ###########')
        tuner = Tuner()
        trials, best_params = tuner.tune(n_trials=n_trials, model_opt=model_opt, dataset=dataset, pscore=pscore_u,
                                         model=model, optimizer=optimizer, data_dir=data_dir, save_dir=save_dir,
                                         seed=seed, epoch=epoch, device=device, cuda=cuda)
        return trials, best_params

    if searcher == 'grid':
        print('######### trainer ###########')
        print('beta1', beta1, 'lr:', lr, 'l2:', l2, 'bs:', bsize)

        model = UDIPSCTR(model_opt, dataset, model, optimizer, data_dir, save_dir, bsize, beta1, lr, l2, cuda
                         ).to(device)

        best_auc, best_logloss, best_te_auc, best_te_logloss = model.fit(epoch, pscore_u, early_stop_cnt=5)
        print('AUC: ', best_auc, ' LOGlOSS: ', best_logloss, 'test_AUC: ', best_te_auc,
              'test_LOGlOSS: ', best_te_logloss)
