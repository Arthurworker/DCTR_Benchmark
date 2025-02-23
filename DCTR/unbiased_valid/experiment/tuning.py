import os
import yaml
import time
from pathlib import Path
from utils.io import load_yaml


def hyper_parameter_tuning(model_opt, params, dataset, model, save_path, data_dir, cuda, source=None, seed=0):
    table_path = load_yaml('config/global.yml', key='path')['tables']
    trials, algorithm, best_params = None, None, None

    if not os.path.exists(table_path):
        os.makedirs(table_path)

    for algorithm in params['models']:
        trials, best_params = params['models'][algorithm](model_opt=model_opt,
                                                          dataset=dataset,
                                                          model=model,
                                                          data_dir=data_dir,
                                                          source=source,
                                                          seed=seed,
                                                          cuda=cuda,
                                                          searcher='optuna')
    dataset_path = dataset + '/' + model + '/'
    if not os.path.exists(table_path + save_path):
        if not os.path.exists(table_path + dataset_path):
            os.makedirs(table_path + dataset_path)

    trials.to_csv(table_path + save_path)

    if Path(table_path + dataset_path + 'op_hyper_params.yml').exists():
        pass
    else:
        if dataset == 'coat':
            yaml.dump(dict(coat=dict()), open(table_path + dataset_path + 'op_hyper_params.yml', 'w'),
                      default_flow_style=False)
        else:
            yaml.dump(dict(kuairand=dict()), open(table_path + dataset_path + 'op_hyper_params.yml', 'w'),
                      default_flow_style=False)
    time.sleep(0.5)
    hyper_params_dict = yaml.safe_load(open(table_path + dataset_path + 'op_hyper_params.yml', 'r'))

    if source is not None:
        algorithm = source + '_' + algorithm

    hyper_params_dict[dataset][algorithm] = best_params
    yaml.dump(hyper_params_dict, open(table_path + dataset_path + 'op_hyper_params.yml', 'w'), default_flow_style=False)
