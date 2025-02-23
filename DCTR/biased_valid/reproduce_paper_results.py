import argparse
import os
import yaml
import timeit
import pandas as pd

from utils.modelnames import models
from utils.io import load_yaml, save_dataframe_csv
from utils.argcheck import check_int_positive

import sys
from utils import trainUtils


def execute(model_opt, algorithm, dataset, model, data_dir, cuda, source=None, seed=0, bsize=256, lr=5e-4, l2=1e-5,
            alpha1=0.5, alpha2=0.5, beta1=0.5, beta2=0.5):

    _, _, best_te_auc, best_te_logloss = algorithm(model_opt=model_opt, dataset=dataset, model=model, data_dir=data_dir,
                                                   source=source, seed=seed, bsize=bsize, lr=lr, l2=l2, alpha1=alpha1,
                                                   alpha2=alpha2, beta1=beta1, beta2=beta2, cuda=cuda, searcher='grid',
                                                   with_test=True)

    return best_te_auc, best_te_logloss


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    sys.path.extend(["./modules", "./dataloader", "./utils"])

    if args.dataset.lower() == "coat":
        field_dim = trainUtils.get_stats("data/coat/stats")
        data_dir = "data/coat/tfrecord2"
        field = len(field_dim)
        feature = sum(field_dim)

    else:
        field_dim = trainUtils.get_stats("data/kuairand/stats_2")
        data_dir = "data/kuairand/threshold2_2"
        field = len(field_dim)
        feature = sum(field_dim)

    model_opt = {
        "latent_dim": args.dim, "feat_num": feature, "field_num": field, "mlp_dropout": args.mlp_dropout,
        "use_bn": args.mlp_bn, "mlp_dims": args.mlp_dims, "cross": args.cross
    }

    # read optimal hyperparameters
    parameters_dict = yaml.safe_load(
        open(table_path + args.dataset + '/' + args.model + '/op_hyper_params.yml', 'r'))[
        args.dataset]
    model_list = parameters_dict.keys()

    frame = []
    for model in model_list:
        init_row = {'model': model, 'bsize': 256, 'lr': 5e-4, 'l2': 1e-5, 'alpha1': 0.5, 'alpha2': 0.5, 'beta1': 0.5,
                    'beta2': 0.5, 'source': None}
        row = init_row.copy()

        df = pd.DataFrame(columns=['model', 'bsize', 'lr', 'l2', 'alpha1', 'alpha2', 'beta1', 'beta2', 'source'])

        if model.startswith('bias'):
            row.update(parameters_dict[model])
            row['source'] = 'bias'
            row['model'] = 'BASE-CTR'
        elif model.startswith('unif'):
            row.update(parameters_dict[model])
            row['source'] = 'unif'
            row['model'] = 'BASE-CTR'
        elif model.startswith('combine'):
            row.update(parameters_dict[model])
            row['source'] = 'combine'
            row['model'] = 'BASE-CTR'
        else:
            row.update(parameters_dict[model])

        opt = {"model_opt": model_opt, "dataset": args.dataset, "model": model, "data_dir": data_dir, "cuda": args.cuda,
               "source": row['source']}
        print(opt)

        start = timeit.default_timer()

        auc, logloss = execute(model_opt, models[row['model']], dataset=args.dataset, model=args.model,
                               data_dir=data_dir, cuda=args.cuda, source=row['source'], seed=args.seed,
                               bsize=row['bsize'], lr=row['lr'], l2=row['l2'], alpha1=row['alpha1'],
                               alpha2=row['alpha2'], beta1=row['beta1'], beta2=row['beta2'])

        stop = timeit.default_timer()
        print('Time: ', stop - start)

        result_dict = row
        result_dict['auc'], result_dict['logloss'] = auc, logloss
        df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)

        frame.append(df)

    results = pd.concat(frame)
    save_dataframe_csv(results, table_path, args.dataset+'/'+args.model+'/'+args.table_name)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Reproduce")
    # 'coat' or 'kuairand'
    parser.add_argument("-dataset", type=str, help="specify dataset", default="coat")
    # 'fm' or 'dcn' or 'dnn' or 'deepfm'
    parser.add_argument("-model", type=str, help="specify model", default="fm")

    parser.add_argument('-tb', dest='table_name', default="op_final_result.csv")

    # fixed hyperparameters
    parser.add_argument("-dim", type=int, help="embedding dimension", default=16)
    parser.add_argument("-mlp_dims", type=int, nargs='+', default=[1024, 512, 256], help="mlp layer size")
    parser.add_argument("-mlp_dropout", type=float, default=0.0, help="mlp dropout rate")
    parser.add_argument("-mlp_bn", action="store_true", default=False, help="mlp batch normalization")
    parser.add_argument("-cross", type=int, help="cross layer", default=3)

    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)

    # device information
    parser.add_argument("-cuda", type=int, choices=range(-1, 8), default=0, help="device info")

    args = parser.parse_args()

    main(args)


