import argparse
import os

from utils.modelnames import models
from utils.io import load_yaml
from utils.argcheck import check_int_positive
from experiment.tuning import hyper_parameter_tuning

import sys
from utils import trainUtils


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    sys.path.extend(["./modules", "./dataloader", "./utils"])

    if args.dataset.lower() == "coat":
        field_dim = trainUtils.get_stats("data/coat/stats")
        data_dir = "data/coat/tfrecord"
        field = len(field_dim)
        feature = sum(field_dim)

    else:
        field_dim = trainUtils.get_stats("data/kuairand/stats_2")
        data_dir = "data/kuairand/threshold_2"
        field = len(field_dim)
        feature = sum(field_dim)

    model_opt = {
        "latent_dim": args.dim, "feat_num": feature, "field_num": field, "mlp_dropout": args.mlp_dropout,
        "use_bn": args.mlp_bn, "mlp_dims": args.mlp_dims, "cross": args.cross
    }

    opt = {"model_opt": model_opt, "dataset": args.dataset, "model": args.model, "data_dir": data_dir,
           "cuda": args.cuda, "source": args.source}
    print(opt)

    params = load_yaml(args.grid)
    params['models'] = {params['models']: models[params['models']]}

    hyper_parameter_tuning(model_opt, params, dataset=args.dataset, model=args.model,
                           save_path=args.dataset+'/'+args.model+'/'+args.table_name, data_dir=data_dir, cuda=args.cuda,
                           source=args.source, seed=args.seed)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="CTR ParameterTuning")
    # 'coat' or 'kuairand'
    parser.add_argument("-dataset", type=str, help="specify dataset", default="coat")
    # 'fm' or 'dcn' or 'dnn' or 'deepfm'
    parser.add_argument("-model", type=str, help="specify model", default="fm")

    parser.add_argument('-y', dest='grid', default='config/drjl_ctr.yml')
    parser.add_argument('-tb', dest='table_name', default="op_fm_bias.csv")
    parser.add_argument('-sr', dest='source', default=None)  # 'unif' or 'comb' or 'bias' or None

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
