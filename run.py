# -*- coding:utf-8 -*-
import copy
import os
import argparse
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from benedict import benedict
from mytool import tool
from mytool import mytorch as mtorch
from mytool import callback as mcallback
from mytool import metric as mmetric
from mytool import plot as mplot
from datasets.dataset import load_mat
from typing import *


def bind_boolind_for_fn(func, train_bool_ind, val_bool_ind):
    def binded_func(scores, labels):
        if scores.requires_grad == True:
            return func(scores[train_bool_ind], labels[train_bool_ind].long())
        else:
            return func(scores[val_bool_ind], labels[val_bool_ind].long())

    tool.set_func_name(binded_func, tool.get_func_name(func))
    return binded_func


def train_one_args(args, data=None):
    # load data
    if data is not None:
        adjs, inputs, labels, train_bool, val_bool, n_view, n_node, n_feats, n_class = data
    else:
        adjs, inputs, labels, train_bool, val_bool, n_view, n_node, n_feats, n_class = load_mat(**args['dataset_args'])
    dataload = [((inputs, adjs), labels), ]

    # build model and init callback_list
    device = args['device']
    if device == 'tpu':
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    ModelClass = tool.import_class(**args['model_class_args'])
    model = ModelClass(n_view=n_view, n_feats=n_feats, n_class=n_class, **args['model_args'])
    wrapmodel = mtorch.WrapModel(model).to(device)
    callback_list: List[mcallback.Callback] = [mcallback.PruningCallback(args['trial'], args['tuner_monitor'])] if args[
        'tuner_flag'] else []

    # loss optimizer lr_scheduler
    loss_fn = nn.CrossEntropyLoss()
    OptimizerClass = tool.import_class(**args['optimizer_class_args'])
    optimizer = OptimizerClass(wrapmodel.parameters(), **args['optimizer_args'])
    SchedulerClass = tool.import_class(**args['scheduler_class_args'])
    scheduler = SchedulerClass(optimizer, **args['scheduler_args'])

    # warp scheduler
    def sche_func(epoch, lr, epoch_logs):
        scheduler.step(epoch_logs[args['scheduler_monitor']])

    scheduler_callback = mcallback.SchedulerWrapCallback(sche_func, True)
    callback_list.append(scheduler_callback)

    # training
    wrapmodel.compile(
        loss=bind_boolind_for_fn(loss_fn, train_bool, val_bool),
        optimizer=optimizer,
        metric=[
            bind_boolind_for_fn(mmetric.acc, train_bool, val_bool),
            bind_boolind_for_fn(mmetric.f1, train_bool, val_bool),
            bind_boolind_for_fn(mmetric.precision, train_bool, val_bool),
            bind_boolind_for_fn(mmetric.recall, train_bool, val_bool),
        ]
    )

    # add callbacks
    callback_list.extend([
        mcallback.EarlyStoppingCallback(quiet=args['quiet'], **args['earlystop_args']),
        mcallback.TunerRemovePreFileInDir([
            args['earlystop_args']['checkpoint_dir'],
        ], 10, 0.8),
    ])

    # fit
    history = wrapmodel.fit(
        dataload=dataload,
        epochs=args['epochs'],
        device=device,
        val_dataload=dataload,
        callbacks=callback_list,
        quiet=args['quiet']
    )

    return history.history


def train_times(args, num_times, data=None):
    """
    train model with args for num_times
    :param args: Dict
    :param num_times: int
    :return:
        repeat_df_list: List[pd.DataFrame] record logs for each time
    """
    # set seed
    if data is None:
        if 'seed' not in args.keys():
            args['seed'] = 42
        random.seed(args['seed'])
        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])

        # load data
        data = load_mat(**args['dataset_args'])

    repeat_df_list = []
    for idx in range(num_times):
        logs = train_one_args(args, data)
        repeat_df_list.append(pd.DataFrame(logs))

    return repeat_df_list


def mean_std_with_df_list(df_list, field_check_str, direction='max'):
    """
    compute the mean and std of the field in df_list
    :param df_list:
    :param field_check_str:
    :param direction:
    :return:
        mean_: Dict[
    """
    metric_history = mtorch.History()
    for df in df_list:
        df_col_names = df.columns
        if direction == 'max':
            metric_dict = df.iloc[:, df_col_names.str.contains(field_check_str)].max(axis=0).to_dict()
        else:
            metric_dict = df.iloc[:, df_col_names.str.contains(field_check_str)].min(axis=0).to_dict()
        metric_history.update(metric_dict)

    mean_, std_ = metric_history.mean(), metric_history.std()
    return mean_, std_


def train_and_save_with_besthp(best_args, if_plot=False):
    # flag tuner
    best_args['tuner_flag'] = False

    # make save dir
    best_dir = best_args['best_trial_save_dir']
    dataset_name = tool.get_basename_split_ext(best_args['dataset_args']['mat_path'])
    best_dataset_dir = os.path.join(best_dir, dataset_name)
    if not os.path.exists(best_dataset_dir):
        os.makedirs(best_dataset_dir)

    # train times with best hp
    repeat_df_list = train_times(best_args, best_args['best_trial'])

    # save df list/ plot df and save
    for idx, df in enumerate(repeat_df_list):
        df.to_csv(os.path.join(best_dataset_dir, f'df{idx}.csv'), index=False, header=True)
        if if_plot:
            fig = mplot.plot_LossMetricTimeLr_with_df(df)
            fig.savefig(os.path.join(best_dataset_dir, f'df{idx}.png'))
            plt.close(fig)
            del fig

    # compute/collect  mean/std metric
    mean_, std_ = mean_std_with_df_list(repeat_df_list, 'metric', 'max')
    mean_std_metric_dict = {}  # {key1:{mean:float,std:float}, key2:{mean:float,std:float}...}
    for key in mean_.keys():
        mean_std_metric_dict[key] = {'mean': mean_[key], 'std': std_[key]}

    # save best_args with metric
    best_args.update(mean_std_metric_dict)
    save_conf_path = os.path.join(best_dataset_dir, 'conf.yaml')
    tool.save_yaml_args(save_conf_path, best_args)


def expand_parser_and_change_args(args):
    parser_args = vars(args)
    parser_args.pop('func')

    # get parser args
    expand_args = benedict()
    change_args = parser_args.pop('change_args')  # List
    for k, v in parser_args.items():
        expand_args[k] = v
    if change_args is not None:
        for change_arg in change_args:
            k, v = change_arg.split('=')
            expand_args[k] = eval(v)
    return expand_args


def parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # public
    def add_public_argument(parser):
        parser.add_argument('--config-paths','-cps',
                            nargs='+',
                            required=True,
                            help='yaml config paths. e.g. config/3sources.yaml',
                            dest='config_paths')
        parser.add_argument('--change-args','-ca',
                            nargs='*',
                            default=None,
                            help='change args. e.g. dataset_args.topk=10 model_args.hid_dim=64',
                            dest='change_args')
        parser.add_argument('--quiet','-q',
                            action='store_true',
                            default=False,
                            help='whether to show logs')

    # run
    parser_run = subparsers.add_parser('run')
    add_public_argument(parser_run)
    parser_run.set_defaults(func=parser_run_func)
    parser_run.add_argument('--run-times','-rt',
                            default=5,
                            type=int,
                            help='run times',
                            dest='run_times')
    parser_run.add_argument('--result-dir','-rd',
                            default='best/',
                            type=str,
                            help='result dir',
                            dest='result_dir')

    args = parser.parse_args()
    args.func(args)


def parser_run_func(args):
    expand_args = expand_parser_and_change_args(args)

    for conf in expand_args['config_paths']:
        yaml_args = benedict.from_yaml(conf)

        # update parser args
        expand_args = tool.remove_dict_None_value(expand_args)  # clean None value
        yaml_args.deepupdate(expand_args)

        # flag tuner
        yaml_args['tuner_flag'] = False

        args = yaml_args.dict()
        if tool.has_hyperparameter(args):
            raise ValueError('Has hyperparameter!!')
        else:
            # only one config and train times
            best_args = args
            best_args['best_trial'] = expand_args['run_times']
            best_args['best_trial_save_dir'] = expand_args['result_dir']
            train_and_save_with_besthp(best_args, if_plot=False)


if __name__ == '__main__':
    parser_args()