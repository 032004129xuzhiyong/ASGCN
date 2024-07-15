# -*- coding:utf-8 -*-
import copy
import os
import argparse
import optuna
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


def objective(trial: optuna.trial.Trial, extra_args):
    """
    Parameters:
    trial (optuna.trial.Trial): The trial object from the Optuna library.
    extra_args (dict): A dictionary of extra arguments for the model training.

    Returns:
    float: The value of the monitored metric for the current trial.
    """
    # Create a deep copy of the extra arguments to avoid modifying the original dictionary
    args = copy.deepcopy(extra_args)

    # Modify the arguments with the current trial's parameters
    args = tool.modify_dict_with_trial(args, trial)

    # Set the trial object and the tuner flag in the arguments
    args['trial'] = trial
    args['tuner_flag'] = True

    # set seed
    if 'seed' not in args.keys():
        args['seed'] = 42
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])

    # Run the model training for the first time and get the logs
    data = load_mat(**args['dataset_args'])  # load data
    first_logs = train_one_args(args,data)

    # Run the model training for other times
    args['tuner_flag'] = False
    if isinstance(args['tuner_n_repeats'],int):
        args['tuner_n_repeats'] = max(args['tuner_n_repeats'], 1)
    else:
        args['tuner_n_repeats'] = 1
    repeat_df_list = train_times(args, args['tuner_n_repeats']-1, data)
    repeat_df_list.append(pd.DataFrame(first_logs))

    # compute the mean metric/loss
    mean_logs = {}
    mean_, std_ = mean_std_with_df_list(repeat_df_list, 'metric', 'max')
    mean_logs.update(mean_)
    mean_, std_ = mean_std_with_df_list(repeat_df_list, 'loss', 'min')
    mean_logs.update(mean_)

    # Set the mean logs and the configuration as user attributes of the trial
    trial.set_user_attr('mean_logs', mean_logs)
    args.pop('trial')
    trial.set_user_attr('config', args)

    # Return the value of the monitored metric for the current trial
    return mean_logs[args['tuner_monitor']]


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

    # tuner
    parser_tuner = subparsers.add_parser('tuner')
    add_public_argument(parser_tuner)
    parser_tuner.set_defaults(func=parser_tuner_func)

    # run
    parser_run = subparsers.add_parser('run')
    add_public_argument(parser_run)
    parser_run.set_defaults(func=parser_run_func)
    parser_run.add_argument('--run-times','-rt',
                            default=1,
                            type=int,
                            help='run times',
                            dest='run_times')
    parser_run.add_argument('--result-dir','-rd',
                            default='temp_result/',
                            type=str,
                            help='result dir',
                            dest='result_dir')

    # grid search
    parser_grid = subparsers.add_parser('grid')
    add_public_argument(parser_grid)
    parser_grid.set_defaults(func=parser_grid_func)
    parser_grid.add_argument('--grid-search-space','-gss',
                             nargs='*',
                             default=None,
                             help='grid search space. e.g. dataset_args.topk=[10,20,30] model_args.hid_dim=[64,128]',
                             dest='grid_search_space')
    parser_grid.add_argument('--search-space-simultaneously','-sss',
                             action='store_true',
                             default=False,
                             help='search space simultaneously',
                             dest='search_space_simultaneously')

    args = parser.parse_args()
    args.func(args)


def parser_tuner_func(args):
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
            # tuner
            args['tuner_flag'] = True

            if 'loss' in args['tuner_monitor']:
                direction = 'minimize'
            else:
                direction = 'maximize'
            study = optuna.create_study(direction=direction,
                                        study_name=tool.get_basename_split_ext(args['dataset_args']['mat_path']),
                                        storage=optuna.storages.RDBStorage('sqlite:///./tuner.db',
                                                                           heartbeat_interval=60,
                                                                           grace_period=120,
                                                                           failed_trial_callback=optuna.storages.RetryFailedTrialCallback(3)),
                                        load_if_exists=True,
                                        pruner=mtuner.CombinePruner([
                                            optuna.pruners.MedianPruner(n_warmup_steps=30),
                                            optuna.pruners.PercentilePruner(0.1, n_warmup_steps=30)
                                        ]),
                                        #pruner=optuna.pruners.NopPruner(),
                                        sampler=optuna.samplers.TPESampler())
            study.optimize(lambda trial: objective(trial, args),
                           n_trials=args['tuner_n_trials'],
                           gc_after_trial=True,
                           show_progress_bar=True,
                           callbacks=[mcallback.StudyStopWhenTrialKeepBeingPrunedCallback(20)])

            # get best args
            best_hp = study.best_params
            # print
            for i in range(5):
                print('*' * 50)
            tool.print_dicts_tablefmt([best_hp], ['Best HyperParameters!!'])
            for i in range(5):
                print('*' * 50)

            # train times with best args
            best_args = tool.modify_dict_with_trial(args, study.best_trial)
            train_and_save_with_besthp(best_args, if_plot=True)
        else:
            raise ValueError('No hyperparameter to tune!!')


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
            train_and_save_with_besthp(best_args, if_plot=True)


def parser_grid_func(args):
    parser_args = vars(args)
    grid_search_space = parser_args.pop('grid_search_space')  # List
    expand_args = expand_parser_and_change_args(args)

    # get grid search space
    # grid_search_space: List[str] e.g. ['dataset_args.topk=[10,20,30]','model_args.hid_dim=[64,128]']
    # in grid_search_space, key is full name, value is list
    # but in parser_grid_search_space, key is short name, value is list
    parser_grid_search_space = {}
    if grid_search_space is not None:
        for grid_search_arg in grid_search_space:
            k, v = grid_search_arg.split('=')
            parser_grid_search_space[k.split('.')[-1]] = eval(v)

    for conf in expand_args['config_paths']:
        # best/dataset/conf.yaml It has no hyperparameter to tune
        yaml_args = benedict.from_yaml(conf)

        # update parser args
        # update ahead of transform_dict_to_search_space to remove other change_args
        expand_args = tool.remove_dict_None_value(expand_args)
        yaml_args.deepupdate(expand_args)

        if not tool.has_hyperparameter(yaml_args.dict()):
            if grid_search_space is not None:
                if len(grid_search_space) > 1 and yaml_args['search_space_simultaneously']:
                    # search space simultaneously
                    for grid_search_arg in grid_search_space:
                        k, v = grid_search_arg.split('=')
                        # add something in location which will be searched in grid search.
                        # It is a placehold, and will be replaced by grid search.
                        v = eval(v)
                        if isinstance(v[0], int):
                            yaml_args[k] = {'type': 'int', 'low': 0, 'high': 10}
                        elif isinstance(v[0], float):
                            yaml_args[k] = {'type': 'float', 'low': 0.0, 'high': 10.0}
                        elif isinstance(v[0], str):
                            yaml_args[k] = {'type': 'categorical', 'choices': v}
                    # now yaml_args has hyperparameter to tune
                    # tuner
                    yaml_args['tuner_flag'] = True
                    args = yaml_args.dict()
                    if 'loss' in args['tuner_monitor']:
                        direction = 'minimize'
                    else:
                        direction = 'maximize'
                    # change study_name
                    # because simultaneous search space can be not compatible with separate search space
                    study_name = 'grid_'+tool.get_basename_split_ext(args['dataset_args']['mat_path']) + '_simultaneously'
                    study = optuna.create_study(direction=direction,
                                                study_name=study_name,
                                                storage=optuna.storages.RDBStorage('sqlite:///./tuner.db',
                                                                                   heartbeat_interval=60,
                                                                                   grace_period=120,
                                                                                   failed_trial_callback=optuna.storages.RetryFailedTrialCallback(3)),
                                                load_if_exists=True,
                                                pruner=optuna.pruners.NopPruner(),
                                                sampler=optuna.samplers.GridSampler(parser_grid_search_space))
                    study.optimize(lambda trial: objective(trial, args),
                                   n_trials=99999,
                                   gc_after_trial=True,
                                   show_progress_bar=True,
                                   callbacks=[mcallback.StudyStopWhenTrialKeepBeingPrunedCallback(20)])
                else:
                    # separate parser_grid_search_space to many search space
                    # then search space one by one
                    for grid_search_arg in grid_search_space:
                        k, v = grid_search_arg.split('=')
                        # add something in location which will be searched in grid search.
                        # It is a placehold, and will be replaced by grid search.
                        v = eval(v)
                        cur_sep_yaml_args = copy.deepcopy(yaml_args)
                        if isinstance(v[0], int):
                            cur_sep_yaml_args[k] = {'type': 'int', 'low': 0, 'high': 10}
                        elif isinstance(v[0], float):
                            cur_sep_yaml_args[k] = {'type': 'float', 'low': 0.0, 'high': 10.0}
                        elif isinstance(v[0], str):
                            cur_sep_yaml_args[k] = {'type': 'categorical', 'choices': v}
                        # now cur_sep_yaml_args has hyperparameter to tune
                        # note: cur_sep_yaml_args is a copy of yaml_args and it has only one hyperparameter to tune
                        # tuner
                        cur_sep_yaml_args['tuner_flag'] = True
                        cur_sep_parser_grid_search_space = {k.split('.')[-1]: v}
                        args = cur_sep_yaml_args.dict()
                        if 'loss' in args['tuner_monitor']:
                            direction = 'minimize'
                        else:
                            direction = 'maximize'
                        study = optuna.create_study(direction=direction,
                                                    study_name='grid_'+tool.get_basename_split_ext(args['dataset_args']['mat_path']),
                                                    storage=optuna.storages.RDBStorage('sqlite:///./tuner.db',
                                                                                       heartbeat_interval=60,
                                                                                       grace_period=120,
                                                                                       failed_trial_callback=optuna.storages.RetryFailedTrialCallback(3)),
                                                    load_if_exists=True,
                                                    pruner=optuna.pruners.NopPruner(),
                                                    sampler=optuna.samplers.GridSampler(cur_sep_parser_grid_search_space))
                        study.optimize(lambda trial: objective(trial, args),
                                       n_trials=99999,
                                       gc_after_trial=True,
                                       show_progress_bar=True,
                                       callbacks=[mcallback.StudyStopWhenTrialKeepBeingPrunedCallback(20)])
            else:
                raise ValueError('No hyperparameter to tune!!')
        else:
            # yaml file has hyperparameter to tune
            if grid_search_space is not None:
                raise ValueError('yaml file has hyperparameter to tune, grid_search_space should be None!!')

            search_grid_num = len(yaml_args.search('type',in_keys=True,in_values=False,exact=True,case_sensitive=True))
            if search_grid_num > 1 and not yaml_args['search_space_simultaneously']:
                raise ValueError('yaml file has more than one hyperparameter to tune, '
                                 'search_space_simultaneously should be True!!')
            if search_grid_num > 1:
                print('yaml file has more than one hyperparameter to grid search!!')

            # tuner
            yaml_args['tuner_flag'] = True
            parser_grid_search_space = tool.transform_dict_to_search_space(yaml_args.dict())
            args = yaml_args.dict()
            if 'loss' in args['tuner_monitor']:
                direction = 'minimize'
            else:
                direction = 'maximize'
            # change study_name
            # because simultaneous search space can be not compatible with separate search space
            if search_grid_num > 1:
                study_name = 'grid_'+tool.get_basename_split_ext(args['dataset_args']['mat_path']) + '_simultaneously'
            else:
                study_name = 'grid_'+tool.get_basename_split_ext(args['dataset_args']['mat_path'])
            study = optuna.create_study(direction=direction,
                                        study_name=study_name,
                                        storage=optuna.storages.RDBStorage('sqlite:///./tuner.db',
                                                                           heartbeat_interval=60,
                                                                           grace_period=120,
                                                                           failed_trial_callback=optuna.storages.RetryFailedTrialCallback(3)),
                                        load_if_exists=True,
                                        pruner=optuna.pruners.NopPruner(),
                                        sampler=optuna.samplers.GridSampler(parser_grid_search_space))
            study.optimize(lambda trial: objective(trial, args),
                           n_trials=99999,
                           gc_after_trial=True,
                           show_progress_bar=True,
                           callbacks=[mcallback.StudyStopWhenTrialKeepBeingPrunedCallback(20)])


if __name__ == '__main__':
    parser_args()