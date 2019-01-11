import os
import torch
import shutil
import random
import numpy as np
import pandas as pd
from config import config
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        pass


class EarlyStopping(object):
    ''' adopted from Stefano Nardo's gist https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d '''
    def __init__(self, mode='min', min_delta=0.0, patience=10, percentage=False, initial_bad=0):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = initial_bad
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False


    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False


    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


def set_seed(seed=2050):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, model_name, loss_func, is_best_loss, is_best_f1, fold):
    filename = "{}/{}/{}/{}_checkpoint.pth.tar".format(config.weights_dir, model_name, fold, loss_func)
    torch.save(state, filename)
    if is_best_loss:
        shutil.copyfile(
            filename,
            "{}/{}_fold_{}_{}_model_best_loss.pth.tar".format(config.best_models_dir, model_name, fold, loss_func)
        )
    if is_best_f1:
        shutil.copyfile(
            filename,
            "{}/{}_fold_{}_{}_model_best_f1.pth.tar".format(config.best_models_dir, model_name, fold, loss_func)
        )


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]
    return lr


def time_to_str(t, mode='min'):
    if mode == 'min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr, min)
    elif mode == 'sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min, sec)
    else:
        raise NotImplementedError


def str2bool(v):
    from argparse import ArgumentTypeError
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    raise ArgumentTypeError('Boolean value expected.')


def target_to_numpy(labels: pd.Series):
    y = np.zeros((len(labels), config.num_classes))
    for row_idx, target in enumerate(labels):
        for col_idx in list(map(int, target.split())):
            y[row_idx, col_idx] = 1
    return y


def get_label_weights(labels, mu=0.5):
    y = target_to_numpy(labels)
    return np.round(np.clip(np.log(mu * y.sum() / y.sum(axis=0)), 1, 10), 2)


def get_sample_weights(labels, label_weights=None, use_largest_possible=True):
    if not label_weights:
        label_weights = get_label_weights(labels)
    sample_weights = []
    for label in labels:
        if use_largest_possible:
            sample_weights.append(max([label_weights[col] for col in map(int, label.split())]))
        else:
            sample_weights.append(sum([label_weights[col] for col in map(int, label.split())]))
    return sample_weights


def train_test_multilabel_stratified_shuffle_split(dataset, test_size=0.2, random_state=42):
    y = target_to_numpy(dataset['Target'])
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, valid_idx = list(msss.split(X =dataset, y=y))[0]
    return train_idx, valid_idx


def multilabel_stratified_K_fold(dataset, n_folds=5, shuffle=False, random_state=42):
    y = target_to_numpy(dataset['Target'])
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    return  list(mskf.split(X=dataset, y=y))


if __name__ == '__main__':
    def test_earlystop():
        earlystopper = EarlyStopping(mode='max', min_delta=0.001, patience=10, percentage=False)
        metrics = [1,2,3,4,5,6,7,8,9] + [9] * 15 + [8, 7]
        for idx, metric in enumerate(metrics):
            if earlystopper.step(metric):
                break
        assert [idx, metric] == [18, 9], 'EarlyStopping test failed'


    def plot_msss(dataset='train'):
        from matplotlib import pyplot as plt
        import seaborn as sns
        data = pd.read_csv(config.train_csv) if dataset == 'train' else pd.read_csv(config.hpa_csv)
        N = data.shape[0]
        indices = np.array([np.nan] * N)

        tr_idx, val_idx = train_test_multilabel_stratified_shuffle_split(data)
        indices[tr_idx] = 1
        indices[val_idx] = 0

        cmap_cv = plt.cm.coolwarm
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.scatter(x=range(len(indices)),y=[0.5] * N, c=indices, marker='_', lw=10, cmap=cmap_cv, vmin=-.2, vmax=1.2)
        if not os.path.exists(config.plots_dir):
            os.mkdir(config.plots_dir)
        plt.savefig('{}/{}_msss.png'.format(config.plots_dir, dataset))
        plt.close('all')


    def plot_mskf(dataset='train', num_folds=5, num_samples=200):
        from matplotlib import pyplot as plt
        import seaborn as sns
        data = pd.read_csv(config.train_csv) if dataset == 'train' else pd.read_csv(config.hpa_csv)
        N = data.shape[0]
        indices = np.array([np.nan] * N)
        cmap_cv = plt.cm.coolwarm
        fig, ax = plt.subplots(figsize=(16, 9))
        for fold, (tr_idx, val_idx) in enumerate(multilabel_stratified_K_fold(data, n_folds=num_folds, shuffle=True)):
            indices[tr_idx] = 1
            indices[val_idx] = 0
            ax.scatter(x=range(num_samples) , y=[fold * 1 + 0.5] * num_samples,  c=indices[range(num_samples)],
                       marker='_', lw=40, cmap=cmap_cv, vmin=-.2, vmax=1.2)
        plt.savefig('{}/{}_mskf_{}_{}.png'.format(config.plots_dir, dataset, num_folds, num_samples))
        plt.close('all')


    def test_mskf(dataset='train', num_folds=5):
        data = pd.read_csv(config.train_csv) if dataset == 'train' else pd.read_csv(config.hpa_csv)
        splits = multilabel_stratified_K_fold(data, n_folds=num_folds, shuffle=True)
        assert all(pd.Series(np.concatenate([i[1] for i in splits])).value_counts() == 1), 'splits have overlap!'
        assert all(pd.Series(np.concatenate([i[0] for i in splits])).value_counts() == num_folds - 1), \
            'splits have overlap!'


    def test_label_weights(dataset='train'):
        data = pd.read_csv(config.train_csv) if dataset == 'train' else pd.read_csv(config.hpa_csv)
        label_weights = get_label_weights(data['Target'])
        print('---- label - weight ----')
        for label, label_weight in zip(range(config.num_classes), label_weights):
            print("{: <5} - {: <5}".format(label, label_weight))
        return label_weights


    def test_sample_weights(dataset='train'):
        data = pd.read_csv(config.train_csv) if dataset == 'train' else pd.read_csv(config.hpa_csv)
        test_label_weights(dataset=dataset)
        sample_weights = get_sample_weights(data['Target'])
        print('--- sample label - weight ----')
        for label, sample_weight in zip(data['Target'][:10], sample_weights[:10]):
            print("{: <20} - {: <5}".format(label, sample_weight))
        return sample_weights

    import sys
    if len(sys.argv) > 1:
        command = str(sys.argv[1])
        if command == 'earlystop':
            test_earlystop()
        if command == 'plot_msss':
            plot_msss()
        if command == 'plot_mskf':
            plot_mskf()
        if command == 'test_mskf':
            test_mskf()
        if command == 'label_weights':
            test_label_weights()
        if command == 'sample_weights':
            test_sample_weights()
