import os
import time
import numpy as np
import pandas as pd
import gc; gc.enable()
import torch; torch.backends.cudnn.benchmark = True
import warnings; warnings.filterwarnings('ignore')
from utils import *
from model import *
from multiprocessing import Pool
from tqdm import tqdm
from config import config
from datetime import datetime
from timeit import default_timer as timer
from data import ProteinDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch import nn, optim
from sklearn.metrics import f1_score


def train(train_loader, model, criterion, optimizer, epoch, val_metrics, best_results, start_time, batches_per_update=1,
          train_batch_per_epoch=-1
    ):
    ''' Training model for an iteration with option of accumulating gradients '''
    running_loss = AverageMeter()
    running_f1   = AverageMeter()
    total = train_batch_per_epoch if train_batch_per_epoch > 0 else len(train_loader)

    model.train()
    optimizer.zero_grad()
    for i, (images, target) in enumerate(train_loader):
        if i == total:
            break
        images = images.cuda(non_blocking=True)
        target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
        output = model(images)

        f1_batch = f1_score(target.cpu(), output.sigmoid().cpu() > val_metrics['threshold'], average='macro')
        running_f1.update(f1_batch, images.size(0))

        loss = criterion(output, target)
        running_loss.update(loss.item(), images.size(0))

        loss /= batches_per_update
        loss.backward()

        if (i + 1) % batches_per_update == 0:
            optimizer.step()
            optimizer.zero_grad()

        print('\r', end='', flush=True)
        message = 'train %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
            i / total + epoch, epoch,
            running_loss.avg, running_f1.avg,
            val_metrics['loss_avg'], val_metrics['f1_avg'],
            str(best_results['loss'])[:8], str(best_results['f1'])[:8],
            time_to_str((timer() - start_time), 'min')
        )
        print(message, end='', flush=True)

    print("\n")
    return {
        'loss_avg':  running_loss.avg,
        'f1_avg':    running_f1.avg,
        'threshold': val_metrics['threshold']
    }


def validation(val_loader, model, criterion, epoch, train_metrics, best_results, start, valid_batch_per_epoch=-1):
    ''' validation loop '''
    running_loss = AverageMeter()
    running_f1   = AverageMeter()
    total = valid_batch_per_epoch if valid_batch_per_epoch > 0 else len(val_loader)
    y_valid = []
    y_preds = []

    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if i == total:
                break
            images = images.cuda(non_blocking=True)
            target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
            output = model(images)

            loss = criterion(output, target)
            running_loss.update(loss.item(), images.size(0))

            prediction = output.sigmoid()
            y_preds.append(prediction.cpu().numpy())
            y_valid.append(target.cpu().numpy())

            f1_batch = f1_score(target.cpu(), prediction.cpu().numpy() > train_metrics['threshold'], average='macro')
            running_f1.update(f1_batch,images.size(0))

            print('\r', end='', flush=True)
            message = 'val   %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                i / total + epoch, epoch,
                train_metrics['loss_avg'], train_metrics['f1_avg'],
                running_loss.avg, running_f1.avg,
                str(best_results['loss'])[:8], str(best_results['f1'])[:8],
                time_to_str((timer() - start), 'min')
            )
            print(message, end='', flush=True)

    y_valid = np.vstack(y_valid)
    y_preds = np.vstack(y_preds)

    # --- global thresholding on the full validation data, the resulting f1 is what lr_scheduler and earlystopper uses
    best_f1 = 0
    best_threshold = 0
    for threshold in [.01 + .01 * i for i in range(99)]:
        threshold = np.round(threshold, 3)
        full_validation_f1 = f1_score(y_valid, y_preds > threshold, average='macro')
        if full_validation_f1 > best_f1:
            best_f1 = full_validation_f1
            best_threshold = threshold

    print('\n--------------- on full validation set, best f1 score is {:.4f} with threshold value {:.2f}'\
          .format(best_f1, best_threshold))

    return {
        'loss_avg':  running_loss.avg,
        'f1_avg':    best_f1,
        'threshold': best_threshold
    }


def get_thresholds(tup):
    targets, probs, boot = tup
    sampled_indices = np.random.choice(targets.shape[0], targets.shape[0])
    sampled_targets = targets[sampled_indices]
    sampled_predics = probs[sampled_indices]
    # ---- find label specific threshold
    results = []
    for label in range(targets.shape[1]):
        for threshold in [0.01 + 0.01 * i for i in range(99)]:
            threshold = np.round(threshold, 3)
            f1 = f1_score(sampled_targets[:, label], sampled_predics[:, label] > threshold)
            results.append(pd.Series({"boot": int(boot), "label": int(label), "threshold": threshold, "f1": f1}))
    return results


def thresholding(val_loader, model, model_name, fold, flag, num_bootstraps=30, tta=1, valid_with_hpa=False):
    truetargets = []

    if tta > 1:
        flag += '_tta'

    if valid_with_hpa:
        flag += '_vhpa'

    model.eval()
    # --- get predicted probabilities on validation set
    with torch.no_grad():
        for tta_rounds in tqdm(range(tta), total=tta, desc="get probabilities on threshold val set"):
            probabilities = []
            for idx, (images, target) in tqdm(enumerate(val_loader), total=len(val_loader), desc='inference on validation set', leave=False):
                if tta_rounds == 0:
                    target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
                    truetargets.append(target.cpu().numpy())
                images = images.cuda(non_blocking=True)
                prediction = model(images).sigmoid()
                probabilities.append(prediction.cpu().numpy())
            probabilities = np.vstack(probabilities)
            if tta_rounds == 0:
                overall_probs = probabilities / tta
            else:
                overall_probs += probabilities / tta

    truetargets = np.vstack(truetargets)
    np.save("{}/{}/{}/{}_{}_oof.npy".format(config.weights_dir, model_name, fold, flag, tta), overall_probs)
    # --- find global threshold
    best_f1 = 0
    best_threshold = 0
    for threshold in [.01 + .01 * i for i in range(99)]:
        threshold = np.round(threshold, 3)
        full_validation_f1 = f1_score(truetargets, overall_probs > threshold, average='macro')
        if full_validation_f1 > best_f1:
            best_f1 = full_validation_f1
            best_threshold = threshold
    print("--- best global threshold on validation set {:.4f} with f1 score {:.4f}".format(best_threshold, best_f1))

    # --- find individual thresholds for labels via bootstrap
    p = Pool(8)
    results = p.map(get_thresholds, [(truetargets, probabilities, i) for i in range(num_bootstraps)])
    results = [e for l in results for e in l]

    results = pd.DataFrame(results)
    results['boot'] = results['boot'].astype(int)
    results['label'] = results['label'].astype(int)
    results = results.groupby(['label', 'threshold'])['f1'].mean()
    results = results.reset_index()
    thresholds = results.groupby('label')['f1'].max().reset_index()
    thresholds = thresholds.merge(right=results, how='left', on=['label', 'f1'])
    thresholds = thresholds.groupby(['label', 'f1'])['threshold'].mean().reset_index()

    # --- get overall f1 score using individual thresholds
    preds = np.zeros_like(overall_probs)
    for label in thresholds.label:
        preds[:, int(label)] = overall_probs[:, int(label)] > float(thresholds.loc[thresholds.label == label]['threshold'])
    f1 = f1_score(truetargets, preds, average='macro')
    print("--- best f1 score with label specific thresholds is {:.4f}".format(f1))
    return {
        "detail":       results,
        "thresholds":   thresholds[["label", "threshold"]].set_index('label').T.to_dict(orient='record')[0],
        "threshold":    best_threshold,
        "best_f1":      f1,
        "best_flat_f1": best_f1
    }


def test(test_loader, model, model_name, results, fold=0, loss_func="f1", flag="best_f1", save_raw_scores=False,
         tta=1, valid_with_hpa = False
    ):
    sub = pd.read_csv(config.test_csv)
    overall_probs = np.zeros((sub.shape[0], config.num_classes))

    if tta > 1:
        flag += '_tta'

    if valid_with_hpa:
        flag += '_vhpa'

    model.eval()
    for tta_rounds in tqdm(range(tta)):
        probabilities = []
        for images, _ in tqdm(test_loader, total=len(test_loader), desc='inference on test images', leave=False):
            with torch.no_grad():
                images = images.cuda(non_blocking=True)
                probas = model(images).sigmoid().cpu().data.numpy()
                probabilities.append(probas)
        probabilities = np.vstack(probabilities)
        overall_probs +=  probabilities / tta

    if save_raw_scores:
        np.save("{}/{}_{}_fold_{}_{}_{}_raw.npy".format(config.submit_dir, model_name, loss_func, fold, flag, tta), overall_probs)


    print("--- make prediction with a global threshold {}".format(results['threshold']))
    predictions = []
    for row in range(len(overall_probs)):
        predictions.append(' '.join(map(str, np.array(range(28))[overall_probs[row,:] > results['threshold']])))
    sub['Predicted'] = predictions
    sub.to_csv('{}/{}_{}_fold_{}_{}_{}_sub_thresh_{}.csv'.format(config.submit_dir, model_name, loss_func, fold,
                flag, tta, results['threshold']), index=None)

    print("--- make prediction with label specific thresholds")
    print(results["thresholds"])
    predictions = []
    for row in range(len(overall_probs)):
        predictions.append(" ".join(str(int(label)) for (label, threshold) in results['thresholds'].items()
                                                        if overall_probs[row, label] > threshold))
    sub['Predicted'] = predictions
    sub.to_csv('{}/{}_{}_fold_{}_{}_{}_thresholds_val_f1_{:.3f}.csv'.format(config.submit_dir, model_name,
               loss_func, fold, flag, tta, results['best_f1']), index=None)


def main(model_name='bninception', resume=True, resume_with_state='', split_method = 'msss', fold=0, seed=2050,
         train_with_hpa=False, valid_with_hpa=False, predict_only=False, weighted_samples=False, reset_lr=-1, gpu="0",
         data_loader_workers=4, earlystop_initial_bad=0, loss_func='bce', epoches_to_run=10, batch_size=96,
         train_batch_per_epoch=-1, valid_batch_per_epoch=-1, flag="best_f1", weighted_validation=False,
         batches_per_update=1, lr_scheduler_patience=4, earlystop_patience=7, tta=1
    ):
    # --- set overall seed
    set_seed(seed)

    # --- cross validation set up
    train_csv = pd.read_csv(config.train_csv)
    hpa_csv   = pd.read_csv(config.hpa_csv)
    test_csv  = pd.read_csv(config.test_csv)


    if split_method == 'msss':
        train_tr_idx, train_val_idx = train_test_multilabel_stratified_shuffle_split(train_csv)
        hpa_tr_idx  , hpa_val_idx   = train_test_multilabel_stratified_shuffle_split(hpa_csv)
    else:
        train_tr_idx, train_val_idx = multilabel_stratified_K_fold(train_csv, n_folds=5, shuffle=False)[fold]
        hpa_tr_idx  , hpa_val_idx   = multilabel_stratified_K_fold(hpa_csv, n_folds=5, shuffle=False)[fold]

    train_paths  = train_csv.iloc[train_tr_idx,  :]['Id'].map(lambda x: os.path.join(config.train_data_dir, x))
    valid_paths  = train_csv.iloc[train_val_idx, :]['Id'].map(lambda x: os.path.join(config.train_data_dir, x))
    test_paths   = test_csv['Id'].map(lambda x: os.path.join(config.test_data_dir, x))
    train_labels = train_csv.iloc[train_tr_idx,  :]['Target']
    valid_labels = train_csv.iloc[train_val_idx, :]['Target']
    test_labels = None

    hpa_train_paths  = hpa_csv.iloc[hpa_tr_idx,  :]['Id'].map(lambda x: os.path.join(config.hpa_data_dir, x))
    hpa_valid_paths  = hpa_csv.iloc[hpa_val_idx, :]['Id'].map(lambda x: os.path.join(config.hpa_data_dir, x))
    hpa_train_labels = hpa_csv.iloc[hpa_tr_idx,  :]['Target']
    hpa_valid_labels = hpa_csv.iloc[hpa_val_idx, :]['Target']

    if train_with_hpa:
        train_paths  = pd.concat([train_paths,  hpa_train_paths])
        train_labels = pd.concat([train_labels, hpa_train_labels])

    if valid_with_hpa:
        valid_paths  = pd.concat([valid_paths,  hpa_valid_paths])
        valid_labels = pd.concat([valid_labels, hpa_valid_labels])
    print("----- training with {} images; validate with {} images".format(len(train_paths), len(valid_paths)))

    # --- data generators
    train_gen = ProteinDataset(train_paths, train_labels, augment=True,  mode="train")
    val_gen   = ProteinDataset(valid_paths, valid_labels, augment=False, mode="train")

    train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True,  pin_memory=True,
                              num_workers=data_loader_workers)
    val_loader   = DataLoader(val_gen,   batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=data_loader_workers)

    if weighted_samples:
        train_sample_weights = get_sample_weights(train_labels)
        train_sampler = WeightedRandomSampler(train_sample_weights, len(train_labels))
        train_loader = DataLoader(train_gen, batch_size=batch_size, sampler=train_sampler, pin_memory=True,
                                  num_workers=data_loader_workers)
    if weighted_validation:
        val_sample_weights = get_sample_weights(valid_labels)
        val_sampler = WeightedRandomSampler(val_sample_weights, len(valid_labels))
        val_loader = DataLoader(val_gen, batch_size=batch_size, sampler=val_sampler, pin_memory=True,
                                  num_workers=data_loader_workers)

    # --- get model
    model = get_model(model_name)

    # --- loss
    if loss_func == "bce":
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif loss_func == "f1":
        criterion = F1_Loss().cuda()

    # --- optimizer
    optimizer = optim.Adam(model.parameters(), lr = config.lr)

    # --- learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=lr_scheduler_patience,
        verbose=True, cooldown=1, min_lr=1e-6)

    # --- early stopping
    earlystopper = EarlyStopping(mode='max', min_delta=0.001, patience=earlystop_patience, percentage=False,
        initial_bad=earlystop_initial_bad)

    # --- stuff to monit during training/validation
    start_epoch = 0
    best_results = {'loss': 1.000001, 'f1': 0.000011}
    val_metrics = {'loss_avg': 1.000001, 'f1_avg': 0.000011, 'threshold': 0.5}

    # --- resume training from previous checkpoint
    if resume:
        # ---- load model state
        if not resume_with_state:
            # load the latest iteration if no checkpoint be specified
            resume_with_state = "{}/{}/{}/{}_checkpoint.pth.tar".format(config.weights_dir, model_name, fold, loss_func)
        print("--- loading model from state {}".format(resume_with_state))
        model_state = torch.load(resume_with_state)
        model.load_state_dict(model_state["state_dict"])

        # ---- get epoch, loss, f1 history
        start_epoch = model_state['epoch']
        # ---- save a copy of old checkpoint
        shutil.copyfile(
            resume_with_state,
            resume_with_state + "_epoch_{}_cp".format(start_epoch)
        )
        best_results = {'loss': model_state['best_loss'], 'f1': model_state['best_f1']}

        # ---- get optimizer state
        if "optimizer" in model_state:
            optimizer = optim.Adam(model.parameters(), lr = config.lr)
            optimizer.load_state_dict(model_state["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        # ---- get lr scheduler state
        if "scheduler" in model_state:
            scheduler.load_state_dict(model_state["scheduler"])
            earlystopper.best = model_state['best_f1']

        # ---- free up memeory
        del model_state
        gc.collect()

    # --- hardcode a new learinig rate
    if reset_lr > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = reset_lr
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, verbose=True,
                                                         patience=lr_scheduler_patience, cooldown=1, min_lr=1e-6)
        print('--- resetting learning rate')

    # --- create checkpoint folder
    model_checkpoint_dir = os.path.join(config.weights_dir, model_name, str(fold))
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir)

    # --- logging header
    header = 'START  ' if not resume else 'RESUME '
    header = 'PREDICT' if predict_only else header
    log = Logger()
    log.open("{}/{}_log_train.txt".format(config.logs_dir, model_name), mode="a")
    log.write(
        "\n" + "|" + "-" * 48 +
        " [ {} fold {} {} ] ".format(header, fold, datetime.now().strftime("%Y-%m-%d %H:%M:%S")) +
        '-' * 48 + "|" + "\n"
    )
    log.write('|--------------------------|------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n')
    log.write('| mode    iter     epoch   |         loss   f1_macro        |         loss   f1_macro       |         loss   f1_macro       | time       |\n')
    log.write('|----------------------------------------------------------------------------------------------------------------------------------------|\n')

    model.cuda()
    start = timer()
    if not predict_only:
        print('--- initial learning rate is: {}'.format(get_learning_rate(optimizer)))
        for epoch in range(start_epoch, start_epoch + epoches_to_run):
            train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, best_results, start,
                                  batches_per_update, train_batch_per_epoch)
            val_metrics   = validation(val_loader, model, criterion, epoch, train_metrics, best_results, start,
                                       valid_batch_per_epoch)
            scheduler.step(val_metrics['f1_avg'])

            is_best_loss = val_metrics['loss_avg'] < best_results['loss']
            best_results['loss'] = min(val_metrics['loss_avg'], best_results['loss'])
            is_best_f1 = val_metrics['f1_avg'] > best_results['f1']
            best_results['f1'] = max(val_metrics['f1_avg'], best_results['f1'])
            save_checkpoint(
                state = {
                    "model_name": model_name,
                    "epoch":      epoch + 1,
                    "fold":       fold,
                    "best_f1":    best_results['f1'],
                    "best_loss":  best_results['loss'],
                    "optimizer":  optimizer.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                    "state_dict": model.state_dict(),
                },
                fold         = fold,
                model_name   = model_name,
                loss_func    = loss_func,
                is_best_f1   = is_best_f1,
                is_best_loss = is_best_loss
            )

            print('\r', end='', flush=True)
            log.write('%s  %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                "best", epoch, epoch,
                train_metrics['loss_avg'], train_metrics['f1_avg'],
                val_metrics['loss_avg'], val_metrics['f1_avg'],
                str(best_results['loss'])[:8], str(best_results['f1'])[:8],
                time_to_str((timer() - start), 'min'))
            )
            log.write("\n")

            if earlystopper.step(val_metrics['f1_avg']):
                print('--- terminated via early stopping\n')
                break

            time.sleep(10)

    # --- load model
    if flag == "last_iter":
        model_state_loc = "{}/{}/{}/{}_checkpoint.pth.tar".format(config.weights_dir, model_name, str(fold), loss_func)
    else:
        model_state_loc = "{}/{}_fold_{}_{}_model_{}.pth.tar".format(config.best_models_dir, model_name, str(fold),
                           loss_func, flag)

    model_state = torch.load(model_state_loc)
    model.load_state_dict(model_state["state_dict"])

    if tta > 1:
        test_gen  = ProteinDataset(test_paths,  test_labels,  augment=True, mode="test")
        val_gen   = ProteinDataset(valid_paths, valid_labels, augment=True, mode="train")
    else:
        test_gen  = ProteinDataset(test_paths,  test_labels,  augment=False, mode="test")
        val_gen   = ProteinDataset(valid_paths, valid_labels, augment=False, mode="train")

    val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=False, pin_memory=True,
                              num_workers=data_loader_workers)
    test_loader  = DataLoader(test_gen, batch_size=batch_size, shuffle=False, pin_memory=True,
                              num_workers=data_loader_workers)

    # --- thresholding and predicts
    print("--- thresholding and make predictions with model {}: ".format(model_state_loc))
    results = thresholding(val_loader, model, model_name, fold, flag, num_bootstraps=50, tta=tta)
    test(test_loader = test_loader, model=model, model_name=model_name, results=results, fold=fold,
         loss_func=loss_func, flag=flag, save_raw_scores=True, tta=tta)

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description='model training aruguments')
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--predict_only', type=str2bool, default='false')
    parser.add_argument('--model_name', type=str, default="bninception")
    parser.add_argument('--loss_func', type=str, default="bce")
    parser.add_argument('--resume', type=str2bool, default='false')
    parser.add_argument('--resume_with_state', type=str, default='')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epoches_to_run', type=int, default=65535)
    parser.add_argument('--flag', type=str, default="last_iter")
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--batches_per_update', type=int, default=1)
    parser.add_argument('--train_batch_per_epoch', type=int, default=-1)
    parser.add_argument('--valid_batch_per_epoch', type=int, default=-1)
    parser.add_argument('--split_method', type=str, default='msss')
    parser.add_argument('--seed', type=int, default=2050)
    parser.add_argument('--earlystop_initial_bad', type=int, default=0)
    parser.add_argument('--earlystop_patience', type=int, default=7)
    parser.add_argument('--lr_scheduler_patience', type=int, default=4)
    parser.add_argument('--data_loader_workers', type=int, default=4)
    parser.add_argument('--train_with_hpa', type=str2bool, default='false')
    parser.add_argument('--valid_with_hpa', type=str2bool, default='false')
    parser.add_argument('--reset_lr', type=float, default=-1)
    parser.add_argument('--weighted_samples', type=str2bool, default='false')
    parser.add_argument('--tta', type=int, default=1)
    parser.add_argument('--weighted_validation', type=str2bool, default='false')
    args=parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    assert args.split_method in ['msss', 'mskf'], "invalid split_method"
    assert args.loss_func in ['f1', 'bce'], "invalid loss_func"
    assert args.model_name in ['bninception', 'nasnet_large', 'resnet50', 'resnet34'], "invalid model_name"

    main(**vars(args))
