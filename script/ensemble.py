import pandas as pd
import numpy as np
from config import config
from utils import target_to_numpy, multilabel_stratified_K_fold
from sklearn.metrics import f1_score

flag = "last_iter_tta_10"

# --- use oof predictions to find threshold
train_csv = pd.read_csv(config.train_csv)
y = target_to_numpy(train_csv['Target'])
oofs = np.zeros((31072, 28))
splits = multilabel_stratified_K_fold(train_csv, n_folds=5, shuffle=False)
for fold, split in enumerate(splits):
    _, val_idx = split
    oofs[val_idx] = np.load('{}/resnet50/{}/last_iter_tta_10_oof.npy'.format(config.weights_dir, fold))

best_f1 = 0
best_threshold = 0.01
for threshold in [0.01 + 0.01 * i for i in range(99)]:
    y_pred = []
    for row in range(len(oofs)):
        y_pred.append(' '.join(map(str, np.array(range(28))[oofs[row,:] > threshold])))
        if y_pred[-1] == '':
            y_pred[-1] = str(oofs[row, :].argmax())
    f1 = f1_score(y, target_to_numpy(y_pred), average='macro')
    print(np.round(threshold,3), np.round(f1, 3))
    if f1 > best_f1:
        best_threshold = threshold
        best_f1 = f1


# simply average of 5 model probabilities, use ^ found threshold to generate final prediction
# for row with no prediction, use the one with maximum output

sub = pd.read_csv(config.test_csv)
probabilities = np.zeros((11702, 28))
for fold in [0,1,2,3,4]:
    probabilities += np.load("{}/resnet50_bce_fold_{}_{}_raw.npy".format(config.submit_dir, fold, flag)) / 5

predictions = []
for row in range(len(probabilities)):
    predictions.append(' '.join(map(str, np.array(range(28))[probabilities[row,:] > best_threshold])))
    if predictions[-1] == '':
        predictions[-1] = str(probabilities[row, :].argmax())
sub['Predicted'] = predictions
sub.to_csv('{}/resnet50_bce_5_folds_last_iter_tta_10_thresh_0.39_threshold_max_fixed.csv'.format(config.submit_dir), index=False)

# append public leak to submission
external = pd.read_csv(config.hpa_csv)
public_leak = pd.read_csv(config.public_leak)
public_leak['Id'] = public_leak.Extra.map(lambda x: '_'.join(x.split('_')[1:]))
public_leak = public_leak.merge(external, on='Id', how='left')

osub = sub.copy()
for idx, row in sub.iterrows():
    if row['Id'] in public_leak['Test'].values:
        sub.iloc[idx, 1] = public_leak.loc[public_leak['Test'] == row['Id']]['Target'].values[0]
sub.to_csv('{}/resnet50_bce_5_folds_last_iter_tta_10_thresh_0.39_threshold_max_fixed_with_leak.csv', index=False)