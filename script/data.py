import random
import numpy as np
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import albumentations as A
from utils import *
from config import config
import torch
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    ''' dataset class for competition '''
    def __init__(self, image_prefix_paths, labels, augment=True, mode="train"):
        self.image_prefix_paths = image_prefix_paths
        self.labels = labels
        self.augment = augment
        self.mode = mode


    def __len__(self):
        return len(self.image_prefix_paths)


    def __getitem__(self, index):
        X = self.__read_image(index)
        y = str(self.image_prefix_paths.iloc[index].split('/')[-1])

        if self.mode != "test":
            labels = np.array(list(map(int, self.labels.iloc[index].split(' '))))
            y = np.eye(config.num_classes, dtype=np.float)[labels].sum(axis=0)

        if self.augment:
            X = self.__augment_image(X)

        X = np.rollaxis(X, 2, 0) # swap to make channel first
        X = torch.from_numpy(X)
        return X.float(), y


    def __read_image(self, index):
        image = np.zeros(shape=(512, 512, 4)) if config.channels == 4 else np.zeros(shape=(512, 512, 3))
        img_prefix_path = self.image_prefix_paths.iloc[index]

        image[:,:,0] = np.array(cv2.imread(img_prefix_path + "_red.png", 0))
        image[:,:,1] = np.array(cv2.imread(img_prefix_path + "_green.png", 0))
        image[:,:,2] = np.array(cv2.imread(img_prefix_path + "_blue.png", 0))

        if config.channels == 4:
            image[:,:,3] = np.array(cv2.imread(img_prefix_path + "_yellow.png", 0))
        if config.img_height != 512:
            image = cv2.resize(image, (config.img_width, config.img_height))

        image /= 255
        return image


    def __augment_image(self, image):
        random.seed()
        np.random.seed()

        train_aug = A.Compose(
            transforms = [
                A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512, p=1),
                A.RandomRotate90(p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.IAAAffine(shear=15, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=.3),
                A.Blur(blur_limit=2, p=.33),
                A.OpticalDistortion(p=.33),
                A.RandomBrightnessContrast(p=.33)
            ],
            p=1
        )

        test_aug = A.Compose(
            transforms = [
                A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512, p=1),
                A.RandomRotate90(p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            p=1
        )

        if self.mode == 'train':
            augmented_image = train_aug(image=image)['image']
        else:
            augmented_image = test_aug(image=image)['image']
        return augmented_image


if __name__ == '__main__':
    def speed_test(num_workers=4):
        import os
        import pandas as pd
        from torch.utils.data import DataLoader
        from timeit import default_timer as timer

        train_csv = pd.read_csv(config.train_csv)
        train_csv['image_path'] = train_csv['Id'].map(lambda x: os.path.join(config.train_data_dir, x))
        train_gen = ProteinDataset(train_csv['image_path'], train_csv['Target'], augment=False, mode="train")
        data_loader = DataLoader(train_gen, batch_size=32, shuffle=True, pin_memory=True, num_workers=num_workers)

        start = timer()
        for idx, (X, y) in enumerate(data_loader):
            if (idx + 1) * data_loader.batch_size >= len(data_loader.dataset) // 5:
                print('time to cycle through 20 percent of training set ({} images) : {}'.\
                    format(idx * data_loader.batch_size, time_to_str(timer() - start, 'sec')))
                break


    def augmentation_inspection():
        import pandas as pd
        from matplotlib import pyplot as plt
        import os

        np.random.seed()

        def save_plots(data_gen, num_images, num_augs, file_prefix):
            image_indices = np.random.choice(range(len(data_gen)), num_images)
            for image_idx in image_indices:
                fig, ax = plt.subplots(nrows=num_augs, ncols=4, figsize=(20, 5 * num_augs))
                images = [data_gen[image_idx][0] for _ in range(num_augs)]
                for row, image in enumerate(images):
                    for col, cmap in enumerate(['Reds', 'Greens', 'Blues', 'Oranges']):
                        ax[row][col].set_title(cmap)
                        ax[row][col].imshow(image[col].numpy(), cmap=cmap)
                plt.savefig('../sample_data/{}_image_aug_inspect_{}.png'.format(file_prefix, image_idx))
                plt.close('all')

        train_csv = pd.read_csv(config.train_csv)
        train_gen = ProteinDataset(
            image_prefix_paths =  train_csv['Id'].map(lambda x: os.path.join(config.train_data_dir, x)),
            labels = train_csv['Target'],
            augment=True,
            mode="train"
        )
        save_plots(train_gen, 10, 6, 'train')

        test_csv = pd.read_csv(config.test_csv)
        test_gen = ProteinDataset(
            image_prefix_paths = test_csv['Id'].map(lambda x: os.path.join(config.test_data_dir, x)),
            labels = None,
            augment=True,
            mode="test"
        )
        save_plots(test_gen, 10, 6, 'test')

        hpa_csv = pd.read_csv(config.hpa_csv)
        hpa_gen = ProteinDataset(
            image_prefix_paths = hpa_csv['Id'].map(lambda x: os.path.join(config.hpa_data_dir, x)),
            labels = hpa_csv['Target'],
            augment=True,
            mode="train"
        )
        save_plots(hpa_gen, 10, 6, 'hpa')

        print('checkout ../sample_data to see augmentation results')


    def eda(num_samples=5, num_duplicates=1):
        import pandas as pd
        from matplotlib import pyplot as plt
        import seaborn as sns
        plt.style.use('fivethirtyeight')
        from utils import target_to_numpy

        train_csv = pd.read_csv(config.train_csv)
        hpa_csv   = pd.read_csv(config.hpa_csv)
        train_gen = ProteinDataset(
            image_prefix_paths =  train_csv['Id'].map(lambda x: os.path.join(config.train_data_dir, x)),
            labels = train_csv['Target'],
            augment=False,
            mode="train"
        )
        hpa_gen = ProteinDataset(
            image_prefix_paths =  hpa_csv['Id'].map(lambda x: os.path.join(config.hpa_data_dir, x)),
            labels = hpa_csv['Target'],
            augment=False,
            mode="train"
        )
        y_train = target_to_numpy(train_csv['Target'])
        y_hpa = target_to_numpy(hpa_csv['Target'])

        train_label_card = (pd.Series(y_train.sum(axis=1)).value_counts() / y_train.sum()).reset_index()
        train_label_card.columns = ['card', 'freq']
        train_label_card['data'] = 'train'
        hpa_label_card   = (pd.Series(y_hpa.sum(axis=1)).value_counts() / y_hpa.sum()).reset_index()
        hpa_label_card.columns = ['card', 'freq']
        hpa_label_card['data'] = 'hpa'
        label_card = pd.concat([train_label_card, hpa_label_card], axis=0)
        fig, ax = plt.subplots(figsize=(16, 9))
        g = sns.catplot(x="card", y="freq", hue="data", data=label_card, height=10, kind="bar", palette="muted",
                        ci=None, legend_out=False, aspect=2)
        fig.suptitle("label cardinality distribution")
        plt.savefig('{}/label_cardinality_distribution.png'.format(config.plots_dir))
        plt.close('all')


        train_label_dist = (pd.Series(y_train.sum(axis=0)) / y_train.sum()).reset_index()
        train_label_dist.columns = ['label', 'freq']
        train_label_dist['data'] = 'train'
        hpa_label_dist   = (pd.Series(y_hpa.sum(axis=0)) / y_hpa.sum()).reset_index()
        hpa_label_dist.columns = ['label', 'freq']
        hpa_label_dist['data'] = 'hpa'
        probed_lb_dist = pd.DataFrame({
            'label': list(range(28)),
            'freq': [0.36239782, 0.043841336, 0.075268817, 0.059322034, 0.075268817, 0.075268817, 0.043841336,
                     0.075268817, 0.007, 0.007, 0.007, 0.043841336, 0.043841336, 0.014198783, 0.043841336, 0.007,
                     0.014198783, 0.014198783, 0.028806584, 0.059322034, 0.007, 0.126126126, 0.028806584, 0.075268817,
                     0.007, 0.222493888, 0.028806584, 0.007
                 ],
            'data': 'lb_prob'
        })
        label_dist = pd.concat([train_label_dist, hpa_label_dist, probed_lb_dist], axis=0)
        fig, ax = plt.subplots(figsize=(20, 9))
        g = sns.catplot(x="label", y="freq", hue="data", data=label_dist, kind="bar", height=10, palette="muted",
                        ci=None, legend_out=False, aspect=2)
        fig.suptitle("label distribution")
        plt.savefig('{}/label_distribution.png'.format(config.plots_dir))
        plt.close('all')
        label_dist.to_csv("{}/label_dist_plotdf.csv".format(config.plots_dir), index=False)

        sns.set_style("whitegrid", {'axes.grid' : False})
        pure_train = train_csv.iloc[y_train.sum(axis=1) == 1]
        pure_hpa   = hpa_csv.iloc[y_hpa.sum(axis=1) == 1]

        for dup in range(num_duplicates):
            for label in range(28):
                print("generating sample plots for label {}".format(label))
                train_sample = pure_train.loc[pure_train.Target == str(label)]
                if train_sample.shape[0] != 0:
                    train_sample = train_sample.sample(num_samples, replace=True)
                    train_images = [train_gen[i][0] for i in train_sample.index]
                    fig, ax = plt.subplots(nrows=num_samples, ncols=4, figsize=(20, 5 * num_samples))
                    for row, image in enumerate(train_images):
                        for col, cmap in enumerate(['Reds', 'Greens', 'Blues', 'Oranges']):
                            ax[row][col].set_title(cmap)
                            ax[row][col].imshow(image[col].numpy(), cmap=cmap)
                            ax[row][col].set_yticklabels([])
                            ax[row][col].set_xticklabels([])
                    fig.suptitle('Train label {} samples'.format(label), fontsize=20)
                    plt.subplots_adjust(top=0.95)
                    plt.savefig('{}/train_label_{}_dup_{}_samples.png'.format(config.plots_dir, label, dup))
                    plt.close('all')

                hpa_sample   = pure_hpa.loc[pure_hpa.Target == str(label)]
                if hpa_sample.shape[0] != 0:
                    hpa_sample = hpa_sample.sample(num_samples, replace=True)
                    hpa_images   = [hpa_gen[i][0] for i in hpa_sample.index]
                    fig, ax = plt.subplots(nrows=num_samples, ncols=4, figsize=(20, 5 * num_samples))
                    for row, image in enumerate(hpa_images):
                        for col, cmap in enumerate(['Reds', 'Greens', 'Blues', 'Oranges']):
                            ax[row][col].set_title(cmap)
                            ax[row][col].imshow(image[col].numpy(), cmap=cmap)
                            ax[row][col].set_yticklabels([])
                            ax[row][col].set_xticklabels([])
                    fig.suptitle('HPA label {} samples'.format(label), fontsize=20)
                    plt.subplots_adjust(top=0.95)
                    plt.savefig('{}/hpa_label_{}_dup_{}_samples.png'.format(config.plots_dir, label, dup))
                    plt.close('all')

    import sys
    if len(sys.argv) > 1:
        command = str(sys.argv[1])
        if command == 'speed':
            num_workers = int(sys.argv[2]) if len(sys.argv) == 3 else 8
            speed_test(num_workers)
        if command == 'augment':
            augmentation_inspection()
        if command == 'eda':
            num_samples    = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
            num_duplicates = int(sys.argv[3]) if len(sys.argv) >= 4 else 1
            eda(num_samples)
