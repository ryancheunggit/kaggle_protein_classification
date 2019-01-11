import os

class DefaultConfigs(object):
    project_root_path = "/media/ren/crucial/Projects/human-protein-atlas-image-classification/"
    num_classes       = 28
    img_width         = 512
    img_height        = 512
    channels          = 3     # originally used 4
    initial_lr        = 0.001

    def __init__(self):
        self.plots_dir       = self.project_root_path + "plots"
        self.logs_dir        = self.project_root_path + "logs"

        self.train_data_dir  = self.project_root_path + "data/train"
        self.test_data_dir   = self.project_root_path + "data/test"
        self.hpa_data_dir    = self.project_root_path + "data/external"

        self.weights_dir     = self.project_root_path + "checkpoints"
        self.best_models_dir = self.project_root_path + "checkpoints/best_models"
        self.submit_dir      = self.project_root_path + "submissions"

        self.train_csv       = self.project_root_path + "data/train.csv"
        self.test_csv        = self.project_root_path + "data/sample_submission.csv"
        self.hpa_csv         = self.project_root_path + "data/HPAv18RBGY_wodpl.csv"
        self.public_leak     = self.project_root_path + "data/TestEtraMatchingUnder_259_R14_G12_B10.csv"

config = DefaultConfigs()

for k, v in config.__dict__:
    if '_dir' in k and not os.path.exists(v):
        os.mkdir(v)