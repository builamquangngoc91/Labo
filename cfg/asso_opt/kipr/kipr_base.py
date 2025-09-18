_base_ = "../base.py"
# dataset
proj_name = "kipr"
concept_root = "datasets/kipr/concepts/"
img_split_path = "datasets/kipr/splits"
img_path = "datasets/kipr/images"
concept_type = "all"
img_ext = ".png"
raw_sen_path = concept_root + "concepts_raw.npy"
concept2cls_path = concept_root + "concept2cls.npy"
cls_name_path = concept_root + "cls_names.npy"
num_cls = 3

## data loader
bs = 128
on_gpu = False

# concept select
num_concept = num_cls * 25

# weight matrix fitting
lr = 5e-6
max_epochs = 10000

# weight matrix
use_rand_init = False
init_val = 1.0
asso_act = "softmax"
use_l1_loss = False
use_div_loss = False
lambda_l1 = 0.01
lambda_div = 0.005

# CLIP Backbone
clip_model = "RN50"
