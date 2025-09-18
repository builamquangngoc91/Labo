_base_ = '../base.py'
# dataset
proj_name = "flower"
concept_root = 'datasets/flower/concepts/'
img_split_path = 'datasets/flower/splits'
img_path = 'datasets/flower/images'
concept_type = "all"
img_ext = '.jpg'
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = 102

## data loader
bs = 128
on_gpu = True

# concept select
num_concept = num_cls * 25

# weight matrix fitting
lr = 5e-6
max_epochs = 10000

# weight matrix
use_rand_init = False
init_val = 1.
asso_act = 'softmax'
use_l1_loss = False
use_div_loss = False
lambda_l1 = 0.01
lambda_div = 0.005

# CLIP Backbone
clip_model = 'ViT-L/14'