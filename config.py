from easydict import EasyDict as edict
import json

config = edict()

config.model = edict()
config.model.result_path = "samples"
config.model.checkpoint_path = "checkpoint"
config.model.log_path = "log"
config.model.scale = 4
config.model.resblock_depth  = 10
config.model.recursive_depth = 1

config.valid = edict()
config.valid.hr_folder_path = '/media/zhehu/DATA/Public_Dataset/NTIRE2017/DIV2K_valid_HR/'
config.valid.lr_folder_path = '/media/zhehu/DATA/Public_Dataset/NTIRE2017/DIV2K_valid_LR_bicubic/X4/'

config.train = edict()
config.train.hr_folder_path = '/media/zhehu/DATA/Public_Dataset/NTIRE2017/DIV2K_train_HR/'
config.train.lr_folder_path = '/media/zhehu/DATA/Public_Dataset/NTIRE2017/DIV2K_train_LR_bicubic/X4/'
config.train.batch_size = 4 # use large number if you have enough memory
config.train.in_patch_size = 64
config.train.out_patch_size = config.model.scale * config.train.in_patch_size
config.train.batch_size_each_folder = 30
config.train.log_write = False
config.train.lr_init = 5*1.e-6
config.train.lr_decay = 0.5
config.train.decay_iter = 10
config.train.beta1 = 0.90
config.train.n_epoch = 300
config.train.dump_intermediate_result = True

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
