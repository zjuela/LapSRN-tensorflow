#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, random
import numpy as np
import scipy

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import *

###====================== HYPER-PARAMETERS ===========================###
batch_size = config.train.batch_size
patch_size = config.train.in_patch_size
ni = int(np.sqrt(config.train.batch_size))



def load_file_list():
    train_hr_file_list = []
    train_lr_file_list = []
    valid_hr_file_list = []
    valid_lr_file_list = []

    directory = config.train.hr_folder_path
    for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory,y))]:
        train_hr_file_list.append("%s%s"%(directory,filename))

    directory = config.train.lr_folder_path
    for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory,y))]:
        train_lr_file_list.append("%s%s"%(directory,filename))

    directory = config.valid.hr_folder_path
    for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory,y))]:
        valid_hr_file_list.append("%s%s"%(directory,filename))

    directory = config.valid.lr_folder_path
    for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory,y))]:
        valid_lr_file_list.append("%s%s"%(directory,filename))

    return train_hr_file_list,train_lr_file_list,valid_hr_file_list,valid_lr_file_list



def prepare_nn_data(hr_img_list, lr_img_list, idx_img=None):
    i = np.random.randint(len(hr_img_list)) if (idx_img is None) else idx_img

    input_image  = get_imgs_fn(lr_img_list[i])
    output_image = get_imgs_fn(hr_img_list[i])
    scale        = int(output_image.shape[0] / input_image.shape[0])
    assert scale == config.model.scale

    out_patch_size = patch_size * scale
    input_batch  = np.empty([batch_size,patch_size,patch_size,3])
    output_batch = np.empty([batch_size,out_patch_size,out_patch_size,3])

    for idx in range(batch_size):
        in_row_ind   = random.randint(0,input_image.shape[0]-patch_size)
        in_col_ind   = random.randint(0,input_image.shape[1]-patch_size)    

        input_cropped = augment_imgs_fn(input_image[in_row_ind:in_row_ind+patch_size,
                                                in_col_ind:in_col_ind+patch_size])
        input_cropped = normalize_imgs_fn(input_cropped)
        input_cropped = np.expand_dims(input_cropped,axis=0)
        input_batch[idx] = input_cropped
    
        out_row_ind    = in_row_ind * scale
        out_col_ind    = in_col_ind * scale
        output_cropped = output_image[out_row_ind:out_row_ind+out_patch_size,
                                    out_col_ind:out_col_ind+out_patch_size]
        output_cropped = normalize_imgs_fn(output_cropped)
        output_cropped = np.expand_dims(output_cropped,axis=0)
        output_batch[idx] = output_cropped

    return input_batch,output_batch



def train():
    save_dir = "%s/%s_train"%(config.model.result_path,tl.global_flag['mode'])
    checkpoint_dir = "%s"%(config.model.checkpoint_path)
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###========================== DEFINE MODEL ============================###
    t_image = tf.placeholder('float32', [batch_size, patch_size, patch_size, 3], name='t_image_input')
    t_target_image = tf.placeholder('float32', [batch_size, patch_size*config.model.scale, patch_size*config.model.scale, 3], name='t_target_image')
    t_target_image_down = tf.image.resize_images(t_target_image, size=[patch_size*2, patch_size*2], method=0, align_corners=False) 

    net_image2, net_grad2, net_image1, net_grad1 = LapSRN(t_image, is_train=True, reuse=False)
    net_image2.print_params(False)

    ## test inference
    net_image_test, net_grad_test, _, _ = LapSRN(t_image, is_train=False, reuse=True)

    ###========================== DEFINE TRAIN OPS ==========================###
    mse_loss2 = tl.cost.mean_squared_error(net_image2.outputs, t_target_image, is_mean=True)
    mse_loss1 = tl.cost.mean_squared_error(net_image1.outputs, t_target_image_down, is_mean=True)
    mse_loss  = mse_loss1 + mse_loss2 * 4
    g_vars    = tl.layers.get_variables_with_name('LapSRN', True, True)
    
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(config.train.lr_init, trainable=False)

    g_optim = tf.train.AdamOptimizer(lr_v, beta1=config.train.beta1).minimize(mse_loss, var_list=g_vars)
    
    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/params_{}.npz'.format(tl.global_flag['mode']), network=net_image2)
 
    ###========================== PRE-LOAD DATA ===========================###
    train_hr_list,train_lr_list,valid_hr_list,valid_lr_list = load_file_list()
 
    ###========================== Intermediate validation ===============================###
    sample_ind = 53
    sample_input_imgs,sample_output_imgs = prepare_nn_data(valid_hr_list,valid_lr_list,sample_ind)
    tl.vis.save_images(truncate_imgs_fn(sample_input_imgs),  [ni, ni], save_dir+'/train_sample_input.png')
    tl.vis.save_images(truncate_imgs_fn(sample_output_imgs), [ni, ni], save_dir+'/train_sample_output.png')

    ###========================== Training ====================###
    sess.run(tf.assign(lr_v, config.train.lr_init))
    print(" ** learning rate: %f" % config.train.lr_init)

    for epoch in range(config.train.n_epoch):
        ## update learning rate
        if epoch != 0 and (epoch % config.train.decay_iter == 0):
            lr_decay = config.train.lr_decay ** (epoch // config.train.decay_iter)
            lr = config.train.lr_init * lr_decay
            sess.run(tf.assign(lr_v, lr))
            print(" ** learning rate: %f" % (lr))

        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        ## load image data
        idx_list = np.random.permutation(len(train_hr_list))
        for idx_file in range(len(idx_list)):
            step_time = time.time()
            batch_input_imgs,batch_output_imgs = prepare_nn_data(train_hr_list,train_lr_list,idx_file)
            errM, _ = sess.run([mse_loss, g_optim], {t_image: batch_input_imgs, t_target_image: batch_output_imgs})
            total_mse_loss += errM
            n_iter += 1
        
        print("[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, config.train.n_epoch, time.time() - epoch_time, total_mse_loss/n_iter))

        ## save model and evaluation on sample set
        if (epoch != 0) and (epoch % 1 == 0):
            tl.files.save_npz(net_image2.all_params,  name=checkpoint_dir+'/params_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            sample_out, sample_grad_out = sess.run([net_image_test.outputs,net_grad_test.outputs], {t_image: sample_input_imgs})#; print('gen sub-image:', out.shape, out.min(), out.max())
            tl.vis.save_images(truncate_imgs_fn(sample_out), [ni, ni], save_dir+'/train_predict_%d.png' % epoch)
            tl.vis.save_images(truncate_imgs_fn(np.abs(sample_grad_out)), [ni, ni], save_dir+'/train_grad_predict_%d.png' % epoch)
            


def test(file):
    try:
        img = get_imgs_fn(file)
    except IOError:
        print('cannot open %s'%(file))
    else:
        checkpoint_dir = config.model.checkpoint_path
        save_dir = "%s/%s"%(config.model.result_path,tl.global_flag['mode'])
        input_image = normalize_imgs_fn(img)

        size = input_image.shape
        print('Input size: %s,%s,%s'%(size[0],size[1],size[2]))
        t_image = tf.placeholder('float32', [None,size[0],size[1],size[2]], name='input_image')
        net_g, _, _, _ = LapSRN(t_image, is_train=False, reuse=False)

        ###========================== RESTORE G =============================###
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_train.npz', network=net_g)

        ###======================= TEST =============================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [input_image]})
        print("took: %4.4fs" % (time.time() - start_time))
    
        tl.files.exists_or_mkdir(save_dir)
        print("[*] save images")
        tl.vis.save_image(truncate_imgs_fn(out[0,:,:,:]), save_dir+'/test_out.png')
        tl.vis.save_image(input_image, save_dir+'/test_input.png')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['train','test'], default='train', help='select mode')
    parser.add_argument('-f','--file', help='input file')
    
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    if tl.global_flag['mode'] == 'train':
        train()
    elif tl.global_flag['mode'] == 'test':
        if (args.file is None):
            raise Exception("Please enter input file name for test mode")
        test(args.file)
    else:
        raise Exception("Unknow --mode")
