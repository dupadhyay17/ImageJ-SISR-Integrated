# uncompyle6 version 3.6.7
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.1 (default, Apr  4 2017, 09:40:21) 
# [GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.38)]
# Embedded file name: .\train_v5_Chris.py
# Compiled at: 2018-10-28 16:53:24
# Size of source mod 2**32: 9355 bytes
from configobj import ConfigObj
from time import time, sleep
from datetime import datetime
import tensorflow as tf
from tqdm import tqdm
import glob, sys, numpy as np, random
from PIL import Image
import batch_utils_same_image as batch_utils, unet as network, ops, os, sys
from tf_ssim import structural_similarity
import parameters, warnings
warnings.filterwarnings('ignore')
dict = parameters.import_parameters('PARAMETERS.txt')
NUM_EPOCHS = int(dict['num_epochs'])
LEARNING_RATE_G = float(dict['learning_rate_g'])
LEARNING_RATE_D = float(dict['learning_rate_d'])
MSE_WEIGHT = float(dict['mse_weight'])
SSIM_WEIGHT = float(dict['ssim_weight'])
BLOCK_SIZE = int(dict['block_size'])
BATCH_SIZE = int(dict['batch_size'])
TRAIN_IMAGE_DIR = str(dict['train_image_dir'])
VALID_IMAGE_DIR = str(dict['valid_image_dir'])
OUTPUT_MODEL_DIR = str(dict['output_model_dir'])
NUM_GEN = int(dict['num_gen'])

def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()
    tc.is_training, vc.is_training = (True, False)
    tc.batch_size, vc.batch_size = BATCH_SIZE, BATCH_SIZE
    tc.image_size, vc.image_size = BLOCK_SIZE, BLOCK_SIZE
    tc.n_channels, vc.n_channels = (64, 64)
    tc.n_threads, vc.n_threads = (1, 1)
    tc.n_blocks, vc.n_blocks = (5, 5)
    tc.n_levels, vc.n_levels = (4, 4)
    tc.checkpoint = 200
    tc.inde_iters = 1
    tc.q_limit, vc.q_limit = (1000, 1000)
    tc.lamda, vc.lamda = MSE_WEIGHT, MSE_WEIGHT
    tc.nu, vc.nu = SSIM_WEIGHT, SSIM_WEIGHT
    tc.alpha, vc.alpha = (1, 1)
    return (
     tc, vc)


def inference(batch_loader, config, get_loss=True):
    with tf.device('/cpu:0'):
        x, y = batch_loader.get_batch()
    with tf.device('/gpu:0'):
        with tf.compat.v1.variable_scope('Generator', reuse=(tf.compat.v1.AUTO_REUSE)):
            output = network.Generator(x, config).output
        with tf.compat.v1.variable_scope('Discriminator', reuse=(tf.compat.v1.AUTO_REUSE)):
            logits_fake = network.Discriminator(output, config).output
            logits_real = network.Discriminator(y, config).output
        if not get_loss:
            return output
        else:
            with tf.name_scope('G_loss'):
                dis_loss = tf.reduce_mean(-tf.math.log(logits_fake + 1e-08))
                tf.summary.scalar('dis_loss', dis_loss)
                mse_loss = tf.reduce_mean(tf.square(y - output))
                tf.summary.scalar('mse_loss', mse_loss)
                ssim_loss = tf.reduce_mean(-tf.math.log((structural_similarity(y, output) + 1) / 2))
                tf.summary.scalar('ssim_loss', ssim_loss)
                g_loss = config.lamda * mse_loss + config.alpha * dis_loss + config.nu * ssim_loss
                tf.summary.scalar('total_loss', g_loss)
            with tf.name_scope('D_loss'):
                fake_loss = tf.reduce_mean(-tf.math.log(1.0 - logits_fake + 1e-08))
                tf.summary.scalar('fake_loss', fake_loss)
                real_loss = tf.reduce_mean(-tf.math.log(logits_real + 1e-08))
                tf.summary.scalar('real_loss', real_loss)
                d_loss = (real_loss + fake_loss) / 2.0
                tf.summary.scalar('total_loss', d_loss)
            return (
             g_loss, dis_loss, mse_loss, ssim_loss, d_loss, real_loss, fake_loss)


def train_g(g_loss):
    with tf.device('/gpu:0'):
        with tf.name_scope('G_Optimizer'):
            gen_var_list = tf.compat.v1.get_collection((tf.compat.v1.GraphKeys.GLOBAL_VARIABLES), scope='Generator')
            gen_optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE_G, name='Adam', beta1=0.9)
            gvs = gen_optimizer.compute_gradients(g_loss, var_list=gen_var_list, gate_gradients=(tf.compat.v1.train.AdamOptimizer.GATE_OP))
            capped_gvs = [(tf.clip_by_norm(grad, 0.5), var) for grad, var in gvs]
            train_op = gen_optimizer.apply_gradients(capped_gvs)
    return train_op


def train_d(d_loss):
    with tf.device('/gpu:0'):
        with tf.name_scope('D_Optimizer'):
            dis_var_list = tf.compat.v1.get_collection((tf.compat.v1.GraphKeys.GLOBAL_VARIABLES), scope='Discriminator')
            dis_optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE_D, name='Adam', beta1=0.9)
            gvs = dis_optimizer.compute_gradients(d_loss, var_list=dis_var_list, gate_gradients=(tf.compat.v1.train.AdamOptimizer.GATE_GRAPH))
            capped_gvs = [(tf.clip_by_norm(grad, 0.5), var) for grad, var in gvs]
            train_op = dis_optimizer.apply_gradients(capped_gvs)
    return train_op


results_file = open('results.txt', 'w')
if __name__ == '__main__':
    train_images = glob.glob(TRAIN_IMAGE_DIR + '*.tif')
    valid_images = glob.glob(VALID_IMAGE_DIR + '*.tif')
    print('Number of training images:' + str(len(train_images)), '; Number of validation images:' + str(len(valid_images)))
    random.shuffle(train_images)
    train_config, valid_config = init_parameters()
    patch_size = train_config.image_size
    valid_config.q_limit = 1000
    with tf.Graph().as_default():
        input_ = tf.compat.v1.placeholder((tf.float32), shape=[None, patch_size, patch_size, 1], name='batchLoader_input')
        label_ = tf.compat.v1.placeholder((tf.float32), shape=[None, patch_size, patch_size, 1], name='batchLoader_label')
        train_bl = batch_utils.TrainBatchLoader(train_images, input_, label_, train_config)
        valid_bl = batch_utils.ValidBatchLoader(valid_images, input_, label_, valid_config)
        G_loss, _, _, _, D_loss, D_real_loss, D_fake_loss = inference(train_bl, train_config, get_loss=True)
        valid_G_loss, valid_DIS_loss, valid_MSE_loss, valid_SSIM_loss, valid_D_loss, valid_D_real_loss, valid_D_fake_loss = inference(valid_bl, valid_config, get_loss=True)
        G_train_step = train_g(G_loss)
        D_train_step = train_d(D_loss)
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as (sess):
            merged = tf.compat.v1.summary.merge_all()
            now = datetime.now()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(max_to_keep=0)
            train_bl.start_threads(sess, n_threads=(train_config.n_threads))
            valid_bl.start_threads(sess, n_threads=(valid_config.n_threads))
            for i in range(15):
                sleep(1)

            print('Queue size: Training = %d, Validation = %d' % (
             train_bl.queue.size().eval(), valid_bl.queue.size().eval()))
            n_eval_steps = valid_config.q_limit // valid_config.batch_size
            check = train_config.checkpoint
            min_loss = float('inf')
            start_time = time()
            numGen = NUM_GEN
            for x in tqdm((range(1, NUM_EPOCHS + 1)), unit='epoch', initial=1):
                d_fake_loss, d_real_loss, g_loss = (0, 0, 0)
                for i in range(check):
                    for jj in range(numGen):
                        _, b = sess.run([G_train_step, G_loss])
                        g_loss += b

                    _, a1, a2 = sess.run([D_train_step, D_fake_loss, D_real_loss])
                    d_fake_loss += a1
                    d_real_loss += a2
                    sys.stdout.flush()

                res = np.mean([sess.run([valid_G_loss, valid_MSE_loss, valid_SSIM_loss, valid_DIS_loss, valid_D_fake_loss, valid_D_real_loss]) for _ in range(n_eval_steps)], axis=0)
                format_str = 'epoch = %d Valid_G_loss = %.3f Valid_G_mse_loss = %.3f Valid_G_ssim_loss =%.4f Valid_G_dis_loss = %.3f Valid_D_fake_loss = %.3f Valid_D_real_loss: %.3f Train_G_loss = %.3f Train_D_fake = %.3f Train_D_real = %.3f'
                text = format_str % (
                 x, res[0], res[1], res[2], res[3], res[4], res[5], g_loss / (check * numGen),
                 d_fake_loss / check, d_real_loss / check)
                text = text + ' time = ' + str(int(time() - start_time)) + ' sec'
                results_file.write(text + '\n')
                results_file.flush()
                if res[0] < min_loss and x >= 100:
                    min_loss = res[0]
                    saver.save(sess, OUTPUT_MODEL_DIR + 'best_model')
# okay decompiling train_v5_Chris.pyc
