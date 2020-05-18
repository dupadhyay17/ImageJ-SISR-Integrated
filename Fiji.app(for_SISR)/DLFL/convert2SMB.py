# uncompyle6 version 3.6.7
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.1 (default, Apr  4 2017, 09:40:21) 
# [GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.38)]
# Embedded file name: .\convert2SMB.py
# Compiled at: 2018-07-11 19:04:26
# Size of source mod 2**32: 4785 bytes
from configobj import ConfigObj
import glob, os, tensorflow as tf, numpy as np, scipy.misc, sys, ops, unet as network, parameters, time
dict = parameters.import_parameters('DLFL/PARAMETERS.txt')
OUTPUT_MODEL_DIR = str(dict['output_model_dir'])
NUM_EPOCHS = int(dict['num_epochs'])

def T(x):
    image = np.clip(x, 0, 255)
    return image


if __name__ == '__main__':
    time.sleep(2)
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            input_ = tf.placeholder((tf.float32), shape=[None, None, 1])
            label_ = tf.placeholder((tf.float32), shape=[None, None, 1])
            inp = tf.expand_dims(input_, axis=0)
            lab = tf.expand_dims(label_, axis=0)
            config = ConfigObj()
            config.is_training = False
            config.n_channels = 64
            config.n_levels = 4
            config.image_size = 1024
            with tf.variable_scope('Generator'):
                with tf.device(ops.get_available_gpus()[0]):
                    G = network.Generator(inp, config)
                    tf_output = G.output
            RUN_ONCE = True
            print('converting!')
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as (sess):
                cur_out_dir = OUTPUT_MODEL_DIR + 'best_model\\'
                tf.train.Saver().restore(sess, OUTPUT_MODEL_DIR + 'best_model')
                if RUN_ONCE:
                    tensor_info_x = tf.saved_model.utils.build_tensor_info(input_)
                    tensor_info_y = tf.saved_model.utils.build_tensor_info(tf_output)
                    signature_network = tf.saved_model.signature_def_utils.build_signature_def(inputs={'input_images': tensor_info_x},
                      outputs={'output_images': tensor_info_y},
                      method_name=(tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
                    if os.path.isdir(cur_out_dir):
                        os.system('rd /s /q ' + cur_out_dir)
                    builder = tf.saved_model.builder.SavedModelBuilder(cur_out_dir)
                    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_network},
                      clear_devices=True)
                    builder.save()
                    for x in range(1, NUM_EPOCHS + 1):
                        cur_out_dir = OUTPUT_MODEL_DIR + '{}\\'.format(x)
                        tf.train.Saver().restore(sess, OUTPUT_MODEL_DIR + '{}'.format(x))
                        if RUN_ONCE:
                            tensor_info_x = tf.saved_model.utils.build_tensor_info(input_)
                            tensor_info_y = tf.saved_model.utils.build_tensor_info(tf_output)
                            signature_network = tf.saved_model.signature_def_utils.build_signature_def(inputs={'input_images': tensor_info_x},
                              outputs={'output_images': tensor_info_y},
                              method_name=(tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
                            if os.path.isdir(cur_out_dir):
                                os.system('rd /s /q ' + cur_out_dir)
                            builder = tf.saved_model.builder.SavedModelBuilder(cur_out_dir)
                            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_network},
                              clear_devices=True)
                            builder.save()

                    RUN_ONCE = False
            print('done converting!')
# okay decompiling convert2SMB.pyc
