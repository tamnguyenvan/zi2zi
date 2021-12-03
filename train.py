# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os

from model.dataset import TrainDataProvider

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
import argparse

from model.unet import UNet
import horovod.tensorflow as hvd


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_id', dest='experiment_id', type=int, default=0,
                    help='sequence id for the experiments you prepare to run')
parser.add_argument('--image_size', dest='image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', dest='L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', dest='Lconst_penalty', type=int, default=15, help='weight for const loss')
parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', dest='Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', dest='embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch each process (CPU or GPU)')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--schedule', dest='schedule', type=int, default=10, help='number of epochs to half learning rate')
parser.add_argument('--resume', dest='resume', type=int, default=1, help='resume from previous training')
parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=int, default=0,
                    help="freeze encoder weights during training")
parser.add_argument('--fine_tune', dest='fine_tune', type=str, default=None,
                    help='specific labels id to be fine tuned')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')
parser.add_argument('--sample_steps', dest='sample_steps', type=int, default=10,
                    help='number of batches in between two samples are drawn from validation set')
parser.add_argument('--checkpoint_steps', dest='checkpoint_steps', type=int, default=500,
                    help='number of batches in between two checkpoints')
parser.add_argument('--flip_labels', dest='flip_labels', type=int, default=None,
                    help='whether flip training data labels or not, in fine tuning')
args = parser.parse_args()

def main(_):
    hvd.init()
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = str(hvd.local_rank())
    model = UNet(args.experiment_dir, batch_size=args.batch_size, experiment_id=args.experiment_id,
                    input_width=args.image_size, output_width=args.image_size, embedding_num=args.embedding_num,
                    embedding_dim=args.embedding_dim, L1_penalty=args.L1_penalty, Lconst_penalty=args.Lconst_penalty,
                    Ltv_penalty=args.Ltv_penalty, Lcategory_penalty=args.Lcategory_penalty)
    if args.flip_labels:
        model.build_model(is_training=True, inst_norm=args.inst_norm, no_target_source=True)
    else:
        model.build_model(is_training=True, inst_norm=args.inst_norm)

    # global_step = tf.train.get_or_create_global_step()
    hooks = [
        hvd.BroadcastGlobalVariablesHook(0),
    ]
    # checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None

    # with tf.Session(config=config) as sess:
    lr = args.lr
    fine_tune_list = None
    if args.fine_tune:
        ids = args.fine_tune.split(",")
        fine_tune_list = set([int(i) for i in ids])

    # filter by one type of labels
    data_provider = TrainDataProvider(model.data_dir, filter_by=fine_tune_list)
    total_batches = data_provider.compute_total_batch_num(model.batch_size)
    val_batch_iter = data_provider.get_val_iter(model.batch_size)

    g_vars, d_vars = model.retrieve_trainable_vars(freeze_encoder=args.freeze_encoder)
    input_handle, loss_handle, _, summary_handle = model.retrieve_handles()
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    d_optimizer = tf.train.AdamOptimizer(learning_rate * hvd.size(), beta1=0.5)
    d_op = d_optimizer.minimize(loss_handle.d_loss, var_list=d_vars)
    g_optimizer = tf.train.AdamOptimizer(learning_rate * hvd.size(), beta1=0.5)
    g_op = g_optimizer.minimize(loss_handle.g_loss, var_list=g_vars)

    d_optimizer = hvd.DistributedOptimizer(d_optimizer)
    g_optimizer = hvd.DistributedOptimizer(g_optimizer)

    with tf.train.MonitoredTrainingSession(
                                           config=config,
                                           hooks=hooks) as sess:
        model.register_session(sess)
        model.train(d_op=d_op, g_op=g_op, input_handle=input_handle, loss_handle=loss_handle,
                    summary_handle=summary_handle, data_provider=data_provider,
                    total_batches=total_batches, val_batch_iter=val_batch_iter, learning_rate=learning_rate,
                    lr=args.lr, epoch=args.epoch, resume=args.resume,
                    schedule=args.schedule, freeze_encoder=args.freeze_encoder, fine_tune=fine_tune_list,
                    sample_steps=args.sample_steps, checkpoint_steps=args.checkpoint_steps,
                    flip_labels=args.flip_labels)


if __name__ == '__main__':
    tf.app.run()
