#!/usr/bin/env python
#
# Based on: https://github.com/horovod/horovod/blob/master/examples/tensorflow_mnist_eager.py
#
# ==============================================================================
# See https://github.com/horovod/horovod/blob/master/examples/tensorflow_mnist_eager.py
#
# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import horovod.tensorflow as hvd
from mpi4py import MPI
import optuna
import os
import sys

N_STEPS = 200


def objective(trial, comm):

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    tf.enable_eager_execution(config=config)

    conv_k = trial.suggest_categorical('conv_kernel', [3, 5, 7])

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(trial.suggest_int('filter1', 8, 64),
                               trial.suggest_categorical('kernel1', [3, 5, 7]),
                               activation='relu'),
        tf.keras.layers.Conv2D(trial.suggest_int('filter2', 8, 64),
                               trial.suggest_categorical('kernel2', [3, 5, 7]),
                               activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])

    lr = trial.suggest_loguniform('lr', 1e-8, 1e-2)
    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.train.RMSPropOptimizer(lr * hvd.size())

    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data(
        path=os.path.join(os.path.expanduser('~'), 'mnist.npz'))

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(mnist_labels, tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(32)

    checkpoint_dir = './checkpoints-trial{}'.format(trial.trial_id)
    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt,
                                     step_counter=step_counter)

    # Horovod: adjust number of steps based on number of GPUs.
    for (batch, (images, labels)) in enumerate(
            dataset.take(N_STEPS // hvd.size())):
        with tf.GradientTape() as tape:
            logits = mnist_model(images, training=True)
            loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        if batch == 0:
            hvd.broadcast_variables(mnist_model.variables, root_rank=0)

        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, mnist_model.variables)
        opt.apply_gradients(zip(grads, mnist_model.variables),
                            global_step=tf.train.get_or_create_global_step())

        if batch % 10 == 0 and hvd.local_rank() == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting it.
    if hvd.rank() == 0:
        checkpoint.save(checkpoint_dir)
    tf.keras.backend.clear_session()
    return loss_value


def main(_):
    # Horovod: initialize Horovod.
    hvd.init()

    study_name = sys.argv[1]
    storage_url = sys.argv[2]
    study = optuna.Study(study_name, storage_url)

    comm = MPI.COMM_WORLD
    mpi_study = optuna.integration.MPIStudy(study, comm)
    mpi_study.optimize(objective, n_trials=10)

    if comm.rank == 0:
        print(mpi_study.best_trial)


if __name__ == "__main__":
    tf.app.run()