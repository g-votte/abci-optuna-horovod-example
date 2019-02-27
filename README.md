# ABCI Optuna Examples

This is an tutorial material to use Optuna in the [ABCI](https://abci.ai/) infrastructure (unofficial).

This tutorial describes:

- How to launch Optuna storage on an interactive node.
- How to parallelize single node ML training.
- How to parallelize multi-node, MPI-based ML training.

## Launch PostgreSQL in ABCI

```console
$ GROUP=<YOUR_GROUP>

$ qrsh -g $GROUP -l rt_C.small=1 -l h_rt=12:00:00
$ module load singularity/2.6.1
$ singularity build postgres.img docker://postgres

$ mkdir postgres_data
$ singularity run -B postgres_data:/var/lib/postgresql/data postgres.img /docker-entrypoint.sh postgres
```

The RDB URL is as follows:
```console
$ STORAGE_HOST=<HOST_WHERE_POSTGRES_IS_RUNNING>  # e.g., STORAGE_HOST=g0002
$ STORAGE_URL=postgres://postgres@$STORAGE_HOST:5432/
```

## Environment Setup

Build the Horovod image and run a container:

```console
$ module load singularity/2.6.1
$ singularity pull docker://uber/horovod:0.15.2-tf1.12.0-torch1.0.0-py3.5
$ singularity shell --nv horovod-0.15.2-tf1.12.0-torch1.0.0-py3.5.simg
```

With the container, install Python dependencies under the user directory:

```console
$ pip install --user mpi4py psycopg2-binary

# hvd.broadcast_variables is not supported in the old version of Horovod
$ pip install --user -U horovod  
```

To deal with MPI-based learning, you need to install a developing branch of Optuna, because the [MPIStudy](https://github.com/pfnet/optuna/blob/horovod-examples/optuna/integration/mpi.py#L46) class has not been merged to the master.

```console
$ pip uninstall optuna  # If you've already installed Optuna.
$ pip install --user git+https://github.com/pfnet/optuna.git@horovod-examples
```

## Distributed Optimization for Single Node Learning

Let's parallelize a simple Optuna script that optimizes a quadratic function.

Set up the RDB URL and create a study identifier:

```console
$ STORAGE_HOST=<HOST_WHERE_POSTGRES_IS_RUNNING>
$ STORAGE_URL=postgres://postgres@$STORAGE_HOST:5432/

$ STUDY_NAME=`~/.local/bin/optuna create-study --storage $STORAGE_URL`
```

Set up a shell script for qsub command, e.g.:

```console
$ echo "module load singularity/2.6.1" >> run_quadratic.sh
$ echo "singularity shell --nv horovod-0.15.2-tf1.12.0-torch1.0.0-py3.5.simg" >> run_quadratic.sh
$ echo "python abci-optuna-horovod-example/quadratic.py $STUDY_NAME $STORAGE_URL" >> run_quadratic.sh
```

You can parallelize the optimization just by submitting multiple jobs.
For example, the following commands simultaneously run three workers in a study.

```console
$ GROUP=<YOUR_GROUP>

$ qsub -g $GROUP -l rt_C.small=1 run_quadratic.sh
$ qsub -g $GROUP -l rt_C.small=1 run_quadratic.sh
$ qsub -g $GROUP -l rt_C.small=1 run_quadratic.sh
```

You can list the history of optimization as follows.
```console
$ python print_study_history.py $STUDY_NAME $STORAGE_URL
```

## Distributed Optimization for MPI-based Learning

Let's parallelize a script written in Horovod and TensorFlow.

Download MNIST data:

```console
$ wget -O ~/mnist.npz https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

Here, we'll run the example with interactive node. (You also can consolidate the following commands as a batch job.)

```console
$ GROUP=<YOUR_GROUP>
$ qrsh -g $GROUP -l rt_F=1 -l h_rt=01:00:00
```

Run a container:

```console
$ module load singularity/2.6.1
$ singularity shell --nv horovod-0.15.2-tf1.12.0-torch1.0.0-py3.5.simg
```

Create a study identifier in the container:

```console
$ GROUP=<YOUR_GROUP>
$ STORAGE_HOST=<HOST_WHERE_POSTGRES_IS_RUNNING>

$ STORAGE_URL=postgres://postgres@$STORAGE_HOST:5432/
$ STUDY_NAME=`~/.local/bin/optuna create-study --storage $STORAGE_URL`
```

To run the MPI example:

```console
$ mpirun -np 2 -bind-to none -map-by slot -- python tensorflow_mnist_eager_optuna.py $STUDY_NAME $STORAGE_URL
```

You can list the history of optimization as follows.
```console
$ python print_study_history.py $STUDY_NAME $STORAGE_URL
```

## See Also

- [Optuna Tutorial](https://optuna.readthedocs.io/en/latest/tutorial/)
- [Optuna Examples](https://github.com/pfnet/optuna/tree/master/examples)
