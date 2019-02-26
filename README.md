### Environment Setup

```console
wget -O mnist.npz https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

```console
module load singularity/2.6.1
singularity pull docker://uber/horovod:0.15.2-tf1.12.0-torch1.0.0-py3.5
singularity shell --nv horovod-0.15.2-tf1.12.0-torch1.0.0-py3.5.simg
```

```console
pip install --user mpi4py psycopg2-binary
pip install --user -U horovod
pip install --user git+https://github.com/pfnet/optuna.git@horovod-examples
```

### Launch PostgreSQL

```console
GROUP=<YOUR_GROUP>
```

```console
qrsh -g $GROUP -l rt_C.small=1 -l h_rt=24:00:00
module load singularity/2.6.1
singularity build postgres.img docker://postgres
singularity run -B postgres_data:/var/lib/postgresql/data postgres.img /docker-entrypoint.sh postgres
```

### TensorFlow Eager Mode Example

```console
GROUP=<YOUR_GROUP>

qrsh -g $GROUP -l rt_F=1 -l h_rt=01:00:00
module load singularity/2.6.1
singularity shell --nv horovod-0.15.2-tf1.12.0-torch1.0.0-py3.5.simg
```

```console
GROUP=<YOUR_GROUP>
STORAGE_HOST=<HOST_WHERE_POSTGRES_IS_RUNNING>

STORAGE_URL=postgres://postgres@$STORAGE_HOST:5432/
STUDY_NAME=`.local/bin/optuna create-study --storage $STORAGE_URL`
```

```console
mpirun -np 2 -bind-to none -map-by slot -- python tensorflow_mnist_eager_optuna.py $STUDY_NAME $STORAGE_URL
```
