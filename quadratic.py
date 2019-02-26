#!/usr/bin/env python

import optuna
import sys


# Define a simple 2-dimensional objective function whose minimum value is -1 when (x, y) = (0, -1).
def objective(trial):
    x = trial.suggest_uniform('x', -100, 100)
    y = trial.suggest_categorical('y', [-1, 0, 1])
    return x**2 + y


if __name__ == "__main__":
    study_name = sys.argv[1]
    storage_url = sys.argv[2]
    study = optuna.create_study(study_name=study_name, storage=storage_url, load_if_exists=True)
    study.optimize(objective, n_trials=100)
