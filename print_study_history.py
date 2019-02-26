#!/usr/bin/env python

import optuna
import sys

study = optuna.create_study(study_name=sys.argv[1], storage=sys.argv[2], load_if_exists=True)
print(study.trials_dataframe())
