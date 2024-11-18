import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.curdir))

import sqlite
from main import {{cookiecutter.model_name}}

if __name__ == "__main__":
    yaml_file = os.path.join(os.path.abspath(os.curdir), "config.yml")
    x = sqlite.populate_class(yaml_file)
    sqlite.populate_sqlite(x.database_config)
    lr = {{cookiecutter.model_name}}()
    data_train = pd.read_csv(x.database_config.path)
    train_data = lr.preprocess_data(data_train)
    lr.train(train_data)
    lr.export()