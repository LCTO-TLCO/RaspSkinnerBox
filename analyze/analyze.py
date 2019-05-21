import matplotlib as mpl
import pandas as pd
import numpy
import pyper
import os


def read_data():
    data_file = "../RaspSkinnerBox/log/no003_action.csv"
    r = pyper.R(use_pandas='True', use_numpy='True')
    header = ["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"]

    data = pd.read_csv(data_file, names=header)
    r.assign("behavior_data", data)

    print(r("source(file='reshape_data.R')"))
    dt = pd.DataFrame(r.get("dt"))

    print(dt)




if __name__ == "__main__":
    read_data()