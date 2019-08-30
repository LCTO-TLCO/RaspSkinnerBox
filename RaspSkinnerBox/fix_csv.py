import pandas as pd


def fix(file_path):
    ret_val = []
    with open(file_path) as f:
        for line in f.readlines():
            if line.count(",") == 2:
                ret_val.append(line)
                continue
            # event name -> event name\n
            str = line.replace("payoff", "payoff\n").replace("reward", "reward\n")
            ret_val += str.split("\n")
    ret_val = [l.split(",") for l in ret_val]
    # 中身がない列を削除
    ret_val = [l for l in ret_val if len(l) > 0]
    # 改行が消えてない？文字列で残ってる？
    return ret_val