# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 22:51
# @Author  : zxf
import os
import json


def ger_baidu_relation_data(data_file, output_file):
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            data.append(line)

    data = data[:200]
    with open(output_file, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    data_file = "D:/Spyder/data/nlpdata/relation_extract/baidu_relation_DuLE_1.0/dev_data.json"
    output_file = "D:/Spyder/data/nlpdata/relation_extract/baidu_relation_DuLE_1.0_demo/dev_data_demo.json"
    ger_baidu_relation_data(data_file, output_file)