# -*- coding: utf-8 -*-
# @Time    : 2021/5/17 16:24
# @Author  : zxf
import os
import json

from utils import SPO
from utils import extract_spoes
from bert4keras.tokenizers import Tokenizer
from model import Bert4kerasRelationExtract


maxlen = 128
batch_size = 8
config_path = 'D:/spyder/pretrain_model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'D:/spyder/pretrain_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'D:/spyder/pretrain_model/chinese_L-12_H-768_A-12/vocab.txt'


def predict():
    predicate2id, id2predicate = {}, {}

    with open('./data/baidu_relation_DuLE_2.0/schema.json', "r",
              encoding="utf-8") as f:
        for l in f:
            l = json.loads(l)
            if l['predicate'] not in predicate2id:
                id2predicate[len(predicate2id)] = l['predicate']
                predicate2id[l['predicate']] = len(predicate2id)
    print("predicate2id :", predicate2id)
    print("relation number: ", len(predicate2id))

    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    # get model
    subject_model, object_model, train_model = Bert4kerasRelationExtract(config_path, checkpoint_path,
                                                                         len(predicate2id)).get_model()
    train_model.load_weights('./models/baidu_relation_DuLE_2.0/best_model.weights')
    # while True:
        # text = input("input sentence, please:")
    result = []
    i = 0
    with open("./data/baidu_relation_DuLE_2.0/test1_data.json", "r", encoding="utf-8") as f:
        for l in f:
            l = json.loads(l)
            text = l["text"]
            R = set([SPO(spo) for spo in extract_spoes(text, maxlen, tokenizer, subject_model, object_model,
                                                       id2predicate)])
            # print(list(R))
            result.append({"text": text,
                           "bert4keras_relation": list(R)})
            i += 1
            if i % 100 == 0:
                print("predeict number: {}".format(i))

    with open("./models/baidu_relation_DuLE_2.0/test_model_predict.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    predict()