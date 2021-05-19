# -*- coding: utf-8 -*-
# @Time    : 2021/5/18 17:38
# @Author  : zxf
import json
import requests

from flask import Flask
from flask import jsonify
from flask import request

from utils import SPO
from utils import extract_spoes
from bert4keras.tokenizers import Tokenizer
from model import Bert4kerasRelationExtract


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

maxlen = 128
config_path = 'D:/spyder/pretrain_model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'D:/spyder/pretrain_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'D:/spyder/pretrain_model/chinese_L-12_H-768_A-12/vocab.txt'
predicate2id, id2predicate = {}, {}
with open('./data/baidu_relation_DuLE_1.0/all_50_schemas.txt', "r",
          encoding="utf-8") as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)
print("predicate2id :", predicate2id)
print("relation number: ", len(predicate2id))

tokenizer = Tokenizer(dict_path, do_lower_case=True)


@app.route("/relation", methods=['POST'])
def model_predict():
    data = json.loads(request.get_data(), encoding="utf-8")
    text = data.get("text")
    # get model
    # todo 每次请求加载一次模型，提前加载模型会报错，为什么
    subject_model, object_model, train_model = Bert4kerasRelationExtract(config_path, checkpoint_path,
                                                                         len(predicate2id)).get_model()
    train_model.load_weights('./models/baidu_relation_DuLE_1.0/best_model.weights')

    R = set([SPO(spo) for spo in extract_spoes(text, maxlen, tokenizer, subject_model, object_model,
                                               id2predicate)])
    print(list(R))
    return jsonify({"text": text,
                    "relation": list(R)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)