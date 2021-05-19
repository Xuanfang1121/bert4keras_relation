# -*- coding: utf-8 -*-
# @Time    : 2021/5/17 17:07
# @Author  : zxf
import json

from utils import load_data
from utils import Evaluator
from utils import data_generator
from bert4keras.tokenizers import Tokenizer
from model import Bert4kerasRelationExtract
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average

# model params
epoch = 3
maxlen = 128
batch_size = 2
config_path = 'D:/spyder/pretrain_model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'D:/spyder/pretrain_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'D:/spyder/pretrain_model/chinese_L-12_H-768_A-12/vocab.txt'


def main():
    # 加载数据集
    train_data = load_data('./data/baidu_relation_DuLE_1.0_demo/train_data_demo.json')
    valid_data = load_data('./data/baidu_relation_DuLE_1.0_demo/dev_data_demo.json')
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
    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    # get model
    subject_model, object_model, train_model = Bert4kerasRelationExtract(config_path, checkpoint_path,
                                                                         len(predicate2id)).get_model()

    AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
    # optimizer = AdamEMA(learning_rate=1e-5)
    optimizer = AdamEMA(lr=1e-5)
    train_model.compile(optimizer=optimizer)

    train_generator = data_generator(train_data, batch_size, tokenizer, maxlen, predicate2id)
    evaluator = Evaluator(optimizer, valid_data, train_model, maxlen,
                          tokenizer, subject_model, object_model, id2predicate)

    train_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        callbacks=[evaluator]
    )


if __name__ == "__main__":
    main()