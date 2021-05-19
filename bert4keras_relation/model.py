# -*- coding: utf-8 -*-
# @Time    : 2021/5/17 16:24
# @Author  : zxf
from bert4keras.layers import Loss
# from bert4keras.backend import K, batch_gather
from bert4keras.backend import K
from bert4keras.backend import batch_gather
from bert4keras.layers import LayerNormalization
from bert4keras.models import build_transformer_model
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model


def extract_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征"""
    output, subject_ids = inputs
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]


class TotalLoss(Loss):
    """
    subject_loss与object_loss之和，都是二分类交叉熵
    """
    def compute_loss(self, inputs, mask=None):
        subject_labels, object_labels = inputs[:2]
        subject_preds, object_preds, _ = inputs[2:]
        if mask[4] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[4], K.floatx())
        # sujuect部分loss
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
        subject_loss = K.mean(subject_loss, 2)
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        # object部分loss
        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        # 总的loss
        return subject_loss + object_loss


class Bert4kerasRelationExtract(object):
    def __init__(self, config_path, checkpoint_path, relation_nums):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.relation_nums = relation_nums
        # self.predicate2id = predicate2id
        # 加载预训练模型
        self.bert = build_transformer_model(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            return_keras_model=False,
        )

    def get_model(self):
        subject_labels = Input(shape=(None, 2), name='Subject-Labels')
        subject_ids = Input(shape=(2,), name='Subject-Ids')
        object_labels = Input(shape=(None, self.relation_nums, 2), name='Object-Labels')

        # # 加载预训练模型
        # self.bert = build_transformer_model(
        #     config_path=self.config_path,
        #     checkpoint_path=self.checkpoint_path,
        #     return_keras_model=False,
        # )

        # 预测subject
        output = Dense(
            units=2, activation='sigmoid', kernel_initializer=self.bert.initializer
        )(self.bert.model.output)
        subject_preds = Lambda(lambda x: x ** 2)(output)

        subject_model = Model(self.bert.model.inputs, subject_preds)

        # 传入subject，预测object
        # 通过Conditional Layer Normalization将subject融入到object的预测中
        output = self.bert.model.layers[-2].get_output_at(-1)  # 自己想为什么是-2而不是-1
        subject = Lambda(extract_subject)([output, subject_ids])
        output = LayerNormalization(conditional=True)([output, subject])
        output = Dense(
            units=self.relation_nums * 2,
            activation='sigmoid',
            kernel_initializer=self.bert.initializer
        )(output)
        output = Lambda(lambda x: x ** 4)(output)
        object_preds = Reshape((-1, self.relation_nums, 2))(output)

        object_model = Model(self.bert.model.inputs + [subject_ids], object_preds)

        subject_preds, object_preds = TotalLoss([2, 3])([
            subject_labels, object_labels, subject_preds, object_preds,
            self.bert.model.output
        ])

        # 训练模型
        train_model = Model(
            self.bert.model.inputs + [subject_labels, subject_ids, object_labels],
            [subject_preds, object_preds]
        )
        return subject_model, object_model, train_model