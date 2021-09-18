# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 9:26 上午
# @Author  : Michael Zhouy
import torch
import cv2
import matplotlib.pyplot as plt


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # topk返回一个元组, 元组的第一个元素为最大值, 第一个元素为最大值所在的索引
        # 这里应该是top3
        _, pred = output.topk(maxk, dim=1)
        pred = pred.t()  # 转置
        # expand_as()维度扩展为pred的维度
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # print(correct.shape)
        res = []
        for k in topk:
            # print(correct[:k].shape)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))

        # print(res)
        return res


def show_image(paths):
    plt.figure(figsize=(10, 8))
    for idx, path in enumerate(paths):
        plt.subplot(1, len(paths), idx+1)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
