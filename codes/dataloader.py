#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset
import logging
from time import time
import random
import re

class TrainDataset(Dataset):
    def __init__(self, args, triples, nentity, nrelation, negative_sample_size, mode, s,):
        self.args = args
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = 6 * negative_sample_size
        self.ns_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples, 4)
        self.count_max, self.count_min = max(self.count.values()), min(self.count.values())
        self.alloc = self.allocate_negative(self.count, s, negative_sample_size)
        self.alloc_max, self.alloc_min = max(self.alloc.values()), min(self.alloc.values())
        self.p = None
        if 'binrd' in self.args.save_path:
            self.p = float(re.search(r'\[(.*?)\]', self.args.save_path).group(1))
            assert 0 <= self.p <= 1, f'p for binomial random sampling should be a probability, current value of p:{self.p}'
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        if self.mode == 'head-batch': piv_key = (tail, -relation-1)
        elif self.mode == 'tail-batch': piv_key = (head, relation)
        alloc = max(1, self.alloc[piv_key])

        if 'unird' in self.args.save_path:
            alloc = random.randint(self.alloc_min, self.alloc_max)
        elif 'binrd' in self.args.save_path:
            alloc = self.alloc_min + np.random.binomial(self.alloc_max - self.alloc_min, self.p)
        elif self.args.tau > 0:
            variance = self.args.tau * self.count[piv_key]
            std = np.sqrt(variance)

            noise = np.random.normal(
                loc=0.0,
                scale=std
            )

            alloc = self.allocate_negative_onthefly(max(1, int(round(self.count[piv_key] + noise))))
        else:
            pass

        while negative_sample_size < alloc:
            negative_sample = np.random.randint(self.nentity, size=alloc*2)
            if self.mode == 'head-batch':
                mask = np.isin(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.isin(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        if alloc: negative_sample = np.concatenate(negative_sample_list)[:alloc]
        else: negative_sample = np.array([])

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, self.count[(head, relation)], self.count[(tail, -relation-1)], self.mode, alloc

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.concat([_[1] for _ in data], dim=0)
        hr_freq = torch.Tensor([_[2] for _ in data])
        tr_freq = torch.Tensor([_[3] for _ in data])
        batch_type = data[0][4]
        alloc_count = torch.Tensor([_[5] for _ in data])
        return positive_sample, negative_sample, hr_freq, tr_freq, batch_type, alloc_count

    @staticmethod
    def count_frequency(triples, start=4):
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count

    @staticmethod
    def allocate_negative(piv2count: dict, s: float, max_nsample: int):
        alloc = dict()
        min_delta = min([v for v in piv2count.values()])
        for piv, count in piv2count.items():
            ratio = min_delta / count
            value = ratio ** s
            neg_size = int(max_nsample * value)
            alloc[piv] = neg_size

        return alloc

    def allocate_negative_onthefly(self, count: int):
        alloc = 0

        ratio = self.count_min / count
        value = ratio ** self.args.square_gamma
        alloc = int(self.ns_size * value)

        return alloc

    @staticmethod
    def get_true_head_and_tail(triples):

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1

        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
