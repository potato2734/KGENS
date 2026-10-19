#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import gc
import torch
from itertools import chain
import torch.nn as nn
import torch.nn.functional as F
from scatter_ops import scatter_mean, scatter_softmax, scatter_sum

from torch.utils.data import DataLoader

from dataloader import TestDataset

from relation_cardinality_metrics import calculate_relation_cardinality_metrics

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False, ignore_scoring_margin=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        if ignore_scoring_margin:
            self.gamma = nn.Parameter(
                torch.Tensor([0.0]),
                requires_grad=False
            )

        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')


    def forward(self, sample,mode='single',ss_torch_index=None):

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
        elif mode == 'head-batch':
            tail_part, head_part = sample
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1))
            if ss_torch_index is None:
                batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
                head = head.view(batch_size, negative_sample_size, -1)
            else:
                head = head.unsqueeze(1)
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1))
            if ss_torch_index is None:
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
                tail = tail.view(batch_size, negative_sample_size, -1)
            else:
                tail = tail.unsqueeze(1)
        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail,ss_torch_index, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail,ss_torch_index, mode):
        if mode == 'head-batch':
            if ss_torch_index is not None:
                score = head + (relation - tail)[ss_torch_index]
            else:
                score = head + (relation - tail)
        elif mode == 'tail-batch':
            if ss_torch_index is not None:
                score = (head + relation)[ss_torch_index] - tail
            else:
                score = (head + relation) - tail
        else:
            score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail,ss_torch_index, mode):
        if mode == 'head-batch':
            if ss_torch_index is not None:
                score = head * (relation * tail)[ss_torch_index]
            else:
                score = head * (relation * tail)
        elif mode == 'tail-batch':
            if ss_torch_index is not None:
                score = (head * relation)[ss_torch_index] * tail
            else:
                score = (head * relation) * tail
        else:
            score = (head * relation) * tail
        score = self.gamma.item() + score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail,ss_torch_index, mode):

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)


        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            if ss_torch_index is not None:
                score = re_head * re_score[ss_torch_index] + im_head * im_score[ss_torch_index]
            else:
                score = re_head * re_score + im_head * im_score
        elif mode == 'tail-batch':
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            if ss_torch_index is not None:
                score = re_score[ss_torch_index] * re_tail + im_score[ss_torch_index] * im_tail
            else:
                score = re_score * re_tail + im_score * im_tail
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail
        score = self.gamma.item() + score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail,ss_torch_index, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)


        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)


        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail

            if ss_torch_index is not None:
                re_score = re_score[ss_torch_index] - re_head
                im_score = im_score[ss_torch_index] - im_head
            else:
                re_score = re_score - re_head
                im_score = im_score - im_head
        elif mode == 'tail-batch':
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation

            if ss_torch_index is not None:
                re_score = re_score[ss_torch_index] - re_tail
                im_score = im_score[ss_torch_index] - im_tail
            else:
                re_score = re_score - re_tail
                im_score = im_score - im_tail
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score


    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846


        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, alloc_max, alloc_min):

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, hr_freq, tr_freq, mode, alloc_count = next(train_iterator)


        if not args.step % 100:
            gib = 1024 ** 3

            logging.info(f"ALPHA: {args.adversarial_temperature}")
            logging.info(
                "CUDA CURRENT ALLOCATED: %.2f GiB",
                torch.cuda.memory_allocated() / gib,
            )
            logging.info(
                "CUDA CURRENT RESERVED : %.2f GiB",
                torch.cuda.memory_reserved() / gib,
            )
            logging.info(
                "CUDA PEAK ALLOCATED   : %.2f GiB",
                torch.cuda.max_memory_allocated() / gib,
            )
            logging.info(
                "CUDA PEAK RESERVED    : %.2f GiB",
                torch.cuda.max_memory_reserved() / gib,
            )

        uniform_allocation = (
            alloc_min == args.negative_sample_size
            and alloc_max == args.negative_sample_size
            and 0 == args.square
            and 0 == args.lam
        )

        use_kge_subsampling = (
            (('[!c1]' in args.save_path and 'FB15k237' in args.save_path)
            and any(mn in args.save_path for mn in ('ComplEx', 'DistMult')))
            or
            (('[!c1]' in args.save_path and 'wn18rr' in args.save_path)
            and any(mn in args.save_path for mn in ('ComplEx', 'DistMult')))
            or
            (('[!c1]' in args.save_path and 'YAGO3-10' in args.save_path)
            and any(mn in args.save_path for mn in ('TransE','ComplEx', 'DistMult')))
            or
            (('[!c2]' in args.save_path and 'YAGO3-10' in args.save_path)
            and any(mn in args.save_path for mn in ('ComplEx', 'DistMult')))
        )

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            if not uniform_allocation or use_kge_subsampling:
                hr_freq = hr_freq.cuda()
                tr_freq = tr_freq.cuda()
            if not uniform_allocation:
                alloc_count = alloc_count.cuda()

        if use_kge_subsampling:
            subsampling_weight = torch.sqrt(
                1.0 / (hr_freq + tr_freq)
            )

        if uniform_allocation:
            batch_size = positive_sample.size(0)
            negative_sample = negative_sample.view(
                batch_size,
                args.negative_sample_size,
            )
            negative_score = model(
                (positive_sample, negative_sample),
                mode=mode,
            )

            args.umax += args.negative_sample_size
            args.umin += args.negative_sample_size
        else:
            ss_torch_index = torch.repeat_interleave(
                torch.arange(alloc_count.numel(), device=negative_sample.device),
                alloc_count.flatten().long(),
            )

            total_negative_sample_size = torch.sum(alloc_count)

            args.umax += alloc_count.max().item()
            positive_alloc_count = alloc_count[alloc_count > 0]
            args.lzero += (total_negative_sample_size - len(ss_torch_index)).item()
            args.umin += positive_alloc_count.min().item() if positive_alloc_count.numel() > 0 else 0

            negative_score = model((positive_sample, negative_sample), mode, ss_torch_index)

        positive_score = model(positive_sample)

        batch_size = positive_score.shape[0]

        positive_score = F.logsigmoid(
            positive_score
        ).squeeze(dim=1)

        if use_kge_subsampling:
            positive_sample_loss = -(
                subsampling_weight * positive_score
            ).sum() / subsampling_weight.sum()
        else:
            positive_sample_loss = -positive_score.mean()

        if args.negative_adversarial_sampling:
            if uniform_allocation:
                negative_score = (
                    F.softmax(
                        args.adversarial_temperature * negative_score,
                        dim=1,
                    ).detach()
                    * F.logsigmoid(-negative_score)
                ).sum(dim=1)
                if use_kge_subsampling:
                    negative_sample_loss = -(
                        subsampling_weight * negative_score
                    ).sum() / subsampling_weight.sum()
                else:
                    negative_sample_loss = -negative_score.mean()
            else:
                neg_score_flat = negative_score.squeeze(-1)

                alloc_count_flat = alloc_count.reshape(-1)

                alloc_count_expanded = (
                    alloc_count_flat[ss_torch_index]
                    .to(dtype=neg_score_flat.dtype)
                )

                eps = 1e-8
                num_groups = alloc_count_flat.numel()


                base_logits = (
                    args.adversarial_temperature * neg_score_flat
                )


                with torch.no_grad():


                    alloc_float = alloc_count_flat.to(
                        dtype=neg_score_flat.dtype
                    ).clamp_min(1.0)


                    neg_sum = scatter_sum(
                        neg_score_flat,
                        ss_torch_index,
                        dim=0,
                        dim_size=num_groups,
                    )

                    neg_mean = (
                        neg_sum / alloc_float
                    )

                    neg_mean_expanded = (
                        neg_mean[ss_torch_index]
                    )

                    centered_score = (
                        neg_score_flat - neg_mean_expanded
                    )


                    neg_var = scatter_sum(
                        centered_score.pow(2),
                        ss_torch_index,
                        dim=0,
                        dim_size=num_groups,
                    )

                    neg_var = (
                        neg_var / alloc_float
                    )

                    neg_std = torch.sqrt(
                        neg_var + eps
                    )

                    neg_std_expanded = (
                        neg_std[ss_torch_index]
                    )


                    z_score = (
                        centered_score / neg_std_expanded
                    )


                    shape_prob = scatter_softmax(
                        z_score,
                        ss_torch_index,
                        dim=0,
                    )


                    entropy_term = -(
                        shape_prob
                        * torch.log(
                            shape_prob.clamp_min(eps)
                        )
                    )

                    entropy = scatter_sum(
                        entropy_term,
                        ss_torch_index,
                        dim=0,
                        dim_size=num_groups,
                    )


                    max_entropy = torch.log(
                        alloc_float
                    )

                    shape_concentration = (
                        1.0
                        - entropy
                        / max_entropy.clamp_min(eps)
                    )

                    shape_concentration = (
                        shape_concentration
                        .clamp(min=0.0, max=1.0)
                    )


                    shape_concentration = torch.where(
                        alloc_count_flat > 1,
                        shape_concentration,
                        torch.zeros_like(
                            shape_concentration
                        ),
                    )


                    shape_concentration_expanded = (
                        shape_concentration[ss_torch_index]
                    )

                    dynamic_logits = (
                        args.adversarial_temperature
                        * shape_concentration_expanded
                        * z_score
                    )


                adv_logits = (
                    (1.0 - args.lam) * base_logits
                    + args.lam * dynamic_logits
                )


                softmax_weight = scatter_softmax(
                    adv_logits,
                    ss_torch_index,
                    dim=0,
                ).detach()


                generalized_weight = (
                    alloc_count_expanded.pow(args.beta)
                    * softmax_weight
                ).detach()


                negative_log_prob = F.logsigmoid(
                    -neg_score_flat
                )

                if use_kge_subsampling:
                    generalized_weight = (
                        generalized_weight
                        * subsampling_weight[ss_torch_index]
                    )
                    negative_sample_loss = -(
                        generalized_weight
                        * negative_log_prob
                    ).sum() / generalized_weight.sum()
                else:
                    negative_sample_loss = -(
                        generalized_weight
                        * negative_log_prob
                    ).sum() / generalized_weight.sum().clamp_min(1.0)

        else:
            if use_kge_subsampling:
                negative_log_prob = F.logsigmoid(-negative_score)
                if uniform_allocation:
                    if args.sum_ns_loss:
                        negative_score = negative_log_prob.sum(dim=1)
                    else:
                        negative_score = negative_log_prob.mean(dim=1)
                else:
                    negative_log_prob = negative_log_prob.squeeze(-1)
                    if args.sum_ns_loss:
                        negative_score = scatter_sum(
                            negative_log_prob,
                            ss_torch_index,
                            dim=0,
                            dim_size=batch_size,
                        )
                    else:
                        negative_score = scatter_mean(
                            negative_log_prob,
                            ss_torch_index,
                            dim=0,
                            dim_size=batch_size,
                        )
                negative_sample_loss = -(
                    subsampling_weight * negative_score
                ).sum() / subsampling_weight.sum()
            else:
                if args.sum_ns_loss:
                    negative_score = F.logsigmoid(
                        -negative_score
                    ).sum(dim=1)
                else:
                    negative_score = F.logsigmoid(
                        -negative_score
                    ).mean(dim=1)

                negative_sample_loss = -negative_score.mean()

        loss = (
            positive_sample_loss
            + negative_sample_loss
        ) / 2

        if args.regularization != 0.0:
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 +
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
            'max_sample': args.umax/(1+args.step),
            'min_sample': args.umin/(1+args.step),
            'zero_sample': args.lzero/(1+args.step),
        }

        return log

    @staticmethod
    @torch.inference_mode()
    def test_step(model, test_triples, all_true_triples, args,):
        was_training = model.training
        model.eval()
        info4eval = []

        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                "head-batch",
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn,
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                "tail-batch",
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn,
        )

        test_dataloader_list = [
            test_dataloader_head,
            test_dataloader_tail,
        ]

        test_iterator = chain.from_iterable(test_dataloader_list)

        metric_sums = {
            "MRR": 0.0,
            "MR": 0.0,
            "HITS@1": 0.0,
            "HITS@3": 0.0,
            "HITS@10": 0.0,
        }
        num_queries = 0

        for batch in test_iterator:
            positive_sample, negative_sample, filter_bias, mode = batch

            if args.cuda:
                positive_sample = positive_sample.cuda(non_blocking=True)
                negative_sample = negative_sample.cuda(non_blocking=True)
                filter_bias = filter_bias.cuda(non_blocking=True)

            score = model(
                (positive_sample, negative_sample),
                mode=mode,
            )

            score.add_(filter_bias)

            argsort = torch.argsort(
                score,
                dim=1,
                descending=True,
            )

            if mode == "head-batch":
                positive_arg = positive_sample[:, 0].clone()
            elif mode == "tail-batch":
                positive_arg = positive_sample[:, 2].clone()
            else:
                raise ValueError(f"Unsupported test mode: {mode}")

            batch_size = positive_sample.size(0)

            for i in range(batch_size):
                i_scores = score[i, :]
                upper = (i_scores > i_scores[positive_arg[i]]).sum()
                ties = (i_scores == i_scores[positive_arg[i]]).sum()
                ranking = float(upper + (ties + 1) / 2)

                metric_sums["MRR"] += 1.0 / ranking
                metric_sums["MR"] += ranking
                metric_sums["HITS@1"] += float(ranking <= 1)
                metric_sums["HITS@3"] += float(ranking <= 3)
                metric_sums["HITS@10"] += float(ranking <= 10)
                num_queries += 1
                info4eval.append((tuple(positive_sample[i].tolist()), mode, ranking))

                del ranking
            del score
            del filter_bias
            del negative_sample
            del positive_sample
            del batch

            del argsort
            del positive_arg
            del mode

        if num_queries == 0:
            raise RuntimeError("No test queries were evaluated.")

        metrics = {
            metric: total / num_queries
            for metric, total in metric_sums.items()
        }

        metrics.update(
            calculate_relation_cardinality_metrics(args.data_path, info4eval)
        )

        del test_iterator
        del test_dataloader_list
        del test_dataloader_head
        del test_dataloader_tail
        del metric_sums

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if was_training:
            model.train()

        return metrics
