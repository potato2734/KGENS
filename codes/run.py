#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc
import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--default_subsampling', action='store_true')
    parser.add_argument('--uniq_based_subsampling', action='store_true')
    parser.add_argument('--freq_based_subsampling', action='store_true')
    parser.add_argument('--ignore_scoring_margin', action='store_true')

    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None,
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_false')
    parser.add_argument('-sum', '--sum_ns_loss',default=True, action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=5000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('-sg', '--square_gamma', default=0.5, type=float)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--tau', default=1.0, type=float)
    parser.add_argument('--lam', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data', type=str)
    return parser.parse_args(args)

def override_config(args):

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def save_model(model, optimizer, save_variable_list, args):

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def main(args):
    random_seeds(args.seed)

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_logger(args)

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    if 'tree' in args.data_path:
        fn_triples = read_triple(os.path.join(args.data_path, 'false_negatives.txt'), entity2id, relation2id)
        logging.info('#false negatives: %d' % len(fn_triples))

    

    all_true_triples = train_triples + valid_triples + test_triples

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        ignore_scoring_margin=args.ignore_scoring_margin,
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    g = torch.Generator()
    g.manual_seed(args.seed)

    if args.do_train:
        train_dataset_head = TrainDataset(
            args,
            train_triples,
            nentity,
            nrelation,
            args.negative_sample_size,
            'head-batch',
            args.square_gamma,
        )

        train_dataloader_head = DataLoader(
            train_dataset_head,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn,
            generator=g,
        )

        train_dataset_tail = TrainDataset(
            args,
            train_triples,
            nentity,
            nrelation,
            args.negative_sample_size,
            'tail-batch',
            args.square_gamma,
        )

        head_max_count = max([v for v in train_dataset_head.alloc.values()])
        tail_max_count = max([v for v in train_dataset_tail.alloc.values()])
        head_min_count = min([v for v in train_dataset_head.alloc.values()])
        tail_min_count = min([v for v in train_dataset_tail.alloc.values()])

        global_alloc_max = max([head_max_count, tail_max_count])
        assert global_alloc_max == args.negative_sample_size, logging.info(f'gloabl max: {global_alloc_max}, nb: {args.negative_sample_size}')
        global_alloc_min = min([head_min_count, tail_min_count])

        train_dataloader_tail = DataLoader(
            train_dataset_tail,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn,
            generator=g,
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('square = %f' % args.square_gamma)
    logging.info('sum_ns_loss = %d' % args.sum_ns_loss)
    logging.info('negative_sample_size = %d' % args.negative_sample_size)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('beta = %f' % args.beta)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)


    if args.do_train:

        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        best_mrr = 0.0

        args.umax = 0
        args.umin = 0
        args.lzero = 0
        best_vmrr = 0.0

        for step in range(init_step, args.max_steps):
            args.step = step
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args, global_alloc_max, global_alloc_min)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0 and step:
                logging.info('Evaluating on Valid Dataset...')

                log_cuda_memory("BEFORE EVAL")
                torch.cuda.reset_peak_memory_stats()
                v_metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args,)
                log_metrics('Valid', step, v_metrics)

                if v_metrics['MRR'] > best_vmrr:
                    best_vmrr = v_metrics['MRR']
                    logging.info('Saving Parameters...')
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(kge_model, optimizer, save_variable_list, args)
    
                logging.info('Evaluating on Test Dataset...')
                metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args,)
                log_metrics('Test', step, metrics)
                if 'tree' in args.data_path:
                    logging.info('Evaluating on Full Test Dataset...')
                    metrics = kge_model.test_step(kge_model, test_triples, all_true_triples + fn_triples, args,)
                    log_metrics('Full Test', step, metrics)

                log_cuda_memory("AFTER EVAL")
                gc.collect()
                torch.cuda.empty_cache()
                log_cuda_memory("AFTER EVAL CLEANUP")
                torch.cuda.reset_peak_memory_stats()

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args,)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args,)
        log_metrics('Test', step, metrics)
        if 'tree' in args.data_path:
            logging.info('Evaluating on Full Test Dataset...')
            metrics = kge_model.test_step(kge_model, test_triples, all_true_triples + fn_triples, args,)
            log_metrics('Full Test', step, metrics)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args,)
        log_metrics('Test', step, metrics)

def log_cuda_memory(prefix=""):
    if not torch.cuda.is_available():
        return

    gib = 1024 ** 3

    logging.info(
        "%s CUDA CURRENT ALLOCATED: %.2f GiB",
        prefix,
        torch.cuda.memory_allocated() / gib,
    )
    logging.info(
        "%s CUDA CURRENT RESERVED : %.2f GiB",
        prefix,
        torch.cuda.memory_reserved() / gib,
    )
    logging.info(
        "%s CUDA PEAK ALLOCATED   : %.2f GiB",
        prefix,
        torch.cuda.max_memory_allocated() / gib,
    )
    logging.info(
        "%s CUDA PEAK RESERVED    : %.2f GiB",
        prefix,
        torch.cuda.max_memory_reserved() / gib,
    )

if __name__ == '__main__':
    main(parse_args())
