import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from utils import build_sub_graph
from utils import compute_metrics, get_ranking
from torch import nn
import os
from typing import Dict, Tuple, List
import pickle


class KGOptimizer(object):
    def __init__(self, model, optimizer, dataset_name, valid_freq, multi_step, topk, batch_size, neg_sample_size,
                 double_neg=False, use_cuda=False, dropout=0., verbose=True, grad_norm=1.0):
        self.model = model
        self.dataset_name = dataset_name
        self.optimizer = optimizer
        self.grad_norm = grad_norm
        self.batch_size = batch_size
        self.verbose = verbose
        self.double_neg = double_neg
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.neg_sample_size = neg_sample_size
        self.n_entities = model.sizes[0]
        self.n_relations = model.sizes[1]
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.valid_freq = valid_freq
        self.multi_step = multi_step
        self.topk = topk
        self.dropout = dropout
        inp_f = open(os.path.join('../data/'+self.dataset_name, 'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def calculate_loss(self, out_g, output_emb, epoch=-1):
        loss = torch.zeros(1).cuda().to(self.device) if self.use_cuda else torch.zeros(1)
        mean_score = None
        scores, factors = self.model.reason(out_g, output_emb, eval_mode=True, epoch=epoch)
        truth = out_g[:, 2]
        loss += self.loss_fn(scores, truth)
        return loss, mean_score

    def epoch(self, train_data, epoch=-1):
        train_data = train_data.astype('int64')
        losses = []
        actual_examples = train_data[torch.randperm(train_data.shape[0]), :]
        with tqdm(total=train_data.shape[0], unit='ex', disable=not self.verbose) as bar:
            b_begin = 0
            while b_begin < train_data.shape[0]:
                input_batch = actual_examples[b_begin:b_begin + self.batch_size]
                g_list = [build_sub_graph(self.n_entities, self.n_relations, input_batch, self.use_cuda, self.device, self.dropout)]
                output = torch.from_numpy(input_batch).long().cuda().to(self.device) if self.use_cuda else torch.from_numpy(input_batch).long()
                output_emb = self.model.MMforward(g_list)
                loss, mean_score = self.calculate_loss(output, output_emb, epoch=epoch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss.item())
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
        return np.mean(losses)

    def evaluate(self, test_data, valid_mode=False, epoch=-1):
        test_data = test_data.astype('int64')
        valid_losses = []
        valid_loss = None
        ranks = []
        filter_ranks = []
        with torch.no_grad():
            with tqdm(total=test_data.shape[0], unit='ex', disable=not self.verbose) as bar:
                b_begin = 0
                while b_begin < test_data.shape[0]:
                    input_batch = test_data[b_begin:b_begin + self.batch_size]
                    g_list = [build_sub_graph(self.n_entities, self.n_relations, input_batch, self.use_cuda, self.device, self.dropout)]
                    test_triples = torch.LongTensor(input_batch).cuda().to(self.device) if self.use_cuda else torch.LongTensor(input_batch)
                    output_emb = self.model.MMforward(g_list)
                    if valid_mode:
                        loss, mean_score = self.calculate_loss(test_triples, output_emb, epoch=epoch)
                        valid_losses.append(loss.item())
                    if (epoch + 1) % self.valid_freq == 0 or not valid_mode:
                        scores, _ = self.model.reason(test_triples, output_emb, eval_mode=True, epoch=epoch)
                        rank, filter_rank = get_ranking(test_triples, scores, self.to_skip['filtered'])
                        ranks.append(rank)
                        filter_ranks.append(filter_rank)
                    b_begin += self.batch_size
                    bar.update(input_batch.shape[0])
                if valid_losses:
                    valid_loss = np.mean(valid_losses)
                if ranks:
                    ranks = torch.cat(ranks)
                if filter_ranks:
                    filter_ranks = torch.cat(filter_ranks)
                return valid_loss, ranks, filter_ranks