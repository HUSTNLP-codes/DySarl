import os
import torch
import dgl
import torch.nn.functional as F
import numpy as np
import pickle
from typing import Dict, Tuple, List
LOG_DIR = '../results/'
DATA_PATH = '../data/'
MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}
operations = {
    'add': torch.add,
    'mul': lambda x, y: x * y,
    'unary': lambda x, y: x,
    'div': lambda x, y: x / y.clamp_max(-1e-15) if y < 0 else x / y.clamp_min(1e-15),
    'max': torch.maximum,
    'min': torch.minimum,
    'mean': lambda x, y: (x + y) / 2
}
activations = {
    'exp': torch.exp,
    'sig': torch.sigmoid,
    'soft': F.softplus,
    'tanh': torch.tanh,
    '': None
}


def givens_rotations(r, x):
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def givens_reflection(r, x):
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_ref = givens[:, :, 0:1] * torch.cat((x[:, :, 0:1], -x[:, :, 1:]), dim=-1) + givens[:, :, 1:] * torch.cat(
        (x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_ref.view((r.shape[0], -1))


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


def _lambda_x(x, c):
    x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
    return 2 / (1. - c * x_sqnorm).clamp_min(MIN_NORM)


def expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def logmap0(y, c):
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def expmap(u, p, c):
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = ( tanh(sqrt_c / 2 * _lambda_x(p, c) * u_norm) * u / (sqrt_c * u_norm))
    gamma_1 = mobius_add(p, second_term, c)
    return gamma_1


def logmap(p1, p2, c):
    sub = mobius_add(-p1, p2, c)
    sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    lam = _lambda_x(p1, c)
    sqrt_c = c ** 0.5
    return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm


def project(x, c):
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y, c):
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def mobius_matvec(m, x, c, b=None):
    sqrt_c = c ** 0.5
    x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    mx = x @ m.transpose(-1, -2)
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    if b is not None:
        res = project(mobius_add(res, b, c), c)
    return res


def hyp_distance_multi_c(x, v, c, eval_mode=False):
    sqrt_c = c ** 0.5
    if eval_mode:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1)
        xv = x @ v.transpose(0, 1) / vnorm
    else:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True)
        xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True)
    if vnorm.min().item() < 1e-15:
        print("error in expmap0")
    gamma = tanh(sqrt_c * vnorm) / sqrt_c
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


class Dataset(object):
    def __init__(self, data_path: str, name: str):
        self.root = os.path.join(data_path, name)

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(os.path.join(self.root, f + '.pickle'), 'rb')
            self.data[f] = pickle.load(in_file)

        print(self.data['train'].shape)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

        inp_f = open(os.path.join(self.root, 'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2
        return np.vstack((self.data['train'], copy))

    def get_valid(self):
        copy = np.copy(self.data['valid'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2
        return np.vstack((self.data['valid'], copy))

    def get_test(self):
        copy = np.copy(self.data['test'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2
        return np.vstack((self.data['test'], copy))

    def get_shape(self):
        return self.n_entities, self.n_predicates


def add_subject(e1, e2, r, d, n_relations):
    if not e2 in d:
        d[e2] = {}
    if not r+n_relations in d[e2]:
        d[e2][r+n_relations] = set()
    d[e2][r+n_relations].add(e1)


def add_object(e1, e2, r, d):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def get_all_answer(total_data, n_relations):
    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        add_subject(s, o, r, all_ans, n_relations)
        add_object(s, o, r, all_ans)
    return all_ans


def get_savedir(dataset):
    save_dir = os.path.join(LOG_DIR, dataset, 'checkpoint')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def build_sub_graph(n_entities, n_relations, triples, use_cuda, device, dropout=0.):
    def comp_deg_norm(g):
        out_deg = g.out_degrees(range(g.number_of_nodes())).float()
        out_deg[torch.nonzero(out_deg == 0).view(-1)] = 1
        norm = 1.0 / out_deg
        return norm

    if dropout > 0:
        rand_triples = triples[torch.randperm(triples.shape[0]), :]
        triples = rand_triples[:- int(rand_triples.shape[0] * dropout)]
    src, rel, dst = triples.transpose()

    g = dgl.graph((src, dst), num_nodes=n_entities, device=device)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, n_entities, dtype=torch.long, device=device).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.edata['type'] = torch.LongTensor(rel).cuda() if use_cuda else torch.LongTensor(rel)
    return g


def count_params(model):
    total = 0
    for x in model.parameters():
        if x.requires_grad:
            res = 1
            for y in x.shape:
                res *= y
            total += res
    return total


def get_ranking(test_triples, score, filtered, batch_size=512):
    num_triples = len(test_triples)
    n_batch = (num_triples + batch_size - 1) // batch_size
    rank = torch.ones(len(test_triples))
    filter_rank = torch.ones(len(test_triples))
    for idx in range(n_batch):
        batch_start = idx * batch_size
        batch_end = min(num_triples, (idx + 1) * batch_size)
        triples_batch = test_triples[batch_start:batch_end, :]
        filtered_score_batch = score[batch_start:batch_end, :]
        raw_score_batch = filtered_score_batch.clone()
        target_idxs = test_triples[batch_start:batch_end, 2].cpu().tolist()
        target = torch.stack([raw_score_batch[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
        for i, query in enumerate(triples_batch):
            filter_out = [test_triples[batch_start + i, 2].item()]
            raw_score_batch[i, torch.LongTensor(filter_out)] = -1e6
            filter_out += filtered[(query[0].item(), query[1].item())]
            filtered_score_batch[i, torch.LongTensor(filter_out)] = -1e6
        rank[batch_start:batch_end] += torch.sum((raw_score_batch >= target).float(), dim=1).cpu()
        filter_rank[batch_start:batch_end] += torch.sum((filtered_score_batch >= target).float(), dim=1).cpu()
    return rank, filter_rank


def compute_metrics(ranks):
    mean_rank = torch.mean(ranks.float()).item()
    mean_reciprocal_rank = torch.mean(1. / ranks.float()).item()
    hits_at = torch.FloatTensor((list(map(
        lambda x: torch.mean((ranks <= x).float()).item(),
        (1, 5, 10)
    ))))
    return {'MR': mean_rank, 'MRR': mean_reciprocal_rank, 'hits@[1,5,10]': hits_at}


def format_metrics(metrics, split):
    result = "\t {} MR: {:.2f} | ".format(split, metrics['MR'])
    result += "MRR: {:.4f} | ".format(metrics['MRR'])
    result += "H@1: {:.4f} | ".format(metrics['hits@[1,5,10]'][0])
    result += "H@5: {:.4f} | ".format(metrics['hits@[1,5,10]'][1])
    result += "H@10: {:.4f}".format(metrics['hits@[1,5,10]'][2])
    return result