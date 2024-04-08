from abc import ABC, abstractmethod
from layers import *
from utils import givens_rotations, givens_reflection, mobius_add, expmap0, project, hyp_distance_multi_c, logmap0, operations

class KGModel(nn.Module, ABC):
    def __init__(self, sizes, rank, dropout, gamma, bias, init_size, use_cuda=False):
        super(KGModel, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.bias = bias
        self.init_size = init_size
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.rel = nn.Embedding(sizes[1], rank)
        self.bh = nn.Embedding(sizes[0], 1)
        self.bh.weight.data = torch.zeros((sizes[0], 1))
        self.bt = nn.Embedding(sizes[0], 1)
        self.bt.weight.data = torch.zeros((sizes[0], 1))

    def forward(self, queries, ent_emb, eval_mode=False, rel_emb=None, c=None):
        lhs_e, lhs_biases = self.get_queries(queries, ent_emb)
        rhs_e, rhs_biases = self.get_rhs(queries, ent_emb, eval_mode)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)

        factors = self.get_factors(queries, ent_emb)
        return predictions, factors

    @abstractmethod
    def get_queries(self, queries, ent_emb):
        pass

    @abstractmethod
    def get_rhs(self, queries, ent_emb, eval_mode):
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        pass

    def score(self, lhs, rhs, eval_mode):
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                return lhs_biases + rhs_biases.t() + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score

    def get_factors(self, queries, ent_emb):
        head_e = ent_emb[queries[:, 0]]
        rel_e = self.rel(queries[:, 1])
        rhs_e = ent_emb[queries[:, 2]]
        return head_e, rel_e, rhs_e


class BaseH(KGModel):
    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.bias, args.init_size)

        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank))
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank)) - 1.0
        self.multi_c = args.multi_c

        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1))
        else:
            c_init = torch.ones((1, 1))
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rhs(self, queries, ent_emb=None, eval_mode=False):
        if eval_mode:
            return ent_emb, self.bt.weight
        else:
            return ent_emb[queries[:, 2]], self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2

    def get_c(self):
        if self.multi_c:
            return self.c
        else:
            return self.c.repeat(self.sizes[1], 1)


class AttH(BaseH):
    def __init__(self, args):
        super(AttH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank)) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank))
        self.act_att = nn.Softmax(dim=1)
        self.scale = nn.Parameter(torch.Tensor([1. / np.sqrt(self.rank)]), requires_grad=False)

    def get_queries(self, queries, ent_emb, gc_mode=False):
        if gc_mode:
            queries = queries.view(-1, 1)
            s = torch.zeros_like(queries)
            queries = torch.cat((s, queries), 1)
            head = ent_emb
        else:
            head = ent_emb[queries[:, 0]]
        c_p = self.get_c()
        c = F.softplus(c_p[queries[:, 1]])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)

        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act_att(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)

        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])


class DySarl(AttH):
    def __init__(self, args):
        super(DySarl, self).__init__(args)
        self.device = args.device
        self.n_layers = args.n_layers
        self.en_dropout = args.dropout
        self.de_dropout = args.de_dropout
        self.up_dropout = args.up_dropout
        self.dataset = args.dataset

        if self.dataset == "WN9IMG":
            ling_f = r"../pre_train/matrix_wn_ling.npy"
            visual_f = r"../pre_train/matrix_wn_visual.npy"
        elif self.dataset == "FBIMG":
            ling_f = r'../pre_train/matrix_fb_ling.npy'
            visual_f = r'../pre_train/matrix_fb_visual.npy'
        fb_ling, fb_visual = torch.tensor(np.load(ling_f)), torch.tensor(np.load(visual_f))
        self.img_vec = fb_visual.to(torch.float32).cuda()
        self.img_dimension = fb_visual.shape[-1]
        self.ling_vec = fb_ling.to(torch.float32).cuda()
        self.ling_dimension = fb_ling.shape[-1]
        self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, self.rank), requires_grad=True).cuda()
        nn.init.xavier_uniform(self.mats_img)
        self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, self.rank), requires_grad=True).cuda()
        nn.init.xavier_uniform(self.mats_ling)

        self.init_ent_emb = nn.Embedding(self.sizes[0], self.rank)
        self.init_ent_emb.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank))

        self.h = None
        assert args.n_layers > 0
        self.update = SymmetricAttnLayer(self.rank, args.sizes[0], args.use_time, self.init_ent_emb, args.n_head,
                                         args.up_dropout, args.layer_norm, args.double_precision, self.init_size)
        self.layers_h = nn.ModuleList()
        self.layers_e = nn.ModuleList()
        self.build_layers(args)
        self.s_hp = nn.Parameter(torch.Tensor([args.s_hp]), requires_grad=False)
        self.s_delta_ind = args.s_delta_ind
        if args.s_hp < 0:
            if args.s_delta_ind:
                self.delta_l = nn.Parameter(torch.zeros(self.sizes[0], 1), requires_grad=True)
                self.delta_r = nn.Parameter(torch.zeros(self.sizes[1], 1), requires_grad=True)
            else:
                self.delta = nn.Parameter(torch.zeros(self.sizes[0], 1), requires_grad=True)
        self.score_comb = operations[args.s_comb]
        self.score_softmax = args.s_softmax
        self.s_dropout = args.s_dropout
        self.reason_dropout = args.reason_dropout

    def build_layers(self, args):
        for i in range(self.n_layers):
            self.layers_h.append(
                HyperbolicLayer(self.rank, self.sizes[1], self.en_dropout,
                                self.get_queries, args.en_loop, self.init_size, args.en_bias))
            self.layers_e.append(
                EuclideanLayer(self.rank, self.sizes[1], self.rel, self.en_dropout,
                               args.en_loop, self.init_size, args.en_bias))

    def MMforward(self, g):
        self.h = self.init_ent_emb.weight.clone()
        img_vec = torch.mm(self.img_vec, self.mats_img)
        ling_vec = torch.mm(self.ling_vec, self.mats_ling)
        g = g[0].to(self.device)
        hidden_h = self.snap_forward_h(g, self.h)
        hidden_e = self.snap_forward_e(g, self.h)
        struc_vec = self.update([hidden_h, hidden_e])[-1]
        output = self.update([struc_vec, img_vec, ling_vec])[-1]
        return output

    def snap_forward_h(self, g, in_ent_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = in_ent_emb[node_id]
        for i, layer in enumerate(self.layers_h):
            layer(g)
        return g.ndata.pop('h')

    def snap_forward_e(self, g, in_ent_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = in_ent_emb[node_id]
        for i, layer in enumerate(self.layers_e):
            layer(g)
        return g.ndata.pop('h')

    def Fact_Gated_Unit(self, queries, old_score, new_score, act=torch.sigmoid):
        if self.score_softmax:
            old_score = torch.softmax(old_score, 1, old_score.dtype)
            new_score = torch.softmax(new_score, 1, new_score.dtype)
        if self.s_hp[0] < 0:
            if self.s_delta_ind:
                w1 = self.delta_l[queries[:, 0]]
                w2 = self.delta_r[queries[:, 1]]
            else:
                w1 = self.delta[queries[:, 0]]
                w2 = self.delta[queries[:, 1]]
            if act:
                w1 = act(w1)
                w2 = act(w2)
            w = self.score_comb(w1, w2)
            w = F.dropout(w, self.up_dropout, training=self.training)
        else:
            w = self.s_hp.repeat(queries.shape[0], 1)
        score = w * new_score + (1 - w) * old_score
        return score

    def reason(self, queries, output, eval_mode=False, epoch=1000, rel_emb=None, c=None):
        new_factors, old_factors = None, None
        if self.s_hp != 0:
            new_ent_emb = F.dropout(output, self.reason_dropout, training=self.training)
            init_ent_emb = self.init_ent_emb.weight
            new_score, new_factors = self.forward(queries, new_ent_emb, eval_mode=eval_mode)
            old_score, old_factors = self.forward(queries, init_ent_emb, eval_mode=eval_mode)
            score = self.Fact_Gated_Unit(queries, old_score, new_score)
        else:
            score, factor = self.forward(queries, output, eval_mode=eval_mode)
        return score, (old_factors, new_factors)