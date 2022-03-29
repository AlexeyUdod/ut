import torch as tr
import itertools
from tqdm.auto import tqdm
# from torch_geometric.data import Data as trg_data
# from torch_sparse import spspmm, spmm
# import torch_sparse
from importlib import reload
from torch.jit import script
import time
import torch
import copy


t = tr.tensor
coo = tr.sparse_coo_tensor


def one2inf(x:'torch.tensor'
            ) -> 'torch.tensor':
    """Transform tensor from max_1 to max_inf"""

    res = x / (1 - tr.abs(x))
    res[res.isnan()] = 0
    return res


def inf2one(x:'torch.tensor'
            ) -> 'torch.tensor':
    """Transform tensor from max_inf to max_1"""

    res = tr.sign(x) * (1 - (1 / (1 + tr.abs(x))))
    res[res.isnan()] = 0
    return res


def i_sum(x1, x2): return one2inf(inf2one(x1) + inf2one(x2))
def i_mul(x1, x2): return one2inf(inf2one(x1) * inf2one(x2))
def o_sum(x1, x2): return inf2one(one2inf(x1) + one2inf(x2))
def o_mul(x1, x2): return inf2one(one2inf(x1) * one2inf(x2))


def comp_A(dic1, n_preds):
    print('Computing A')
    size = (n_preds, n_preds)
    A2 = tr.empty(size)

    for s in tqdm(dic1.keys()):
        ks = set(dic1[s].keys())
        set1 = set(itertools.permutations(ks, 2)) | {(x, x) for x in ks}
        if len(set1) > 0:
            inds = t(list(set1)).T
            A2 += coo(inds, tr.ones(inds.shape[1]), size=size).to_dense()
    return A2, dic1


def A_cond(A, pers, n):
    A_ = A - (tr.eye(A.shape[0]) * A.diag())
    A2 = A / A.diag().unsqueeze(1)
    A2[A2.isnan()] = 0
    A2_ = A2 - (tr.eye(A2.shape[0]) * A2.diag())
    A2_sparse = (A2_).to_sparse()
    A3 = coo(A2_sparse._indices(), A2_sparse._values() * 2 - 1, size=A_.shape)
    A3_dense = A3.to_dense()
    A5 = (A_ * (A3_dense > pers) * (A_ > n)).to_sparse()
    inds = A5._indices().T
    vals = A5._values().unsqueeze(1)
    res = tr.cat([inds.clone(), vals.clone()], dim=1).int()
    return res[res[:, 2].sort(0, True).indices], A2_


# @tr.jit.script
def sum_sp(m1:  torch.Tensor,
           m2:  torch.Tensor
           ) -> torch.Tensor:
    """Add data to sparse matrix"""

    if not m1.is_sparse:
        # m1 = m1.to_sparse().to(m1.device)
        m1_indices = m1
        m1_values = tr.ones(m1.shape[1]).to(m1.device)
    else:
        m1_indices = m1._indices()
        m1_values = m1._values()

    if not m2.is_sparse:
        # m1 = m2.to_sparse().to(m2.device)
        m2_indices = m2
        m2_values = tr.ones(m2.shape[1]).to(m2.device)
    else:
        m2_indices = m2._indices()
        m2_values = m2._values()

    size = tr.cat([t(m1.size()).unsqueeze(0),
                   t(m2.size()).unsqueeze(0)]
                  ).max(dim=0).values.tolist()
#     print(size)
    res_inds = tr.cat([m1_indices, m2_indices], dim=1).long()
    res_vals = tr.cat([m1_values, m2_values], dim=0)
    res = coo(res_inds, res_vals, size=tuple(size)).coalesce().to(m1.device) #  size = size

    return res


def ti(label, s):
    print(f'{label} {time.time() - s}')
    s = time.time()
    return s


# @tr.jit.script
def add2A(A:            torch.Tensor,
          data:         torch.Tensor,
          ohe:          'bool' = False,
          symetric:     'bool' = True,
          ) -> torch.Tensor:
    """Add data to adjacency matrix"""

    dev = A.device
    # A = A.clone().to(dev)
    size = A.shape
    # if not tr.is_tensor(window):
    #     window = t(window)


    # TODO delete morph to_dense and realize full sparse support
    # if data.is_sparse:
    #     data = data.to_dense()

    # for window in data:
    cond = data.layout is not torch.strided
#     print(cond)
    if cond:
        data = data._indices()
    for i in tr.arange(data[0].max().item()): # .unique()
        # s = time.time()

        # if ohe:
        #     window = t(window).nonzero(as_tuple=True)[0]

        # if symetric:
        #     inds = tr.cartesian_prod(t(window), t(window)).T.to(dev)
        # else:
        #     inds = tr.cartesian_prod(t(window[0]), t(window[1])).T.to(dev)

        window = data[1, data[0] == i]
        # if symetric:
        # window = window.indices().squeeze()
        inds = tr.cartesian_prod(window, window).T.to(dev)
        # else:
        #     inds = tr.cartesian_prod(t(window[0]), t(window[1])).T.to(dev)

        # s = ti('carts prod', s)
        # print(inds)
        if inds.shape[1] > 0:
            # r = coo(inds, tr.ones(inds.shape[1]).to(dev),
            #         size=size).bool().int() # .coalesce()
            r = inds
            r = coo(inds, tr.ones(inds.shape[1]), size = A.size())
            # s = ti('make coo r', s)
            if A.is_sparse:
#                 A = sum_sp(A, r)
#                 A = tr.add(A, r) 

                A += r
            else:
                A += r.to_dense()
            # s = ti('add r to A', s)
        # print(f'iter in {time.time() - s}')
#         if cond:
#             return A.to_sparse()
    return A


def norm_lim(x, lim):
    return lim + (1 - lim) * x


def norm_conf(x, conf=(0.9, 1)):

    cond1 = x > conf[0]
    cond2 = x < (1 - conf[1])
    mask = tr.logical_or(cond1, cond2) * 1
    res = x * mask
    return res


def XCA_add_entity(X, C, A, mem_size=2):
    """Add space for new entity in X, C and A matrixes

    Args:
    X (tensor): [num_entities, 1] entities current status.
    C (tensor): [num_entities, num_entities] entities Confusion matrix.
    A (tensor): [num_entities, num_entities] entities Adjacency matrix.
    mem_size (int): default = 2 size of memory (or window) to remember.

    Results:
    X (tensor): [num_entities + mem_size, 1]
    entities current status.
    C (tensor): [num_entities + mem_size, num_entities + mem_size]
    entities Confusion matrix.
    A (tensor): [num_entities + mem_size, num_entities + mem_size]
    entities Adjacency matrix.

    """
    X_ = tr.zeros(X.shape[0] + mem_size, 1)
    X_[:-mem_size] = X
    X = X_

    C_ = tr.zeros(C.shape[0] + mem_size, C.shape[0] + mem_size)
    C_[:-mem_size, :-mem_size] = C
    C = C_

    ar = tr.arange(A.shape[0], A.shape[0] + mem_size)
    inds = (ar[:-1], ar[1:])
    A_ = tr.zeros(tuple(t(A.shape) + mem_size)).index_put(inds, t(1.))
    A_[:-mem_size, :-mem_size] = A
    A = A_
    return X, C, A


def add_nodes(graph, x_new):
    if graph.x is None:
        return trg_data(x_new, graph.edge_index)
    return trg_data(tr.cat([graph.x, x_new]), graph.edge_index)


def add_edges(graph, e_new):
    if graph.edge_index is None:
        return trg_data(graph.x, e_new)
    return trg_data(graph.x, tr.cat([graph.edge_index, e_new], dim=1))


def add_features(graph, f_new):
    if graph.x is None:
        return trg_data(f_new, graph.edge_index)
    res = trg_data(tr.cat([graph.x, f_new], dim=1), graph.edge_index)
    return res


def mmm(sp1: 'torch.tensor',
        sp2: 'torch.tensor'
        ) -> 'torch.tensor':
    """Multi Matrix Multiplication.\n
    Allow to multiplicate sparse and dense types of matrixes"""

    # if sp1.device != sp2.device:
        # TODO  to device

    def inds_and_size(x):
        x_size = x.size()
        x_inds = x._indices()
        if len(x_size) == 1:
            x_inds = tr.cat([tr.zeros_like(x_inds), x_inds])
            x_size = (1, x_size[0])
        return x_inds, x_size

    trans = False
    if not sp1.is_sparse and not sp2.is_sparse:
        return sp1 @ sp2

    if sp2.is_sparse and not sp1.is_sparse:
        sp1, sp2 = sp2.transpose(0, 1), sp1
        trans = True

    sp1_inds, sp1_size = inds_and_size(sp1)

    if sp2.is_sparse:
        sp2_inds, sp2_size = inds_and_size(sp2)
        res = spspmm(sp1_inds, sp1._values(),
                     sp2_inds, sp2._values(),
                     sp1_size[0], max([sp1_size[1], sp2_size[0]]), sp2_size[1],
                     True)
        return coo(res[0], res[1], size=(sp1_size[0], sp2_size[1]))

    else:
        if len(sp2.size()) == 1:
            sp2 = sp2.unsqueeze(1)
        res = spmm(sp1_inds, sp1._values(),
                   sp1_size[0], sp1_size[1],
                   sp2)
        if trans: res = res.T
        return res.to_sparse()


def norm(m: 'tensor',
         k: 'threshold'=0.95):
    """Normalize nodes state after forward passing"""

    if m.is_sparse:
        vals = m._values()
        vals[vals >= k] = 1
        vals[vals < k] = 0
        m = coo(m._indices(), vals, size=m.size())
    else:
        m[m >= k] = 1
        m[m < k] = 0

    return m


def T_sp(m: 'sparse tensor',
         dims: 'list of dims to transpose' = [1, 0]
         ) -> 'transposed sparse tensor':
    """Transpose sparse tensor (or permute dims)"""

    if not m.is_sparse:
        return m.T
    return coo(m._indices()[dims], m._values(), size=m.size()).to(m.device)


def node_degree(A: 'tensor: graph adj matrix',
                nodes : 'tensor of nodes',
                k: 'threshold' = 0.95
                ) -> 'tensor: with node in degre and node out degre':
    """Get node degrees"""

    r = []
    for node in nodes:

        if A.is_sparse:
            inds = A._indices().to(A.device)
            inds_in = inds[1]
            node_in_degree = inds_in[inds_in == node].shape[0]
            inds_out = inds[0]
            node_out_degree = inds_out[inds_out == node].shape[0]
        else:
            node_in_degree = len((A)[:, node].nonzero(as_tuple=True)[0])
            node_out_degree = len((A)[node].nonzero(as_tuple=True)[0])
        r.append([node_in_degree, node_out_degree])

    return t(r).to(A.device)


def A_split(A):
    """Split matrix to positive and negative matrixes"""

    vals = A._values()
    cond = vals > 0
    inds_pos = cond.nonzero(as_tuple=True)[0]
    inds_neg = (1 - cond).nonzero(as_tuple=True)[0]
    A = coo(A._indices()[:, inds_pos], vals[inds_pos], size=A.size())
    A_neg = coo(A._indices()[:, inds_neg], -(vals[inds_neg]), size=A.size())
    return A, A_neg


def n_forward(A: 'tensor: graph adj matrix',
              inp: 'tensor[A.shape[0]]: nodes start state'
              ) -> 'tensor: graph adj matrix':
    """NNodes activation passing"""

    def _n_forward(A, inp):
        # and
        ands = mmm(inp, A)
        ands = norm(ands)

        # or
        A_t = T_sp(A)
        ors = mmm(inp, A_t).coalesce()
        ors_inds = ors.indices().to(A.device)
        degr = (node_degree(A, ors_inds[1])[:,1]).to(A.device)
        ors_vals = (ors.values() / degr).to(A.device)
        ors_vals[ors_vals.isnan()] = 0
        ors = coo(ors_inds, ors_vals, size=ors.size()).to(A.device)
        ors = norm(ors)

        # join and+or

        # res_inds = tr.cat([ors._indices(), ands._indices()], dim=1)
        # res_vals = tr.cat([ors._values(), ands._values()], dim=0)
        # res = coo(res_inds, res_vals, size=(1, A.shape[1])).coalesce()
        res = sum_sp(ors, ands)
        # print(res)
        res = norm(res)
        # print(res)
        return res

    if not A.is_sparse:
        A = A.to_sparse().to(A.device)

    if A._values().min() < 0:
        A_pos, A_neg = A_split(A)
        res_pos = _n_forward(A_pos, inp)
        res_neg = _n_forward(A_neg, inp)
        res = res_pos * res_neg
    else:
        res = _n_forward(A, inp)

    return res


def n_forward_old(A:    'tensor: graph adj matrix',
                  inp:  'tensor[A.shape[0]]: nodes start state'
                  ) ->  'tensor: graph adj matrix':
    """NNodes activation passing"""

    ands = mmm(inp, A)
    A_t = T_sp(A)
    A_sum = A_t.to_dense().sum(dim=0)
    ors = mmm(inp, A_t).to_dense() / A_sum
    ors[ors.isnan()] = 0
    res = norm(ors + ands.to_dense())

    return res


def diff(x:'torch.tensor',
         dim: int=1
         ) -> 'torch.tensor':
    """Return difference between tensor values in selected dimention"""

    if dim == 0:
        return x - tr.nn.functional.pad(x.T, (1, 0))[:,:-1].T
    return x - tr.nn.functional.pad(x, (1, 0))[:,:-1]


def X2sp(X):

    X[X >= 0.5] = 1
    X[X <  0.5] = 0
    X_s = X.to_sparse(size=X.size)
    return X_s



# @tr.jit.script
class UTensorSparse:


    def __init__(self,
                 inds:  tr.Tensor = None,
                 vals:  tr.Tensor = None,
                 size:  tr.tensor = None):

        # super().__init__()
        self.inds = inds
        self.vals = vals
        self.siz = size

    def indices(self):
        return self.inds

    def values(self):
        if self.inds is not None:
            if self.vals is not None:
                _vals = self.vals
            else:
                _vals = tr.ones(self.inds.shape[1])
            return _vals
        # return

    def size(self):
        if self.siz is None:
            self.siz = self.indices().max(dim=1).values + 1
        return self.siz

    def __repr__(self):
        cls = self.__class__.__name__
        print(f"""({cls}\nIndices =\n{self.inds}\nValues =\n{self.vals})""")
        return ''

    def clone(self):
        self2 = copy.deepcopy(self)
        return self2

    def coalesce(self):
        self2 = self.clone()
        # print(self.values().shape, self.indices().shape)
        self2.inds, self2.vals = torch_sparse.coalesce(self2.indices(), self2.values(), *tuple(self2.size()))
        return self2

    def to_sparse(self):
        return tr.sparse_coo_tensor(self.indices(), self.values(), tuple(self.size().int()))

    def to_dense(self):
        return self.to_sparse().to_dense()


    def add(self, data, coalesce=True):
        """Add data to self from torch.Tensor or ut.UTensorSparse"""

        self2 = self.clone()
        # print(self.vals, self.values())
        if data.__class__.__name__ == 'UTensorSparse':
            self2.inds = tr.cat([self.indices(), data.indices()], dim=1)
            if coalesce or self.vals is not None or data.vals is not None:
                self2.vals = tr.cat([self.values(), data.values()])
            # print(self2.vals)
        else:
            if not data.is_sparse:
                data = data.to_sparse()
            self2.inds = tr.cat([self.indices(), data._indices()], dim=1)
            self2.vals = tr.cat([self.values(), data._values()])

        if coalesce:
            self2 = self2.coalesce()
            # self.inds, self.vals = self2.inds, self2.vals

        return self2


    # @tr.jit.script
class UTSparse:

    def __init__(self,
                 inds:  tr.Tensor = None,
                 vals:  tr.Tensor = None,
                 size:  tr.tensor = None):

        # super().__init__()
        self.inds = inds
        self.vals = vals
        self.siz = size


    def indices(self):
        return self.inds


    def values(self):
        if self.inds is not None:
            if self.vals is not None:
                _vals = self.vals
            else:
                _vals = tr.ones(self.inds.shape[-1])
            return _vals
        # return


    def size(self):
        if self.siz is None:
            size = self.indices().max(dim=1).values + 1
        else:
            size = self.siz
        return size


    def __repr__(self):
        cls = self.__class__.__name__
        print(f"""({cls}:\nIndices =\n{self.inds},\nValues =\n{self.vals},\nsize = {self.siz})""")
        return ''


    def clone(self):
        self2 = copy.deepcopy(self)
        return self2


    def coalesce(self):
        self2 = self.clone()
        self2.inds, self2.vals = torch_sparse.coalesce(self2.indices(),
                                                       self2.values(),
                                                       *tuple(self2.size()))
        return self2


    def to_sparse(self):
        return tr.sparse_coo_tensor(self.indices(), self.values(), tuple(self.size().int()))


    def to_dense(self):
        return self.to_sparse().to_dense()


    def add(self, data, coalesce=True):
        """Add data to self from torch.Tensor or ut.UTSparse"""

        self2 = self.clone()

        print(self2.indices().shape, self2.values().shape, data.indices().shape, data.values().shape)
        if data.__class__.__name__ == 'UTSparse':

            if data.inds is None:
                pass
            elif self2.inds is None:
                self2.inds = data.indices()
            else:
                self2.inds = tr.cat([self2.indices(), data.indices()], dim=1)

            # if coalesce or self.vals is not None or data.vals is not None:
            if not coalesce:
                if data.vals is None:
                    pass
                elif self2.vals is None:
                    self2.vals = data.vals
                else:
                    print(self2.inds.shape, self2.vals.shape)
                    self2.vals = tr.cat([self2.vals, data.vals])
                    print(self2.inds.shape, self2.vals.shape)
            else:
                print(self2.indices().shape, self2.values().shape, data.indices().shape, data.values().shape)
                self2.vals = tr.cat([self2.values(), data.values()])
                print(self2.inds.shape, self2.vals.shape)

            print(self2.inds.shape, self2.vals.shape)

        else:
            if not data.is_sparse:
                data = data.to_sparse()
            self2.inds = tr.cat([self2.indices(), data._indices()], dim=1)
            self2.vals = tr.cat([self2.values(), data._values()])

        if coalesce:
            self2 = self2.coalesce()

        return self2


    def add_imps(self, data, coalesce=False):
        self2 = self.clone()

        for win in data:
            win = win.unique()
            prod = UTSparse(inds = tr.cartesian_prod(win, win).T)
            # print(prod)
            self2 = self2.add(prod, coalesce=coalesce)

        return self2

    # def __index__(self):
    #     return

    def __getitem__(self, i):

        inds = self.inds
        inds = inds[1:, inds[0] == i]
        nnz = inds.nonzero(as_tuple=True)[-1]
        vals = self.vals[nnz]
        return UTSparse(inds = inds, vals = vals)


    # def eye(self):
    #     # TODO multi dimensions eye
    #     if len(A.size()) == 2:
    #         eye = tr.zeros(A.size().max())
    #         inds = A.inds
    #         i = (inds[0] == inds[1]).nonzero(as_tuple=True)[0]
    #         eye[inds[0, i]] = A.values()[i]
    #         return eye


# A = UTSparse(inds, vals)
# A = UTSparse()
# # A3 = A.add(A3, coalesce=False).coalesce().to_dense()
# data = tr.randint(0, 9, (50, 10))
# A = A.add_imps(data, coalesce=True)
# A, A.eye()

# @tr.jit.script
class UTSparse:

    def __init__(self,
                 inds:  tr.Tensor = None,
                 vals:  tr.Tensor = None,
                 size:  tr.tensor = None):

        # super().__init__()
        self.inds = inds
        self.vals = vals
        self.siz = size


    def indices(self):
        return self.inds


    def values(self):
        if self.inds is not None:
            if self.vals is not None:
                _vals = self.vals
            else:
                _vals = tr.ones(self.inds.shape[-1])
            return _vals
        return


    def size(self):
        if self.siz is None:
            size = self.indices().max(dim=1).values + 1
        else:
            size = self.siz
        return size


    def __repr__(self):
        cls = self.__class__.__name__
        print(f"""({cls}:\nIndices =\n{self.inds},\nValues =\n{self.vals},\nsize = {self.siz})""")
        return ''


    def clone(self):
        self2 = copy.deepcopy(self)
        return self2


    def coalesce(self):
        self2 = self.clone()
        self2.inds, self2.vals = torch_sparse.coalesce(self2.indices(),
                                                       self2.values(),
                                                       *tuple(self2.size()))
        return self2


    def to_sparse(self):
        return tr.sparse_coo_tensor(self.indices(), self.values(), tuple(self.size().int()))


    def to_dense(self):
        return self.to_sparse().to_dense()


    def add(self, data, coalesce=True):
        """Add data to self from torch.Tensor or ut.UTSparse"""

        self2 = self.clone()

        p = lambda label: print(label, [x.shape if x is not None else x for x in [self2.inds, self2.vals, data.inds, data.vals] ])
        if data.__class__.__name__ == 'UTSparse':

            vars = [self2.inds, self2.vals, data.inds, data.vals]
            conds = tr.tensor([x is not None for x in vars]).int().tolist()
            code = str(''.join([str(x) for x in conds]))

            if code in ['0000', '1000', '1100']:
                pass
            elif code in ['1110', '1111', '1011']:
                self2.vals = tr.cat([self2.values(), data.values()])
                self2.inds = tr.cat([self2.indices(), data.indices()], dim=1)
            elif code in ['1010',]:
                self2.inds = tr.cat([self2.inds, data.inds], dim=1)
            # elif code in ['0010',]:
            #     self2.inds = data.inds
            elif code in ['0011', '0010',]:
                self2.inds = data.inds
                self2.vals = data.vals
            elif code in ['0100', '0101', '0110', '0111', '0001', '0101', '1001', '1101']:
                raise ValueError('Values have no Indices!')

        else:
            if not data.is_sparse:
                data = data.to_sparse()
            self2.inds = tr.cat([self2.indices(), data._indices()], dim=1)
            self2.vals = tr.cat([self2.values(), data._values()])

        if coalesce:
            self2 = self2.coalesce()

        return self2


    def add_imps(self, data, coalesce=False):
        self2 = self.clone()

        for win in data:
            win = win.unique()
            prod = UTSparse(inds = tr.cartesian_prod(win, win).T)
            self2 = self2.add(prod, coalesce=coalesce)

        return self2

    def __getitem__(self, i):

        inds = self.inds
        inds = inds[1:, inds[0] == i]
        nnz = inds.nonzero(as_tuple=True)[-1]
        vals = self.vals[nnz]
        return UTSparse(inds = inds, vals = vals)


    def eye(self):
        # TODO multi dimensions eye
        if len(self.size()) == 2:
            eye = tr.zeros(self.size().max())
            inds = self.inds
            i = (inds[0] == inds[1]).nonzero(as_tuple=True)[0]
            eye[inds[0, i]] = self.values()[i]
            return eye

    def eye2(self):
        # TODO multi dimensions eye
        if len(self.size()) == 2:
            # eye_vals = tr.zeros(A.size().max())
            inds = self.indices()
            # i = (inds[0] == inds[1]).nonzero(as_tuple=True)[0]
            # eye_vals[inds[0, i]] = A.values()[i]

            cond = inds[0] == inds[1]
            eye_inds = inds[cond]
            eye_vals = self.values()[cond]

            return UTSparse(eye_inds, eye_vals, size = self.size())


    def get_imps(self):
        self2 = self.clone()
        eye = self2.eye()

        return self2


    
def norm01(x):
    """Normalize input to [0 +1]"""
    
    x = x.clone()
    if x.abs().sum() == 0:
        return x
    x -= x.min()
    x = (x / x.max())
    return x


def norm11(x):
    """Normalize input to [-1 +1]"""
    
    x = x.clone()
    if x.abs().sum() == 0:
        return x
    x -= x.min()
    x = (x / x.max()) * 2 - 1
    return x


def norm101(x):
    return (norm11(x.clone()) * 1.999999).int().float()
  
    
def dd(x:tr.Tensor):
    """Divide input tensor to it`s Diagonal"""
    
    x = x.clone()
    x /= x.diag()
    x[x.isnan()] = 0
    return x


def ds(x:tr.Tensor):
    """Divide input tensor to it`s columns Sum"""
    
    x = x.clone()
    x /= x.abs().sum(dim=-1)
    x[x.isnan()] = 0
    return x


def means(l1:list) -> tr.Tensor:
    """Return mean between tensors in input list"""
    
    res = tr.cat([x.unsqueeze(0) for x in l1]).mean(dim=0)
    return res


def atgo(x):  
    """ATanG One. 
    Return atan norm for input <= One"""
    
    return tr.atan(x) / 3.1415926 * 4


def atgi(x):  
    """ATanG Inf
    Return atan norm for unlim input"""
    
    return tr.atan(x) / 3.1415926 * 2


def odif(x1, x2):
    """Ones(max) phase DIFference"""
    
    return atgo(x1) + atgo(x2) - 1


def idif(x1, x2):
    """Inf(max) phase DIFference"""
    
    return atgi(x1) + atgi(x2) - 1


def mdif(x1, x2 = None):
    """Matrix DIFference
    x: [1, n]"""
    
    x1e = x1.float().exp()
    
    if x2 is None:
        x2e = x1e
    else:
        x2e = x2.float().exp()
        
    return (x1e.T**-1 @ x2e).log()


def mc(x1, x2 = None):
    """Matrix Combination"""
    
    if x2 is None:
        x2 = x1
        
    return x1.T @ x2


def fc(x, n = 1):
    """Return n up level fractal combination of input matrix"""
    
    for _ in range(n):
        x = x.flatten().unsqueeze(0)
        x = mc(x)
    return x

def fdif(x):
    """Return  Fractal DIFference of input matrix"""
    
    x = x.flatten().unsqueeze(0)
    x = mdif(x)
    return x


def msym(x):
    """Check Matrix diag SYMmetry"""
    
    return (x + x.T).sum() == 0


def mdft(n, m=None):
    """Return Matrix for DFT"""
    
    ns = tr.arange(n).unsqueeze(0)
    if m is None:
        ms = ns
    else:
        ms = tr.arange(m).unsqueeze(0)
        
    return tr.exp(tr.tensor(pi * 2j / n)) ** mc(ns, ms)


def dft(x, m=None):
    """Return DFT[1,m] of input[1,n]"""
    
    n = x.shape[-1]
    if m is None:
        m = n
        
    return x @ mdft(n, m)


def addn(a, s = 0):
    """ADD Neuron to neuron matrix"""
    
    a = tr.cat([a, a[None, 0]*0 + s])
    a = tr.cat([a, a[:, 0, None]*0 + s], dim=1)
    return a


def E(sig):
    """Energy of wave"""
    
    return sig.abs().sum()


def wave_sum(sig1, sig2):
    """Sum 2 waves with energy save"""
    
    e_sum = E(sig1) + E(sig2)
    r = sig1 + sig2
    res = r / E(r) * e_sum
    return res
