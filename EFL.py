''' Entanglement Feature Learning '''
# global import: numpy, tensorflow
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
import tensorflow as tf
# define the op logdet and its gradient
# from https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
# from tensorflow/python/ops/linalg_grad.py
# Gradient for logdet
def logdet_grad(op, grad):
    a = op.inputs[0]
    a_adj_inv = tf.check_numerics(
                    tf.matrix_inverse(a, adjoint=True), 
                    'zero determinant')
    out_shape = tf.concat([tf.shape(a)[:-2], [1, 1]], axis=0)
    return tf.reshape(grad, out_shape) * a_adj_inv
# define logdet by calling numpy.linalg.slogdet
def logdet(a, name = None):
    with tf.name_scope(name, 'LogDet', [a]) as name:
        res = tf.check_numerics(
                py_func(lambda x: np.linalg.slogdet(x)[1], 
                      [a], 
                      tf.float64, 
                      name=name, 
                      grad=logdet_grad), # set the gradient
                'zero determinant')
        return res
''' Entanglement Region Server '''
# entanglement region object
class Region(object):
# input:
#     blocks : a list/set of block indices
#     partitions :: int : number of blocks in total
# support method: complement, get equivalent class, convert configuration
    def __init__(self, blocks, partitions):
        self.blocks = frozenset(blocks)
        self.partitions = partitions
        self._config = None
    def __len__(self):
        return len(self.blocks)
    def __iter__(self):
        return iter(self.blocks)
    def __repr__(self):
        return repr(set(self.blocks))
    def __hash__(self):
        return hash(self.blocks)
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.blocks == other.blocks
    # complement region
    def complement(self):
        blocks = frozenset(range(self.partitions)) - self.blocks
        return Region(blocks, self.partitions)
    # equivalent regions generator
    def equivalences(self):
        known = set()
        for shift in range(self.partitions): # translation
            for dir in (-1,1): # reflection
                # construct equivalent region
                blocks = frozenset((dir * x + shift)%self.partitions for x in self.blocks)
                eqreg = Region(blocks, self.partitions)
                if eqreg not in known:
                    known.add(eqreg)
                    yield eqreg
                # if half size, also consider complement
                if len(self) == self.partitions//2:
                    eqreg = eqreg.complement()
                    if eqreg not in known:
                        known.add(eqreg)
                        yield eqreg
    # map to Ising configuration (as float)
    def config(self):
        if self._config is None: # if config not cached
            # generate an array of ones
            c = np.ones(self.partitions)
            c[list(self.blocks)] = -1. # set blocks to -1
            self._config = c # keep the config
        else: # if config cached
            c = self._config # retrived the cache
        return c
# entanglement region server
from itertools import combinations
from scipy.special import binom
class RegionServer(object):
    def __init__(self, partitions):
        self.partitions = partitions
        self.default_batch = 7
    # set filler by method
    def fill(self, method):
        if isinstance(method, str):
            self.filler = getattr(self, method)
            self.fargs = tuple()
        elif isinstance(method, tuple):
            self.filler = getattr(self, method[0])
            self.fargs = method[1:]
        self.pool = self.filler(*self.fargs) # fill the pool
    # fetch regions (with auto-refill)
    def fetch(self, batch = None):
        if batch is None: # get all remaining regions in the source
            for region in self.pool:
                yield region
            self.pool = self.filler(*self.fargs) # refill the pool
        else: # a batch is specified
            for i in range(batch):
                region = next(self.pool, None) # get a region from the pool
                if region is None: # if got None, pool empty
                    self.pool = self.filler(*self.fargs) # refill the pool
                    region = next(self.pool) # get again
                yield region
    # representative subregions
    def representative(self):
        assert self.partitions < 10, 'To many partitions, can not get all representatives, Use random method instead.'
        # check for cache
        if hasattr(self, 'representative_cache'): # if chache exist
            # use chache
            for region in self.representative_cache:
                yield region
        else: # if chache not established, built it
            self.representative_cache = []
            # size of subregion from 1 to half 
            for rank in range(1, self.partitions//2 + 1):
                known = set() # hold the known equivalent subregions
                for blocks in combinations(range(self.partitions), rank):
                    region = Region(blocks, self.partitions) # convert to region
                    if region not in known: # if region unknown
                        # add equivalent regions to known
                        known.update(region.equivalences())
                        self.representative_cache.append(region)
                        yield region # yield the representative region
    # consecutive subregions
    def consecutive(self):
        for x in range(1, self.partitions//2+1):
            yield Region(range(0, x), self.partitions)
    # random subregions
    def random(self):
        for k in range(self.default_batch):
            conf = np.random.randint(2, size=self.partitions-1)
            blocks = {0} ^ {i + 1 for i, x in enumerate(conf) if x == 1}
            yield Region(blocks, self.partitions)
    # multiple subregions
    def multiple(self, *args):
        if len(args) == 0: # by default
            ns = np.full(self.default_batch, 1) # single region
        elif len(args) == 1: # one argument
            # specified number of regions
            ns = np.full(self.default_batch, max(min(args[0],self.partitions-args[0]),1))
        else: # specified a range of region numbers
            low = max(args[0], 1) # lower bonded by 1
            high = min(args[1], self.partitions//2) # uppder bounded by half
            ns = np.random.choice(range(low, high+1), self.default_batch)
        for k in range(self.default_batch):
            bdys = np.random.choice(self.partitions, size=2*ns[k], replace=False)
            bdyiter = iter(sorted(bdys))
            blocks = set() # prepare to hold the blocks
            for a, b in zip(bdyiter, bdyiter):
                blocks ^= set(range(a,b)) # add region (a,b)
            yield Region(blocks, self.partitions)
    # weighted subregions:
    def weighted(self, n0 = 1.):
        beta = np.abs(np.log((self.partitions//2)/n0-1.))
        nlst = np.array(range(1, self.partitions//2+1))
        prob = binom(self.partitions//2, nlst)*np.exp(-beta*nlst)
        prob = prob/np.sum(prob)
        ns = np.random.choice(nlst, size=self.default_batch, p=prob)
        for k in range(self.default_batch):
            bdys = np.random.choice(self.partitions, size=2*ns[k], replace=False)
            bdyiter = iter(sorted(bdys))
            blocks = set() # prepare to hold the blocks
            for a, b in zip(bdyiter, bdyiter):
                blocks ^= set(range(a,b)) # add region (a,b)
            yield Region(blocks, self.partitions)
''' Physical System '''
class FreeFermion(object):
# input:
#     mass :: float : mass of the fermions, in [-1.,1.]
#     size :: int : length of the 1D many-body state
    def __init__(self, size, mass, c = 1.):
        assert -1. <= mass <= 1., 'mass must be in the range of [-1.,1.].'
        assert size%2 == 0, 'size must be even.'
        self.size = size
        self.mass = mass # fermion mass
        self.c = c # central charge
        self.info = 'FF({0},{1:.2f},{2:.1f})'.format(self.size, self.mass, self.c)
        self._built = False
    # build system
    def build(self):
        if not self._built:
            self._built = True # change status
            # construct single-particle density matrix
            u = np.tile([1.+self.mass, 1.-self.mass], self.size//2) # hopping
            A = np.roll(np.diag(u), 1, axis=1) # periodic rolling
            A[-1,0] = -A[-1,0] # antiperiodic boundary condition, to avoid zero mode
            A = 1j * (A - np.transpose(A)) # Hamiltonian
            (_, U) = np.linalg.eigh(A) # digaonalization
            V = U[:,:self.size//2] # take lower half levels
            self.G = np.dot(V, V.conj().T) # construct density matrix
    # calculate entanglement entropy given sites
    def S(self, sites):
        if len(sites) == 0: return 0.
        # diagonalize reduced density matrix
        p = np.linalg.eigvalsh(self.G[sites,:][:,sites])
        # return 2nd Renyi entropy
        S = -self.c * np.asscalar(np.sum(np.log(np.abs(p**2 + (1.- p)**2))))
        return S
''' Lattice System '''
# Node object
class Node(object):
    def __init__(self, cell, index):
        self.cell = cell # father cell
        self.index = index
    def __getattr__(self, attr):
        if attr == 'coordinate':
            return self.cell.coordinate + [self.index]
        else:
            raise AttributeError("%s object has no attribute named %r" %
                         (self.__class__.__name__, attr))
    def __repr__(self):
        return 'Node{0}'.format(self.coordinate)
# Cell object
class Cell(object):
    def __init__(self, chain, index, size = 3):
        self.chain = chain # father chain
        self.index = index
        # set up a list of nodes
        self.nodes = [Node(self, i) for i in range(size)]
    def __getattr__(self, attr):
        if attr == 'coordinate':
            return self.chain.coordinate + [self.index]
        elif attr == 'size':
            return len(self)
        else:
            raise AttributeError("%s object has no attribute named %r" %
                         (self.__class__.__name__, attr))
    def __repr__(self):
        return 'Cell{0}'.format(self.coordinate)
    # getitem method returns the node
    def __getitem__(self, key):
        return self.nodes[key]
    def __iter__(self):
        return iter(self.nodes)
    def __len__(self):
        return len(self.nodes)
# Chain object
class Chain(object):
    def __init__(self, lattice, index, size = 0):
        self.lattice = lattice # father lattice
        self.index = index
        # set up a list of cells
        self.cells = [Cell(self, i) for i in range(size)]
        self.UVcells = []
        self.IRcells = []
    def __getattr__(self, attr):
        if attr == 'coordinate':
            return [self.index]
        elif attr == 'size':
            return len(self)
        elif attr == 'IR':
            return self.lattice[self.index + 1]
        elif attr == 'UV':
            return self.lattice[self.index - 1]
        else:
            raise AttributeError("%s object has no attribute named %r" %
                         (self.__class__.__name__, attr))
    def __repr__(self):
        return 'Chain{0}'.format(self.coordinate)
    # getitem method returns the cell
    def __getitem__(self, key):
        return self.cells[key]
    def __iter__(self):
        return iter(self.cells)
    def __len__(self):
        return len(self.cells)
    # remove cells
    def remove(self, inds):
        # del cell from collection
        for i in sorted(inds, reverse = True):
            del self.cells[i]
        # reindexing
        for (i, cell) in enumerate(self.cells):
            cell.index = i
    def removeUV(self):
        self.remove([cell.index for cell in self.UVcells])
        self.UVcells = []
    def removeIR(self):
        self.remove([cell.index for cell in self.IRcells])
        self.IRcells = []
# Lattice (base class)
class Lattice(object):
    def __init__(self):
        self.width = 0
        self.depth = 0
        self.chains = []
        self.node_dict = {}
        self.slot_dict = {}
        self.adj_dict = {}
        self.size ={}
        self._built = False
    # getitem method returns the chain
    def __getitem__(self, key):
        return self.chains[key]
    def __iter__(self):
        return iter(self.chains)
    # build lattice
    def build(self):
        if not self._built:
            self._built = True # change status
            self.construct() # construct lattice structure
            self.collect() # collect nodes and links
    # construct lattice structure
    def construct(self):
        # set up chains here
        pass
    # collect nodes and links
    def collect(self):
        # set up dictionaries here
        pass
    # make link
    def mk_link(self, node0, node1, info):
        # get node indices
        (ind0, ind1) = (self.get_node_index(node) for node in (node0, node1))
        # switch by link type
        typ = info[0]
        if typ == '1':
            self.add_to_adj(typ, [(ind0, ind1), +1.])
            self.add_to_adj(typ, [(ind1, ind0), -1.])
        elif typ.startswith('w'): # wh or wJ
            indv = self.get_slot_index(info)
            self.add_to_adj(typ, [(indv, ind0, ind1), +1.])
            self.add_to_adj(typ, [(indv, ind1, ind0), -1.])
    # get node index
    def get_node_index(self, node):
        # check if node is known
        if node in self.node_dict: # if node is known
            return self.node_dict[node] # return its index
        else: # if node is unknown
            ind = len(self.node_dict) # asign a index to the last by len
            self.node_dict[node] = ind # register the node
            return ind # return its index
    # get slot index
    def get_slot_index(self, info):
        (typ, label) = info # get type
        if typ in self.slot_dict: # if type is known
            label_dict = self.slot_dict[typ] # get the label dict of that type
            if label in label_dict: # if label is known
                return label_dict[label] # return slot index
            else: # if label is unknown
                ind = len(label_dict) # assign a index
                label_dict[label] = ind # register the label
                return ind # return its index
        else: # if type is unknown
            self.slot_dict[typ] = {label: 0} # create a label dict of the type
            return 0 # return the index
    # add to adjacency tensor instruction
    def add_to_adj(self, typ, pair):
        # if the type has been registered
        if typ in self.adj_dict: # type is there
            self.adj_dict[typ].append(pair) # add the pair
        else: # the type has not been estabilished
            self.adj_dict[typ] = [pair] # estabilish the type with the pair
    # return adjacency tensor
    def adjten(self, typ):
        # create empty tensors, shape assigned according to type
        if typ == '1': # constant type -> matrix
            A = np.zeros(shape = [self.size['node'], self.size['node']])
        elif typ.startswith('w'): # other types -> tensor
            A = np.zeros(shape = [self.size[typ], self.size['node'], self.size['node']])
        else:
            raise ValueError('Lattice.adjten receives unknown type "{0}".'.format(typ))
        # filling in the tensor according to adj_dict
        for (p, v) in self.adj_dict[typ]:
            A[p] = v
        return A
    # return link count vector
    def lcvect(self, typ):
        if typ.startswith('w'):
            lc = np.zeros(shape = [self.size[typ]]) # prepare to count links
            for [(a, _, _), _] in self.adj_dict[typ]:
                lc[a] += 0.5 # to mod out double counting
        else:
            raise ValueError('Lattice.lcvect receives unknown type "{0}".'.format(typ))
        return lc
    # print lattice structure
    def print_structure(self, upto = 'node'):
        print('depth = ',self.depth)
        if not upto == 'none':
            for chain in self:
                print(chain)
                if not upto == 'chain':
                    for cell in chain:
                        if cell in chain.UVcells:
                            print('├', cell, 'UV')
                        elif cell in chain.IRcells:
                            print('├', cell, 'IR')
                        else:
                            print('├', cell)
                        if not upto == 'cell': 
                            for node in cell:
                                print('│ ├', node)
# Single Sheeted Lattice (subclass of Lattice) 
class SSLattice(Lattice):
    def __init__(self, width, depth, pattern = [-1,1]):
        super().__init__()
        self.width = width
        self.depth = depth
        self.pattern = pattern
        self.info = 'SSLatt({0},{1},'.format(width,depth)+''.join({-1:'U',1:'I'}[p] for p in pattern)+')'
    # construct lattice according to the pattern
    # pattern = (p_1,p_2,...,p_n) with p_i = +1 for IR, -1 for UV
    def construct(self):
        pattern_len = len(self.pattern)
        # from UV to IR
        chain_size = self.width # intial chain size
        for z in range(self.depth):
            # create a chain of chain_size
            chain = Chain(self, z, size = chain_size)
            # specify UV/IR cells
            if z == 0: # for the input layer all cells are IR
                chain.IRcells = chain.cells
            else:
                for u in range(0, chain_size, pattern_len):
                    for (i, p) in enumerate(self.pattern):
                        j = u + i
                        if j < chain_size:
                            if p == +1: # IR, up wards
                                chain.IRcells.append(chain[j])
                            elif p == -1: # UV, down wards
                                chain.UVcells.append(chain[j])
            # register chain to collections
            self.chains.append(chain)
            # calculate chain size of the next layer (IR)
            chain_size = pattern_len * (len(chain.IRcells) // self.pattern.count(-1))
            # chain size vanish, stop going deeper
            if chain_size == 0:
                self.depth = len(self.chains) # overwrite depth to the current depth
                break # break z loop
        # final touch: remove the IR cells from the top chain
        self[-1].removeIR()
    # collect nodes and links (as directed graph) upto specified depth
    def collect(self):
        # four dictionaries will be generated
        # node_dict = {node: ind, ...} maps Node object to its global index
        #                              in the lattice graph
        # slot_dict = {'wh': {lbl: ind, ...}, 
        #              'wJ': {lbl: ind, ...}}
        #       collects and classifies slots into wh and wJ weights
        #       keeps the slot label to global index mapping
        #       the slot index -> the slice index in the adjacency tensor   
        # adj_dict = {'1': [[(i,j), ±1.] ...],
        #             'wh': [[(s,i,j), ±1.] ...],
        #             'wJ': [[(s,i,j), ±1.] ...]}
        #       stores the instructions to make the adjacency tensor
        # size = {'node': *, 'wh': *, 'wJ': *}
        #       keeps the size information of nodes, wh and wJ
        for chain in self[:self.depth]: # from UV to IR
            z = chain.index # keep chain index in z
            # intra-cell links
            lkinfo = ('1',) # constant 1
            for cell in chain:
                self.mk_link(cell[0], cell[1], info = lkinfo)
            for cell in chain.UVcells:
                self.mk_link(cell[0], cell[2], info = lkinfo)
                self.mk_link(cell[1], cell[2], info = lkinfo)
            for cell in chain.IRcells:
                self.mk_link(cell[0], cell[2], info = lkinfo)
                self.mk_link(cell[2], cell[1], info = lkinfo)
            # inter-cell links
            # horizontal (dual) links
            for cell in chain:
                x = cell.index # keep cell index in x
                # determine the link type
                if z == 0: # for input layer
                    lkinfo = ('wh', x) # weight of h
                elif z == self.depth - 1: # for top layer
                    lkinfo = ('1',) # constant 1
                else: # for the rest of the bulk layers
                    lkinfo = ('wJ', z) # weight of J
                if x == 0: # for the boundary link: reverse direction
                    self.mk_link(chain[0][0], chain[-1][1], info = lkinfo)
                else: # for the remaining links: normal direction
                    self.mk_link(chain[x-1][1], chain[x][0], info = lkinfo)
            # vertical (dual) links
            if z < self.depth - 1: # if not the top layer
                chainIR = chain.IR # get the IR chain
                for (cell0, cell1) in zip(chain.IRcells, chainIR.UVcells):
                    lkinfo = ('wJ', z + 1) # weight of J
                    self.mk_link(cell0[2], cell1[2], info = lkinfo)
        # set sizes
        self.size = {typ: len(lbls) for (typ, lbls) in self.slot_dict.items()}
        self.size['node'] = len(self.node_dict)
''' Ising Model '''
class IsingModel(object):
    def __init__(self, lattice):
        self.lattice = lattice
        self.info = 'Ising({0})'.format(lattice.info)
    # build model (given input log bound dimension lnD)
    def build(self, lnD):
        self.lattice.build() # first build lattice
        # setup adjacency tensors as TF constants
        A_bg = tf.constant(self.lattice.adjten('1'),dtype = tf.float64, name = 'A_bg')
        As_h = tf.constant(self.lattice.adjten('wh'),dtype = tf.float64, name = 'As_h')
        As_J = tf.constant(self.lattice.adjten('wJ'),dtype = tf.float64, name = 'As_J')
        # boundary Ising configurations
        conf0 = tf.ones([self.lattice.size['wh']], dtype = tf.float64, name = 'conf0')
        self.confs = tf.placeholder(dtype = tf.float64, 
                shape = [None, self.lattice.size['wh']], name = 'confs')
        # external field configurations
        with tf.name_scope('h'):
            self.h = lnD/2. # external field strength
            hs = self.h * self.confs
            h0 = self.h * conf0
        # coupling strength (trainable variable)
        self.J = tf.Variable(0.27 * np.ones(self.lattice.size['wJ']), 
                dtype = tf.float64, name = 'J')
        # generate weighted adjacency matrices
        with tf.name_scope('Ising_net'):
            A_J = self.wdotA(self.J, As_J, name = 'A_J')
            A_hs = self.wdotA(hs, As_h, name = 'A_hs')
            A_h0 = self.wdotA(h0, As_h, name = 'A_h0')
            # construct full adjacency matrix
            As = A_hs + A_bg + A_J
            A0 = A_h0 + A_bg + A_J
        # calcualte free energy and model entropy
        with tf.name_scope('free_energy'):
            self.Fs = self.free_energy(As, self.J, hs, name = 'Fs')
            self.F0 = self.free_energy(A0, self.J, h0, name = 'F0')
        self.Smdl = tf.subtract(self.Fs, self.F0, name = 'S_mdl')
        # calculate cost function
        self.Ssys = tf.placeholder(dtype = tf.float64, shape = [None], name = 'S_sys')
        with tf.name_scope('cost'):
            self.MSE = tf.reduce_mean(tf.square(self.Smdl/self.Ssys - 1.))
            self.wall = tf.reduce_sum(tf.nn.relu(self.J[1:]-self.J[:-1]))
            self.cost = self.MSE
            # record cost function
            tf.summary.scalar('logMSE', tf.log(self.MSE))
            tf.summary.scalar('logwall', tf.log(self.wall + 1.e-10))
        # coupling regularizer
        with tf.name_scope('regularizer'):
            Jrelu = tf.nn.relu(self.J) # first remove negative couplings
            # construct the upper bond
            Jmax = tf.concat([tf.reshape(self.h,[1]),Jrelu[:-1]],axis=0)
            # clip by the upper bond and assign to J
            self.regularizer = self.J.assign(tf.minimum(Jrelu, Jmax))
    # convert coupling to weight, and contract with adjacency matrix
    def wdotA(self, f, A, name = 'wdotA'):
        with tf.name_scope(name):
            return tf.tensordot(tf.exp(2. * f), A, axes = 1)
    # free energy (use logdet)
    def free_energy(self, A, J, h, name = 'F'):
        with tf.name_scope(name):
            with tf.name_scope('Jh'):
                Js = J * tf.constant(self.lattice.lcvect('wJ'), name = 'J_count')
                hs = h * tf.constant(self.lattice.lcvect('wh'), name = 'h_count')
                F0 = tf.reduce_sum(Js) + tf.reduce_sum(hs, axis=-1)
            logdetA = logdet(A)
            F = -0.5*logdetA + F0
        return F                
''' Entanglement Feature Learning
input:
    system : object that has a method S to return the entropy
    model : a model that takes takes  
'''
# entanglement feature data server
class DataServer(object):
    def __init__(self, model, system):
        self.model = model
        self.system = system
        self.size = system.size
        self.partitions = model.lattice.width
        assert self.size%self.partitions == 0, 'Size not divisible by partitions.'
        self.blocksize = self.size // self.partitions
        self.lnD = self.blocksize * self.system.c * np.log(2.)
        # set up a entanglement region server
        self.region_server = RegionServer(self.partitions)
        self.method = None
        self.blockmap = None
    # get sites given a region
    def sites(self, region):
        # if the region is over half of the partitions
        if len(region) > self.partitions//2:
            # take the complement region instead
            return self.sites(region.complement())
        else: # map block indices to site indices
            if self.blockmap is None:
                self.blockmap = np.arange(self.size).reshape(
                            [self.partitions, self.blocksize])
            return self.blockmap[list(region)].flatten()
    # calculate entanglement feature and package data
    def pack(self, regions):
        # prepare empty lists for configs and sysS
        confs = []
        Ssys = []
        # go through all regions in the batch
        for region in regions:
            # configuration of Ising boundary
            confs.append(region.config())
            # entanglement entropy from system
            Ssys.append(self.system.S(self.sites(region)))
        # return data as a dict
        return {self.model.confs: np.array(confs), self.model.Ssys: np.array(Ssys)}
    # fetch data
    def fetch(self, method, batch = None):
        if method != self.method: # if method changed
            self.method = method # update method state
            self.region_server.fill(method) # fill the server by new method
        return self.pack(self.region_server.fetch(batch))
# EFL machine
from datetime import datetime
class Machine(object):
    def __init__(self, model, system, method='weighted'):
        self.model = model
        self.system = system
        self.method = method
        self.data_server = DataServer(model, system)
        self.graph = tf.Graph() # TF graph
        self.session = tf.Session(graph = self.graph) # TF session
        self.para = None # parameter dict
        # status flags
        self._built = False
        self._initialized = False
    def __getattr__(self, attr):
        if attr == 'info':
            return self.model.info+self.system.info+''.join(str(x) for x in self.method)
        else:
            raise AttributeError("%s object has no attribute named %r" %
                         (self.__class__.__name__, attr))
    # build machine
    def build(self):
        if not self._built: # if not built
            self._built = True # change status
            # build machine
            self.system.build() # build system
            # add nodes to the TF graph
            with self.graph.as_default():
                self.model.build(self.data_server.lnD) # build model
                self.step = tf.Variable(0,name='step',trainable=False)
                self.learning_rate = tf.placeholder(tf.float32,shape=[],name='lr')
                self.beta1 = tf.placeholder(tf.float32,shape=[],name='beta1')
                self.beta2 = tf.placeholder(tf.float32,shape=[],name='beta2')
                self.epsilon = tf.placeholder(tf.float32,shape=[],name='epsilon')
                self.optimizer = tf.train.AdamOptimizer(
                                learning_rate=self.learning_rate, 
                                beta1=self.beta1, 
                                beta2=self.beta2, 
                                epsilon=self.epsilon)
                self.trainer = self.optimizer.minimize(self.model.cost, 
                                global_step = self.step)
                self.initializer = tf.global_variables_initializer()
                self.regularizer = self.model.regularizer
                self.writer = self.pipe() # set up data pipeline
                self.saver = tf.train.Saver() # add saver
    # pipe data (by summary)
    def pipe(self):
        # get variable names
        var_names = {i:name for name, i in self.model.lattice.slot_dict['wJ'].items()}
        # go through each component of J
        for i in range(self.model.lattice.size['wJ']):
            tf.summary.scalar('J/{0}'.format(var_names[i]), self.model.J[i])
        # optimizer slots
        slot_names = self.optimizer.get_slot_names()
        for slot_name in slot_names:
            slot = self.optimizer.get_slot(self.model.J, slot_name)
            for i in range(self.model.lattice.size['wJ']):
                tf.summary.scalar('{0}/{1}'.format(slot_name, var_names[i]), slot[i])
        self.summary = [tf.summary.merge_all(), self.step]
        timestamp = datetime.now().strftime('%d%H%M%S')
        return tf.summary.FileWriter('./log/' + timestamp)
    # initialize machine
    def initialize(self):
        if not self._initialized: # if not initialized
            self._initialized = True # change status
            # initialize machine
            assert not self.para is None, 'Machine.para was not set yet.'
            self.session.run(self.initializer, self.para) # initialize graph
    # train machine
    def train(self, steps=1, check=20, method=None, batch=None, 
            learning_rate=0.005, beta1=0.9, beta2=0.9, epsilon=1e-8):
        self.build() # if not built, build it
        if method is None:
            method = self.method # by default, use global method
        else:
            self.method = method # otherwise method updated
        # setup parameter feed dict
        self.para = {self.learning_rate:learning_rate, 
                    self.beta1:beta1, 
                    self.beta2:beta2, 
                    self.epsilon:epsilon}
        self.initialize() # if not initialized, initialize it
        # start training loop
        for i in range(steps):
            # construct the feed dict, attach data to para
            self.feed = {**self.para, **self.data_server.fetch(method, batch)}
            try: # zero determinant may cause a problem, try it
                self.session.run(self.trainer, self.feed) # train one step
            except tf.errors.InvalidArgumentError: # when things go wrong
                continue # skip the rest, go the the next batch of data
            self.session.run(self.regularizer) # run regularization
            if self.session.run(self.step)%check == 0: # summarize
                self.writer.add_summary(*self.session.run(self.summary, self.feed))
    # graph export for visualization in TensorBoard 
    def add_graph(self):
        self.writer.add_graph(self.graph) # writter add graph
    # save session
    def save(self):
        # save model, without saving the graph
        path = self.saver.save(self.session, './machine/'+self.info, 
                                write_meta_graph=False)
        print('INFO:tensorflow:Saving parameters to %s'%path)
    # load session
    def load(self):
        self.build() # if not built, build it
        # restore model
        self.saver.restore(self.session, './machine/'+self.info)
        # session is initialized after loading 
        self._initialized = True
# Toolbox 
# I/O 
# JSON pickle: export to communicate with Mathematica 
import jsonpickle
def export(filename, obj):
    with open('./data/' + filename + '.json', 'w') as outfile:
        outfile.write(jsonpickle.encode(obj))
import pickle
# pickle: binary save and load for python.
def save(filename, obj):
    with open('./data/' + filename + '.dat', 'bw') as outfile:
        pickle.dump(obj, outfile)
def load(filename):
    with open('./data/' + filename + '.dat', 'br') as infile:
        return pickle.load(infile)












