''' Entanglement Feature Learning '''
# package required: numpy, tensorflow
import tensorflow as tf
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
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
        # generate an array of ones
        c = np.ones(self.partitions)
        c[list(self.blocks)] = -1. # set blocks to -1
        return c
# entanglement region server
from itertools import combinations
class RegionServer(object):
    def __init__(self, partitions):
        self.partitions = partitions
    # set region source
    def fill(self, method):
        if isinstance(method, str):
            self.filler = getattr(self, method)
            self.fargs = tuple()
        elif isinstance(method, tuple):
            self.filler = getattr(self, method[0])
            self.fargs = method[1:]
        self.pool = self.filler(*self.fargs)
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
    def representatives(self):
        # size of subregion from 1 to half 
        for rank in range(1, self.partitions//2 + 1):
            known = set() # hold the known equivalent subregions
            for blocks in combinations(range(self.partitions), rank):
                region = Region(blocks, self.partitions) # convert to region
                if region not in known: # if region unknown
                    # add equivalent regions to known
                    known.update(region.equivalences())
                    yield region # yield the representative region
    # consecutive subregions
    def consecutives(self):
        for x in range(1, self.partitions):
            yield Region(range(0, x), self.partitions)
''' Physical System '''
class FreeFermion(object):
# input:
#     mass :: float : mass of the fermions, in [-1.,1.]
#     size :: int : length of the 1D many-body state
    def __init__(self, mass, size):
        assert -1. <= mass <= 1., 'mass must be in the range of [-1.,1.].'
        assert size%2 == 0, 'size must be even.'
        self.mass = mass
        self.size = size
        self.info = 'FF[m{0:.2f},{1}]'.format(mass, size)
        self.G = self.mkG() # store single-particle density matrix
    # construct single-particle density matrix
    def mkG(self):
        u = np.tile([1.+self.mass, 1.-self.mass], self.size//2) # hopping
        A = np.roll(np.diag(u), 1, axis=1) # periodic rolling
        A = 1.j * (A - np.transpose(A)) # Hamiltonian
        (_, U) = np.linalg.eigh(A) # digaonalization
        V = U[:,:self.size//2] # take lower half levels
        return np.dot(V, V.conj().T) # construct density matrix
    # calculate entanglement entropy given sites
    def S(self, sites):
        if len(sites) == 0: return 0.
        # diagonalize reduced density matrix
        p = np.linalg.eigvalsh(self.G[sites,:][:,sites])
        # return 2nd Renyi entropy
        return -np.sum(np.log(abs(p**2 + (1.- p)**2)))
''' Lattice System '''
# Node object
class Node(object):
    def __init__(self, cell, index):
        self.cell = cell # father cell
        self.index = index
    def __getattr__(self, attr):
        if attr is 'coordinate':
            return self.cell.coordinate + [self.index]
        else:
            raise AttributeError
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
        if attr is 'coordinate':
            return self.chain.coordinate + [self.index]
        elif attr is 'size':
            return len(self)
        else:
            raise AttributeError
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
        if attr is 'coordinate':
            return [self.index]
        elif attr is 'size':
            return len(self)
        elif attr is 'IR':
            return self.lattice[self.index + 1]
        elif attr is 'UV':
            return self.lattice[self.index - 1]
        else:
            raise AttributeError
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
        self.depth = 0
        self.chains = []
    # getitem method returns the chain
    def __getitem__(self, key):
        return self.chains[key]
    def __iter__(self):
        return iter(self.chains)
    # print lattice structure
    def print_structure(self, upto = 'node'):
        print('depth = ',self.depth)
        if not upto is 'none':
            for chain in self:
                print(chain)
                if not upto is 'chain':
                    for cell in chain:
                        if cell in chain.UVcells:
                            print('├', cell, 'UV')
                        elif cell in chain.IRcells:
                            print('├', cell, 'IR')
                        else:
                            print('├', cell)
                        if not upto is 'cell': 
                            for node in cell:
                                print('│ ├', node)
# Single Sheeted Lattice (subclass of Lattice) 
class SSLattice(Lattice):
    def __init__(self, width, depth, pattern = [-1,1]):
        self.width = width
        self.depth = depth
        self.info = 'SSLatt[{0},{1},{2}]'.format(width,depth,pattern)
        self.chains = []
        self.build(pattern)
    # build lattice according to the pattern
    # pattern = (p_1,p_2,...,p_n) with p_i = +1 for IR, -1 for UV
    def build(self, pattern):
        pattern_len = len(pattern)
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
                    for (i, p) in enumerate(pattern):
                        j = u + i
                        if j < chain_size:
                            if p == +1: # IR, up wards
                                chain.IRcells.append(chain[j])
                            elif p == -1: # UV, down wards
                                chain.UVcells.append(chain[j])
            # register chain to collections
            self.chains.append(chain)
            # calculate chain size of the next layer (IR)
            chain_size = pattern_len * (len(chain.IRcells) // pattern.count(-1))
            # chain size vanish, stop going deeper
            if chain_size == 0:
                self.depth = len(self.chains) # overwrite depth to the current depth
                break # break z loop
        # final touch: remove the IR cells from the top chain
        self[-1].removeIR()
        # collect nodes and links
        self.collect()
    # collect nodes and links (as directed graph) upto specified depth
    def collect(self, depth = None):
        # four dictionaries will be generated
        # node_dict = {node: ind, ...} maps Node object to its global index
        #                              in the lattice graph
        # slot_dict = {'input': {lbl: ind, ...}, 
        #              'parameter': {lbl: ind, ...}}
        #       collects and classifies slots into inputs and parameters
        #       keeps the slot label to global index mapping
        #       the slot index -> the slice index in the adjacency tensor   
        # adj_dict = {'constant': [[(i,j), ±1.] ...],
        #             'input': [[(s,i,j), ±1.] ...],
        #             'parameter': [[(s,i,j), ±1.] ...]}
        #       stores the instructions to make the adjacency tensor
        # size = {'node': *, 'input': *, 'parameter': *, 'variable': *}
        #       keeps the size information of nodes, inputs and parameters
        if depth is None: # the depth of the lattice to be collected
            depth = self.depth
        else: # prevent depth to go beyond bound
            depth = min(depth, self.depth)
        # prepare dicts
        self.node_dict = {}
        self.slot_dict = {}
        self.adj_dict = {}
        # from UV to IR
        for chain in self[:depth]:
            z = chain.index # keep chain index in z
            # intra-cell links
            lkinfo = ('constant',) # constant type 
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
                    lkinfo = ('input', '{0}'.format(x)) # input type
                elif z == depth - 1: # for top layer
                    lkinfo = ('constant',) # constant type
                else: # for the rest of the bulk layers
                    lkinfo = ('parameter', '{0}M'.format(z)) # parameter type
                if x == 0: # for the boundary link: reverse direction
                    self.mk_link(chain[0][0], chain[-1][1], info = lkinfo)
                else: # for the remaining links: normal direction
                    self.mk_link(chain[x-1][1], chain[x][0], info = lkinfo)
            # vertical (dual) links
            if z < depth - 1: # if not the top layer
                chainIR = chain.IR # get the IR chain
                for (cell0, cell1) in zip(chain.IRcells, chainIR.UVcells):
                    lkinfo = ('parameter', '{0}H'.format(z))
                    self.mk_link(cell0[2], cell1[2], info = lkinfo)
        # set sizes
        self.size = {typ: len(lbls) for (typ, lbls) in self.slot_dict.items()}
        self.size['node'] = len(self.node_dict)
        self.size['variable'] = self.size['parameter'] + 1
    # make link
    def mk_link(self, node0, node1, info):
        # get node indices
        (ind0, ind1) = (self.get_node_index(node) for node in (node0, node1))
        # switch by link type
        typ = info[0]
        if typ == 'constant':
            self.add_to_adj(typ, [(ind0, ind1), +1.])
            self.add_to_adj(typ, [(ind1, ind0), -1.])
        elif typ in {'input', 'parameter'}:
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
        if typ == 'constant': # constant type -> matrix
            A = np.zeros(shape = [self.size['node'], self.size['node']])
        elif typ in {'input', 'parameter'}: # other types -> tensor
            A = np.zeros(shape = [self.size[typ], self.size['node'], self.size['node']])
        # filling in the tensor according to adj_dict
        for (p, v) in self.adj_dict[typ]:
            A[p] = v
        return A
    # return link count vector
    def lcvect(self, typ):
        if typ in {'input', 'parameter'}:
            lc = np.zeros(shape = [self.size[typ]]) # prepare to count links
            for [(a, _, _), _] in self.adj_dict[typ]:
                lc[a] += 0.5 # to mod out double counting
        return lc
''' Ising Model '''
class IsingModel(object):
    def __init__(self, lattice):
        self.lattice = lattice
        self.info = 'Ising[{0}]'.format(lattice.info)
        self.partitions = lattice.size['input']
    # build model
    def build(self):
        # setup adjacency tensors as TF constants
        A_bg = tf.constant(self.lattice.adjten('constant'),dtype = tf.float64)
        As_in = tf.constant(self.lattice.adjten('input'),dtype = tf.float64)
        As_pa = tf.constant(self.lattice.adjten('parameter'),dtype = tf.float64)
        # input and default (boundary Ising configurations)
        default = tf.ones([self.partitions], dtype = tf.float64, name = 'default')
        self.input = tf.placeholder(dtype = tf.float64, 
                shape = [None, self.partitions], name = 'input')
        # variable (external field and coupling strengths)
        self.variable = tf.Variable(np.random.rand(self.lattice.size['variable']), 
                dtype = tf.float64, name = 'variable')
        with tf.name_scope('gain'):
            self.gain = self.variable[0] # external field strength
        with tf.name_scope('parameter'):
            self.parameter = self.variable[1:] # coupling strengths
        # input preprocessing
        with tf.name_scope('level_ctrl'):
            hs = self.gain * self.input
            h0 = self.gain * default
        # generate weighted adjacency matrices
        with tf.name_scope('Ising_net'):
            A_pa = self.dotA(self.parameter, As_pa, name = 'A_pa')
            A_hs = self.dotA(hs, As_in, name = 'A_hs')
            A_h0 = self.dotA(h0, As_in, name = 'A_h0')
            # construct full adjacency matrix
            A_com = A_pa + A_bg
            As = A_com + A_hs
            A0 = A_com + A_h0
        # calcualte free energy and model entropy
        with tf.name_scope('free_energy'):
            self.Fs = self.free_energy(As, self.parameter, hs, name = 'Fs')
            self.F0 = self.free_energy(A0, self.parameter, h0, name = 'F0')
        self.prediction = tf.subtract(self.Fs, self.F0, name = 'prediction')
        # calculate cost function
        self.label = tf.placeholder(dtype = tf.float64, shape = [None], name = 'label')
        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(tf.square(self.prediction/self.label - 1.))
    # convert coupling to weight, and contract with adjacency matrix
    def dotA(self, coupling, A, name = 'dotA'):
        with tf.name_scope(name):
            return tf.tensordot(tf.exp(2.*coupling), A, axes = 1)
    # free energy
    def free_energy(self, A, J, h, name = 'F'):
        with tf.name_scope(name):
            with tf.name_scope('Jh'):
                Js = J * tf.constant(self.lattice.lcvect('parameter'), name = 'J_count')
                hs = h * tf.constant(self.lattice.lcvect('input'), name = 'h_count')
                with tf.name_scope('ramp'):
                    rmp = tf.reduce_sum(tf.nn.relu(Js)) + tf.reduce_sum(tf.nn.relu(hs), -1)
                with tf.name_scope('abs'):
                    abs = tf.reduce_sum(tf.abs(Js)) + tf.reduce_sum(tf.abs(hs), -1)
            with tf.name_scope('regularize'):
                with tf.name_scope('factor'):
                    factor = tf.exp(-4./self.lattice.size['node'] * rmp)
                    factor_ten = tf.reshape(factor, tf.concat([tf.shape(factor),[1,1]],0))
                Areg = factor_ten * A
            with tf.name_scope('logdet'):
                logdet = tf.log(tf.matrix_determinant(Areg))
            with tf.name_scope('collect'):
                F = -0.5*logdet - abs
            return F
''' Entanglement Feature Learning
input:
    system : object that has a method S to return the entropy
    model : a model that takes takes  
'''
# entanglement feature data server
class EFData(object):
    def __init__(self, system, model):
        self.system = system
        self.model = model
        self.size = system.size
        self.partitions = model.partitions
        assert self.size%self.partitions == 0, 'size {0} is not divisible by partitions {1}.'.format(self.size, self.partitions)
        # set up a entanglement region server
        self.region_server = RegionServer(self.partitions)
        # set up a block to site mapping
        self.blockmap = np.arange(self.size).reshape([self.partitions, self.size//self.partitions])
    # get sites given a region
    def sites(self, region):
        # if the region is over half of the partitions
        if len(region) > self.partitions//2:
            # take the complement region instead
            return self.sites(region.complement())
        else: # map block indices to site indices
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
        return {self.model.input: np.array(confs), self.model.label: np.array(Ssys)}
    # iter approach
    # set up data source
    def fill(self, method):
        self.region_server.fill(method)
    # fetch data
    def fetch(self, batch = None):
        return self.pack(self.region_server.fetch(batch))
    # non iter approach
    def initialize(self, method):
        self.region_server.fill(method)
        self.source = self.pack(self.region_server.fetch())
# EFL machine
from datetime import datetime
class EFL(object):
    def __init__(self, system, model):
        self.system = system
        self.model = model
        self.info = ''
        self.data = EFData(system, model)
        self.graph = tf.Graph()
        self.session = tf.Session(graph = self.graph)
    # build machine
    def build(self):
        with self.graph.as_default():
            self.model.build()
            # providing training handles
            self.step = tf.Variable(0, name='step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=0.02, beta1=0.9, beta2=0.9999, epsilon=1e-8)
            self.grads_vars = self.optimizer.compute_gradients(self.model.cost, [self.model.variable])
            self.trainer = self.optimizer.apply_gradients(self.grads_vars, global_step = self.step)
#             self.trainer = self.optimizer.minimize(self.model.cost, global_step = self.step)
            self.initializer = tf.global_variables_initializer()
            self.pipe() # set up data pipeline
    # initialize machine
    def initialize(self, data_method = 'representatives'):
        self.build() # build machine
        self.session.run(self.initializer) # initialize graph
        self.data.initialize(data_method) # initialize data server
    # train machine
    def train(self, steps, check = 10):
        for i in range(steps):
            self.session.run(self.trainer, self.data.source)
            if i%check == 0:
                self.writer.add_summary(*self.session.run([self.summary, self.step], self.data.source))
        return self.session.run((self.model.cost, self.model.variable), self.data.source)
    # pipe data (by summary)
    def pipe(self):
        # record cost function
        tf.summary.scalar('cost/cost', self.model.cost)
        tf.summary.scalar('cost/logcost', tf.log(self.model.cost))
        tf.summary.histogram('prediction', self.model.prediction)
        # get optimizer slots
        slot_names = self.optimizer.get_slot_names()
        var_names = {i+1:name for (name, i) in self.model.lattice.slot_dict['parameter'].items()}
        var_names[0] = '0I'
        # go through each component of J
        for i in range(self.model.lattice.size['parameter']):
            tf.summary.scalar('var/{0}/c'.format(var_names[i]), self.model.variable[i])
            tf.summary.scalar('var/{0}/g'.format(var_names[i]), self.grads_vars[0][0][i])
            for slot_name in slot_names:
                tf.summary.scalar('var/{0}/{1}'.format(var_names[i], slot_name), self.optimizer.get_slot(self.model.variable, slot_name)[i])
        self.summary = tf.summary.merge_all()
        self.writer = self.get_writer()
        self.writer.add_graph(self.graph) # writter add graph
    # get a writer (set log file path here)
    def get_writer(self):
        # dir1 = ''.join([self.system.info, self.model.info, self.info])
        dir1 = 'test'
        dir2 = str(datetime.utcnow())
        return tf.summary.FileWriter('./log/'+dir1+'/'+dir2)












