''' Entanglement Feature Learning '''
# package required: numpy, tensorflow
import tensorflow as tf
import numpy as np
tfgraph = tf.Graph()
sess = tf.Session(graph = tfgraph)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
''' Entanglement Region Object
input:
    blocks : a list/set of block indices
    partitions :: int : number of blocks in total
support method: complement, get equivalent class, convert configuration
'''
class Region(object):
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
''' Entanglement Region Server
input:
    partitions :: int : number of entanglement blocks in total
provides generators to yield entanglement subregions
'''
from itertools import combinations
class Regions(object):
    def __init__(self, partitions):
        self.partitions = partitions
    # representative subregions
    def representatives(self):
        # size of subregion from 0 to half 
        for rank in range(self.partitions//2 + 1):
            known = set() # hold the known equivalent subregions
            for blocks in combinations(range(self.partitions), rank):
                region = Region(blocks, self.partitions) # convert to region
                if region not in known: # if region unknown
                    # add equivalent regions to known
                    known.update(region.equivalences())
                    yield region # yield the representative region
    # consecutive subregions
    def consecutives(self):
        for x in range(self.partitions + 1):
            yield Region(range(0, x), self.partitions)
''' Entanglement Entropy (free fermion)
input:
    mass :: float : mass of the fermions, in [-1.,1.]
    size :: int : length of the 1D many-body state
'''
class FreeFermion(object):
    def __init__(self, mass, size):
        assert -1. <= mass <= 1., 'mass must be in the range of [-1.,1.].'
        assert size%2 == 0, 'size must be even.'
        self.mass = mass
        self.size = size
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
''' Ising Models
input:
    Lx :: int : x direction size
    Lz :: int : z direction size
'''
from itertools import product
class CylindricalModel(object):
    def __init__(self, Lx, Lz):
        self.Lx = Lx
        self.Lz = Lz
        # build a index map for sites
        self.inds = {site: i for (i, site) 
            in enumerate(product(range(Lx), range(Lz + 1)))}
        self.N = len(self.inds) # N keeps the number of sites
        self.partitions = Lx # Lx is also the number of partitions
        with tfgraph.as_default():
            self.build() # build model
    # make link, typ = 1,2,3
    def mklink(self, p1, typ):
        (x, z) = p1
        dir = 1.
        if typ == 1: # x++
            if x < self.Lx - 1:
                p2 = (x + 1, z)
            else: # if x = Lx - 1, x + 1 = 0
                p2 = (0, z)
                dir = -1.
        elif typ == 2: # z++
            p2 = (x, z + 1)
        elif typ == 3: # on-site
            p2 = p1
        else:
            raise ValueError('Unknown value {} for typ.'.format(typ))
        return (self.inds[p1], self.inds[p2], dir)
    # make adjacency matrix, typ = 1,2,3
    def mkA(self, ps, typ):
        # allocate a 4x4 matrix for unit cell
        uc = np.zeros((4, 4))
        # set uc
        if typ == 1: # x++
            uc[2, 0] = 1.
        elif typ == 2: # z++
            uc[3, 1] = 1.
        elif typ == 3: # on-site
            uc[0, 3] = 1.
            uc[1, 0] = 1.
            uc[1, 2] = 1.
            uc[2, 0] = 1.
            uc[3, 2] = 1.
            uc[3, 1] = 1.
        else:
            raise ValueError('Unknown value {} for typ.'.format(typ))
        # allocate for adjacency matrix
        A = np.zeros((self.N, self.N))
        # fill up matrix A
        for p1 in ps:
            (i1, i2, dir) = self.mklink(p1, typ)
            A[i1, i2] = dir
        # Kronecker product
        Auc = np.kron(A, uc)
        return Auc - Auc.T
    # build model
    def build(self):
        # construct adjacency matrices as constants
        TJx = tf.constant(
                np.array(
                    [self.mkA([(x, z) for x in range(self.Lx)], 1) 
                    for z in range(1, self.Lz)]),
                dtype = tf.float64)
        TJz = tf.constant(
                np.array(
                    [self.mkA([(x, z) for x in range(self.Lx)], 2) 
                    for z in range(self.Lz)]),
                dtype = tf.float64)
        Thl = tf.constant(
                np.array(
                    [self.mkA([(x, 0)], 1) for x in range(self.Lx)]),
                dtype = tf.float64)
        Thr = tf.constant(
                np.array(
                    [self.mkA([(x, self.Lz)], 1) for x in range(self.Lx)]),
                dtype = tf.float64)
        Abg = tf.constant(
                self.mkA(self.inds.keys(), 3),
                dtype = tf.float64)
        # configurations
        conf0 = tf.zeros([self.Lx], dtype = tf.float64, name = 'conf0') # reference 0
        conf1 = tf.ones([self.Lx], dtype = tf.float64, name = 'conf1') # reference 1
        self.confs = tf.placeholder(dtype = tf.float64, 
                shape = [None, self.Lx], name = 'confs')
        # specify couplings
        with tf.name_scope('J'):
            self.Jx = tf.Variable(0.1*np.ones(self.Lz - 1), dtype = tf.float64, name = 'Jx')
            self.Jz = tf.Variable(0.1*np.ones(self.Lz), dtype = tf.float64, name = 'Jz')
        self.h = tf.Variable(1., dtype = tf.float64, name = 'h')
        # set boundary conditions
        with tf.name_scope('hs'):
            hl = tf.multiply(self.h, self.confs, name = 'hl')
            hr = tf.multiply(self.h, conf0, name = 'hr')
            hl1 = tf.multiply(self.h, conf1, name = 'hl1')
            hr1 = tf.multiply(self.h, conf0, name = 'hr1')
        # generate weighted adjacency matrices
        with tf.name_scope('Ising_net'):
            AJx = self.dotA(self.Jx, TJx, name = 'A_Jx')
            AJz = self.dotA(self.Jz, TJz, name = 'A_Jz')
            Ahl = self.dotA(hl, Thl, name = 'A_hl')
            Ahr = self.dotA(hr, Thr, name = 'A_hr')
            Ahl1 = self.dotA(hl1, Thl, name = 'A_hl1')
            Ahr1 = self.dotA(hr1, Thr, name = 'A_hr1')
        # construct full adjacency matrix
            with tf.name_scope('A_com'):
                Acom = AJx + AJz + Abg
            with tf.name_scope('A'):
                A = Acom + Ahl + Ahr
            with tf.name_scope('A1'):
                A1 = Acom + Ahl1 + Ahr1
        # calcualte free energy and model entropy
        with tf.name_scope('free_energy'):
            self.F = self.free_energy(A, self.Jx, self.Jz, hl, hr, name = 'F')
            self.F1 = self.free_energy(A1, self.Jx, self.Jz, hl1, hr1, name = 'F1')
        self.Smdl = tf.subtract(self.F, self.F1, name = 'S_mdl')
        # calculate cost function
        self.Ssys = tf.placeholder(dtype = tf.float64, shape = [None], name = 'S_sys')
        with tf.name_scope('RES'):
            self.RES = tf.reduce_sum(tf.square(self.Smdl - self.Ssys))    
        self.forgetting_rate = tf.constant(0.1, dtype = tf.float64, name = 'forgetting_rate')
        with tf.name_scope('L2'):
            self.L2 = self.forgetting_rate*(
                tf.reduce_sum(tf.square(self.Jx)) 
                + tf.reduce_sum(tf.square(self.Jz)) 
                + tf.square(self.h))
        self.cost = tf.add(self.RES, self.L2, name = 'cost')
        # providing training handles
        self.initialize = tf.global_variables_initializer()
        self.learning_rate = tf.constant(0.01, dtype = tf.float64, name = 'learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train = self.optimizer.minimize(self.cost)
    # convert coupling to weight, and contract with adjacency matrix
    def dotA(self, f, A, name = 'dotA'):
        with tf.name_scope(name):
            return tf.tensordot(tf.exp(-2*f), A, axes = 1)
    # free energy
    def free_energy(self, A, Jx, Jz, hl, hr, name = 'F'):
        with tf.name_scope(name):
            lndet = tf.log(tf.matrix_determinant(A))
            Jbg = tf.reduce_sum(Jx, -1) + tf.reduce_sum(Jz, -1)
            hbg = tf.reduce_sum(hl, -1) + tf.reduce_sum(hr, -1)
            return -0.5*lndet - self.Lx*Jbg - hbg
''' Entanglement Feature Learning
input:
    system : object that has a method S to return the entropy
    model : a model that takes takes  
'''
class EFL(object):
    def __init__(self, system, model):
        self.system = system
        self.model = model
        self.size = system.size
        self.partitions = model.partitions
        assert self.size%self.partitions == 0, 'size {0} is not divisible by partitions {1}.'.format(self.size, self.partitions)
        # set up a entanglement region server
        self.regions = Regions(self.partitions)
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
    # send a batch of training set
    def training_set(self, method, *args):
        # prepare empty lists for configs and sysS
        confs = []
        Ssys = []
        # go through all regions in the batch
        for region in getattr(self.regions, method)(*args):
            # configuration of Ising boundary
            confs.append(region.config())
            # entanglement entropy from system
            Ssys.append(self.system.S(self.sites(region)))
        # return data as a dict
        return {self.model.confs: np.array(confs), self.model.Ssys: np.array(Ssys)}












