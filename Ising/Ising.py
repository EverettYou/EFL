''' Entanglement Feature Learning '''
# package required: numpy, tensorflow
import tensorflow as tf
import numpy as np
sess = tf.Session()
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
''' Entanglement Regions'''
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
    # map to Ising configuration
    def config(self):
        # generate an array of ones
        c = np.ones(self.partitions)
        c[list(self.blocks)] = -1. # set blocks to -1
        return c
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
    partitions :: int : number of entanglement mini regions
'''
class FreeFermion(object):
    def __init__(self, mass, size, partitions):
        assert -1. <= mass <= 1., 'mass must be in the range of [-1.,1.].'
        assert size%2 == 0, 'size must be even.'
        assert size%partitions == 0, 'size must be divisible by partitions.'
        self.mass = mass
        self.size = size
        self.regions = Regions(partitions)
        self.blockmap = np.arange(size).reshape((partitions, size//partitions))
        self.G = self.mkG()
        return
    # construct single-particle density matrix
    def mkG(self):
        u = np.tile([1.+self.mass, 1.-self.mass], self.size//2) # hopping
        A = np.roll(np.diag(u), 1, axis=1) # periodic rolling
        A = 1.j * (A - np.transpose(A)) # Hamiltonian
        (_, U) = np.linalg.eigh(A) # digaonalization
        V = U[:,:self.size//2] # take lower half levels
        return np.dot(V, V.conj().T) # construct density matrix
    # get sites given a region
    def sites(self, region):
        if len(region) > self.regions.partitions//2:
            return self.sites(region.complement())
        else:
            return self.blockmap[list(region)].flatten()
    # calculate entanglement entropy
    def S(self, region):
        # get indices of sites in the region
        inds = self.sites(region)
        if len(inds) == 0: return 0.
        # diagonalize reduced density matrix
        p = np.linalg.eigvalsh(self.G[inds,:][:,inds])
        # return 2nd Renyi entropy
        return -np.sum(np.log(abs(p**2 + (1.- p)**2)))
''' Ising Models
input:
    Lx :: int : x direction size
    Lz :: int : z direction size
'''
from itertools import product
class SquareModel(object):
    def __init__(self, Lx, Lz):
        self.Lx = Lx
        self.Lz = Lz
        self.inds = {site: i for (i, site) 
            in enumerate(product(range(Lx), range(Lz + 1)))}
        self.N = len(self.inds)
        self.cost = self.get_cost()
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
    # get cost function
    def get_cost(self):
        # construct adjacency matrices as constants
        AJx = tf.constant(
                np.array(
                    [self.mkA([(x, z) for x in range(self.Lx)], 1) 
                    for z in range(1, self.Lz)]),
                dtype = tf.float64, name = 'AJx')
        AJz = tf.constant(
                np.array(
                    [self.mkA([(x, z) for x in range(self.Lx)], 2) 
                    for z in range(self.Lz)]),
                dtype = tf.float64, name = 'AJz')
        Ahl = tf.constant(
                np.array(
                    [self.mkA([(x, 0)], 1) for x in range(self.Lx)]),
                dtype = tf.float64, name = 'Ah1')
        Ahr = tf.constant(
                np.array(
                    [self.mkA([(x, self.Lz)], 1) for x in range(self.Lx)]),
                dtype = tf.float64, name = 'Ah2')
        Abg = tf.constant(
                self.mkA(self.inds.keys(), 3),
                dtype = tf.float64, name = 'Abg')
        # configurations
        conf1 = tf.ones([self.Lx], dtype = tf.float64, name = 'conf1') # reference
        self.confs = tf.placeholder(dtype = tf.float64, name = 'confs')
        # specify couplings
        self.Jx = tf.Variable(np.ones(self.Lz - 1), dtype = tf.float64, name = 'Jx')
        self.Jz = tf.Variable(np.ones(self.Lz), dtype = tf.float64, name = 'Jz')
        self.Jl = tf.Variable(1., dtype = tf.float64, name = 'Jl')
        self.Jr = tf.constant(0., dtype = tf.float64, name = 'Jr')
        # set boundary conditions
        hl = tf.multiply(self.Jl, self.confs, name = 'hl')
        hr = tf.multiply(self.Jr, conf1, name = 'hr')
        hl1 = tf.multiply(self.Jl, conf1, name = 'hl1')
        hr1 = tf.multiply(self.Jr, conf1, name = 'hr1')
        # calculate weights
        wJx = self.wexp(self.Jx, name = 'w_Jx')
        wJz = self.wexp(self.Jz, name = 'w_Jz')
        whl = self.wexp(hl, name = 'w_hl')
        whr = self.wexp(hr, name = 'w_hr')
        whl1 = self.wexp(hl1, name = 'w_hl1')
        whr1 = self.wexp(hr1, name = 'w_hr1')
        # weight contraction
        wAJx = self.wdotA(wJx, AJx, name = 'A_Jx')
        wAJz = self.wdotA(wJz, AJz, name = 'A_Jz')
        wAhl = self.wdotA(whl, Ahl, name = 'A_hl')
        wAhr = self.wdotA(whr, Ahr, name = 'A_hr')
        wAhl1 = self.wdotA(whl1, Ahl, name = 'A_hl1')
        wAhr1 = self.wdotA(whr1, Ahr, name = 'A_hr1')
        # construct full adjacency matrix
        with tf.name_scope('A_com'):
            Acom = wAJx + wAJz + Abg
        with tf.name_scope('A'):
            A = Acom + wAhl + wAhr
        with tf.name_scope('A1'):
            A1 = Acom + wAhl1 + wAhr1
        # calcualte free energy
        F = self.free_energy(A, self.Jx, self.Jz, hl, hr, name = 'F')
        F1 = self.free_energy(A1, self.Jx, self.Jz, hl1, hr1, name = 'F1')
        # calculate cost function
        self.S = tf.placeholder(dtype = tf.float64, name = 'S')
        with tf.name_scope('RES'):
            cost = tf.reduce_sum(tf.square(F - F1 - self.S))
        return cost
    # coupling to weight
    def wexp(self, coupling, name = 'wexp'):
        with tf.name_scope(name):
            return tf.exp(-2*coupling)
    # weight contract with adjacency matrix
    def wdotA(self, w, A, name = 'wdotA'):
        with tf.name_scope(name):
            return tf.tensordot(w, A, axes = 1)
    # free energy
    def free_energy(self, A, Jx, Jz, hl, hr, name = 'F'):
        with tf.name_scope(name):
            lndet = tf.log(tf.matrix_determinant(A))
            Jbg = tf.reduce_sum(Jx, -1) + tf.reduce_sum(Jz, -1)
            hbg = tf.reduce_sum(hl, -1) + tf.reduce_sum(hr, -1)
            return -0.5*lndet - self.Lx*Jbg - hbg
        
        












