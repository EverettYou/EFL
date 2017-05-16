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
        for shift in range(self.partitions): # translation
            for dir in (-1,1): # reflection
                # construct equivalent region
                blocks = frozenset((dir * x + shift)%self.partitions for x in self.blocks)
                eqreg = Region(blocks, self.partitions)
                yield eqreg
                # if half size, also consider complement
                if len(self) == self.partitions//2:
                    yield eqreg.complement()
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















