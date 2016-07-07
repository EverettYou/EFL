    # calculate free energy
    def free_energy(self, v_samples):
        # F = - sum_j ln(2*cosh(sum_i v_i W_ij))
        return -T.sum(T.log(2*T.cosh(T.dot(v_samples, self.W))), axis=1)

    
    # Training function, prepare self._batch for learning
    def train(self, train_set, # numpy.array of training configs
        learning_rate = 0.1, # learning rate
        forgetting_rate = 0.): # forgetting rate
        # get shape of training set
        Ns, Nv1 = train_set.shape
        sample_top = 0
        # check shape consistency
        if Nv1 != self.Nv:
            raise ValueError("The shape (%d,%d) of train_set is inconsistent with the number of visible units %d in this RBM."%(Ns,Nv1,self.Nv))
        # go trough the traning set
        costs = []
        xents = []
        while sample_top < Ns:
            # get batch capacity and sample load
            batch_top = self._batch_top
            batch_capacity = self.Nb - batch_top
            sample_load = Ns - sample_top
            # if fewer sample to fill up the batch
            if sample_load < batch_capacity:
                # dump rest of the samples
                self._batch[batch_top:batch_top + sample_load,:] = train_set[sample_top:,:]
                # update batch and sample tops
                self._batch_top = batch_top + sample_load
                sample_top = Ns # will exit while loop
                # end without learning
            else: # sample_load >= batch_capacity
                # fill up the batch
                self._batch[batch_top:,:] = train_set[sample_top:sample_top + batch_capacity,:]
                # initiate learning
                cost, xent = self.learn(learning_rate, forgetting_rate)
                costs.append(cost)
                xents.append(xent)
                # update batch and sample tops
                self._batch_top = 0
                sample_top = sample_top + batch_capacity
        cost = numpy.mean(costs)
        xent = numpy.mean(xents)
        return cost, xent
    ''' Inferences '''
    # Bottomup inference: generate hidden activation from visible
    def bottomup(self, v_means):
        if self._bottomup is None: # if inference function not constructed
            local_fields_h, h_means = self.propup(self.input)
            self._bottomup = theano.function([self.input],h_means)
        return self._bottomup(v_means)
''' DBN 
layers_sizes::list: a list of integers specifying the number of units in each layer
'''
class DBN(object):
    ''' DBN Constructor '''
    def __init__(self,
        layers_sizes = [], # number of units in each layer
        numpy_rng = None, theano_rng = None):
        # initialize RBM containers
        self.rbms = []
        self.n_layers = len(layers_sizes) - 1
        assert self.n_layers >= 2 # at least two layers
        # set theano rand generator
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState()
        if theano_rng is None:
            seed = numpy_rng.randint(2**30)
            self.theano_rng = T.shared_randomstreams.RandomStreams(seed)
        # build RBM layers
        for i in range(self.n_layers):
            Nv = layers_sizes[i]
            Nh = layers_sizes[i + 1]
            # construct an RBM
            rbm = RBM(Nv=Nv,Nh=Nh,W='local',
                numpy_rng=numpy_rng,theano_rng=theano_rng)
            rbm.layer = i
            self.rbms.append(rbm)
    ''' DBN Physical Dynamics '''
    def reconstruct(self, samples):
        for rbm in self.rbms:
            local_fields, samples = rbm.propup(samples)
        for rbm in reversed(self.rbms):
            local_fields, samples = rbm.propdown(samples)
        return samples
    
    ''' Training '''
    # pretraining RBMs greedily
    def pretrain(self, train_set, epoches = 7, lrs =[], frs=[]):
        # train RBM from bottom up
        for rbm in self.rbms: 
            print('RBM layer %d ---'%rbm.layer)
            for epoch in range(epoches):
                try: # get learning rate
                    lr = lrs[epoch]
                except:
                    lr = 0.1
                try: # get forgetting rate
                    fr = frs[epoch]
                except:
                    fr = 0.
                # train RBM
                cost, xent = rbm.train(train_set,
                           learning_rate=lr,
                           forgetting_rate=fr)
                print('Epoch %d: '%epoch, 'cost = %f, xent = %f'%(cost, xent))
            # generate train_set for the next layer
            train_set = rbm.bottomup(train_set)





''' Orthogonalize
by QR decomposition (column-wise) '''
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply
class Orthogonalize(Op): # define theano Op
    _numpyqr = staticmethod(numpy.linalg.qr) # static numpy qr
    __props__ = () # no properties for this Op
    # creates an Apply node
    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2, "The input of qr function should be a matrix."
        y = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [y])
    # Phython implementation
    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (y,) = outputs
        assert x.ndim == 2, "The input of qr function should be a matrix."
        q, r = self._numpyqr(x,'reduced') # QR decomposition
        # d = diagonal of r as vector
        if r.shape[0] < r.shape[1]:
            d = r[:, 0]
        else:
            d = r[0]
        d.strides = (r.strides[0] + r.strides[1],)
        # column-wise multiply d to q
        q *= d
        # if q columns < x columns, pad zero columns from the right
        if q.shape[1] < x.shape[1]:
            q = numpy.pad(q, ((0,0),(0,x.shape[1]-q.shape[1])),'constant')
        y[0] = q # set output to q
    # string representation
    def __str__(self):
        return 'Orthogonalize'
# alias
orth = Orthogonalize()
