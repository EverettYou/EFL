''' Entanglement Feature Learning '''
# package required: numpy, theano
import numpy
import theano
import theano.tensor as T
import os
''' RBM (Kernel)
* unbiased, spins take values in {-1,+1}.
Nv::int: number of visible units
Nh::int: number of hidden units
Nb::int: number of samples in a batch
W::T.matrix: theano shared weight matrix for a kernel in CDBN.
             None for standalone RBMs.
persistent::T.matrix: theano shared persistent states.
method::str: learning method, 'CD' or 'PCD'
Markov_steps::int: number of Markov steps in Gibbs sampling
'''
class RBM(object):
    ''' RMB Constructor '''
    def __init__(self,
        Nv = 4, # number of visible units
        Nh = 2, # number of hidden unites
        Nb = 20, # number of samples in a batch
        W = None, # theano shared weight matrix
        persistent = None, # theano shared persistent states
        method = 'PCD', # learning method, 'CD' or 'PCD'
        Markov_steps = 15, # Markov_steps to go in Gibbs sampling
        numpy_rng = None, theano_rng = None):
        # set random number generators
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState()
        # symbolize weight matrix
        if W is None:
            self.W = self.init_W('random', Nv, Nh, numpy_rng)
        elif isinstance(W, str):
            self.W = self.init_W(W, Nv, Nh, numpy_rng)
        else: # assuming W is a theano shared matrix
            self.W = W
            # in this case, Nv, Nh are override by the shape of W
            Nv, Nh = W.get_value().shape
        # set number of units
        self.Nv = Nv
        self.Nh = Nh
        # determine persistent states
        if method == 'CD':
            self.persistent = None
        elif method == 'PCD':
            if persistent is None:
                self.persistent = self.init_persistent(Nb, Nh)
            else: # assume persistant is a theano shared matrix
                self.persistent = persistent
                # in this case, Nb is override by the shape of persistent
                Nb, Nh1 = persistent.get_value().shape
                # check Nh consistency
                if Nh1 != Nh:
                    raise ValueError("The shape (%d,%d) of persistent is inconsistent with the shape (%d,%d) of W. They should have the same number of hidden units."%(Nb,Nh1,Nv,Nh))
        else: # unknown method
            raise ValueError("%s is not a known learning method. Learning method should be 'CD' or 'PCD'."%mode)
        self.Nb = Nb
        # set Markov steps
        self.Markov_steps = Markov_steps
        # set theano random number generator
        if theano_rng is None:
            seed = numpy_rng.randint(2**30)
            self.theano_rng = T.shared_randomstreams.RandomStreams(seed)
        else:
            self.theano_rng = theano_rng
        # symbolize visible input for RBM
        self.input = T.matrix('input',dtype=theano.config.floatX)
        # symbolize rates
        self.lr = T.scalar('lr',dtype=theano.config.floatX)
        self.fr = T.scalar('fr',dtype=theano.config.floatX)
        # initialize data storages
        self._batch = numpy.empty(shape=(Nb, Nv),dtype=theano.config.floatX)
        self._batch_top = 0
        # initialize theano functions
        self._learn = None # learning function
        self._bottomup = None
    # construct initial weight matrix variable
    def init_W(self, mode, Nv, Nh, numpy_rng):
        if mode == 'random': # random initialization
            W_raw = numpy_rng.uniform(low=0., high=1., size=(Nv, Nh))
            W_raw *= numpy.sqrt(6./(Nv+Nh))
        elif mode == 'local': # local initialization
            vx = numpy.arange(-Nv/2.,Nv/2.,1.)+0.5
            hx = numpy.arange(-Nh/2.,Nh/2.,1.)+0.5
            hxs, vxs = numpy.meshgrid(hx, vx)
            W_raw = numpy.exp(-(hxs-vxs)**2/2)
        elif mode == 'multiworld': # multiworld initialization
            hw = numpy.exp(-numpy.linspace(0.,1.,Nh))
            W_raw = numpy.repeat([hw],Nv,axis=0)
        else: # unknown mode
            raise ValueError("%s is not a known mode for weight matrix initializaiton. Use 'random', 'local', 'multiworld' to initialize weight matrix."%mode)
        W_mat = numpy.asarray(W_raw, dtype=theano.config.floatX)
        return theano.shared(value=W_mat, name='W', borrow=True)
    # construct initial persistent states
    def init_persistent(self, Nb, Nh):
        ones = numpy.ones((Nb, Nh), dtype=theano.config.floatX)
        return theano.shared(ones, name='persistent', borrow=True)
    ''' RBM Physical Dynamics '''
    # calculate free energy
    def free_energy(self, v_samples):
        # F = - sum_j ln(2*cosh(sum_i v_i W_ij))
        return -T.sum(T.log(2*T.cosh(T.dot(v_samples, self.W))), axis=1)
    # Propagate visible spin local fields upwards to hidden spins
    def propup(self, v_samples):
        local_fields = T.dot(v_samples, self.W)
        return [local_fields, T.tanh(local_fields)]
    # Infer hidden spins given visible spins
    def sample_h_given_v(self, v0_samples):
        # compute local fields and expectations of hiddens
        local_fields_h1, h1_means = self.propup(v0_samples)
        h1_probabilities = (h1_means+1)/2
        # get a sample of hiddens for once (n=1) with probability p
        h1_samples = self.theano_rng.binomial(
            size=h1_means.shape,
            n=1, p=h1_probabilities,
            dtype=theano.config.floatX)*2-1
        return [local_fields_h1, h1_means, h1_samples]
    # Propagate hidden spin local fields downwards to visible spins
    def propdown(self, h_samples):
        local_fields = T.dot(h_samples, self.W.T)
        return [local_fields, T.tanh(local_fields)]
    # Infer visible spins given hidden spins
    def sample_v_given_h(self, h0_samples):
        # compute local fields and expectations of visibles
        local_fields_v1, v1_means = self.propdown(h0_samples)
        v1_probabilities = (v1_means+1)/2
        # get a sample of visibles for once (n=1) with probability p
        v1_samples = self.theano_rng.binomial(
            size=v1_means.shape,
            n=1, p=v1_probabilities,
            dtype=theano.config.floatX)*2-1
        return [local_fields_v1, v1_means, v1_samples]
    # One step of Gibbs sampling from hiddens
    def gibbs_hvh(self, h0_samples):
        local_fields_v1, v1_means, v1_samples = self.sample_v_given_h(h0_samples)
        local_fields_h1, h1_means, h1_samples = self.sample_h_given_v(v1_samples)
        return [local_fields_v1, v1_means, v1_samples,
                local_fields_h1, h1_means, h1_samples]
    # One step of Gibbs sampling from visibles
    def gibbs_vhv(self, v0_samples):
        local_fields_h1, h1_means, h1_samples = self.sample_h_given_v(v0_samples)
        local_fields_v1, v1_means, v1_samples = self.sample_v_given_h(h1_samples)
        return [local_fields_h1, h1_means, h1_samples,
                local_fields_v1, v1_means, v1_samples]
    ''' Implement CD-k or PCD-k  '''
    # Get cost and updates
    def get_cost_updates(self):
        # determine chain start
        if self.persistent is None: # CD, generate new state from input
            local_fields_ph, ph_meands, ph_samples = self.sample_h_given_v(self.input)
            chain_start = ph_samples
        else: # PCD, use old state of the chain
            chain_start = self.persistent
        # relax to equilibrium by Gibbs sampling
        ([local_fields_nv_chain, nv_means_chain, nv_samples_chain,
          local_fields_nh_chain, nh_means_chain, nh_samples_chain],
         updates) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None,None,None,None,None,chain_start],
            n_steps=self.Markov_steps, # k = Markov steps
            name='gibbs_hvh')
        if self.persistent: # PCD, update persistent to new hiddens
            updates[self.persistent] = nh_samples_chain[-1]
        # take chain end
        local_fields_nv = local_fields_nv_chain[-1]
        nv_samples = nv_samples_chain[-1]
        # cost = free energy (clamped) - free energy (unclamped)
        clamped = T.mean(self.free_energy(self.input))
        unclamped = T.mean(self.free_energy(nv_samples))
        cost = clamped - unclamped
        # compute the gradients on weights
        grad_W = T.grad(cost, self.W, consider_constant=[nv_samples])
        # add W update rules to updates dict
        updates[self.W] = T.nnet.relu(self.W * (1-self.fr) - grad_W * self.lr)
        # determine cross entropy
        if self.persistent: # PCD, use pseudo-likelihood
            xent = self.get_pseudo_xentropy(updates)
        else: # CD, use reconstruction entropy
            xent = self.get_xentropy(updates, local_fields_nv)
        return cost, xent, updates    
    ''' Estimate Cross-Entropy '''
    # Pseudo-likelihood cross enetropy
    def get_pseudo_xentropy(self, updates):
        # index of bit i to flip
        indx = theano.shared(value=0, name='indx')
        # binarize the input
        xi = T.sgn(self.input)
        # calculate free energy for the given configuration
        fe_xi = self.free_energy(xi)
        # flip the spins at indx
        xi_flip = T.set_subtensor(xi[:,indx], -xi[:,indx])
        # calculated free energy for flipped configuration
        fe_xi_flip = self.free_energy(xi_flip)
        # local entropy * Nv = total entropy
        entropy = -T.mean(T.log(T.nnet.sigmoid(fe_xi_flip-fe_xi)))*self.Nv
        # increment indx mod Nv as part of updates
        updates[indx] = (indx + 1)%self.Nv
        return entropy
    # Reconstruction cross entropy
    def get_xentropy(self, updates, local_fields_nv):
        nv_prob = T.nnet.sigmoid(2*local_fields_nv)
        iv_prob = (self.input+1)/2
        entropy = -T.mean(
            T.sum(iv_prob*T.log(nv_prob)+(1-iv_prob)*T.log(1-nv_prob),axis = 1))
        return entropy
    ''' Training Functionalities '''
    # Learning function
    def learn(self, learning_rate = 0.1, forgetting_rate = 0.):
        # takes learning samples from self._batch
        if self._learn is None: # if learning function not constructed
            # prepare learning function outputs
            cost, xent, updates = self.get_cost_updates()
            # compile learning function
            self._learn = theano.function(
                [self.input, self.lr, self.fr],
                [cost, xent],
                updates = updates,
                name = 'RBM._learn')
        # now learning function is ready, apply it to learning samples
        cost, xent = self._learn(self._batch, learning_rate, forgetting_rate)
        return numpy.asscalar(cost), numpy.asscalar(xent)
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










