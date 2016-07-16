''' Entanglement Feature Learning '''
# package required: numpy, theano
import numpy
import theano
import theano.tensor as T
import os
numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
''' RBM (method CD)
* unbiased, spins take values in {-1,+1}.
Nv::int: number of visible units
Nh::int: number of hidden units
W::T.matrix: theano shared weight matrix for a kernel in CDBN.
             None for standalone RBMs.
type::string: type of RBM in a DBM, can be 'default', 'bottom', 'top', 'intermediate' 
Markov_steps::int: number of Markov steps in Gibbs sampling
'''
class RBM(object):
    Markov_steps = 15 # set Markov steps
    ''' RMB Constructor '''
    def __init__(self,
            Nv = 1, Nh = 1, W = None, input = None, type = 'default',
            numpy_rng = None, theano_rng = None):
        # set number of units
        self.Nv = Nv
        self.Nh = Nh
        # set random number generators
        self.init_rng(numpy_rng, theano_rng)
        # symbolize weight matrix
        self.W = self.init_W(W)
        self.type = type # set RBM type
        # symbolize visible input for RBM
        if input is None:
            self.input = T.matrix('RBM.input',dtype=theano.config.floatX)
        else:
            self.input = input
        self.batch_size, _ = self.input.shape # set batch size from input
        self.output = self.propup(self.input)
        # symbolize rates
        self.lr = T.scalar('lr',dtype=theano.config.floatX)
        self.fr = T.scalar('fr',dtype=theano.config.floatX)
        # build learning function for default RBM 
        if self.type is 'default':
            cost, updates = self.get_cost_updates()
            self.learn = theano.function(
                [self.input, 
                 theano.In(self.lr, value=0.1),
                 theano.In(self.fr, value=0.)],
                cost, updates = updates, name = 'RBM.learn')
    # initialize random number generator
    def init_rng(self, numpy_rng, theano_rng):
        # set numpy random number generator
        if numpy_rng is None:
            self.numpy_rng = numpy.random.RandomState()
        else:
            self.numpy_rng = numpy_rng
        # set theano random number generator
        if theano_rng is None:
            seed = self.numpy_rng.randint(2**30)
            self.theano_rng = T.shared_randomstreams.RandomStreams(seed)
        else:
            self.theano_rng = theano_rng
    # construct initial weight matrix variable
    def init_W(self, W):
        def build_W(mode):
            Nv, Nh = self.Nv, self.Nh
            if mode == 'random': # random initialization
                W_raw = self.numpy_rng.uniform(low=0., high=1., size=(Nv, Nh))
                W_raw *= numpy.sqrt(6./(Nv+Nh))
            elif mode == 'local': # local initialization
                vx = numpy.arange(0.5/Nv,1.,1./Nv)
                hx = numpy.arange(0.5/Nh,1.,1./Nh)
                hxs, vxs = numpy.meshgrid(hx, vx)
                W_raw = numpy.exp(-(hxs-vxs)**2*(5*Nh**2))
            elif mode == 'localrand':
                vx = numpy.arange(0.5/Nv,1.,1./Nv)
                hx = numpy.arange(0.5/Nh,1.,1./Nh)
                hxs, vxs = numpy.meshgrid(hx, vx)
                W_raw = numpy.exp(-(hxs-vxs)**2*(5*Nh**2))
                W_raw *= self.numpy_rng.uniform(low=0., high=2., size=(Nv, Nh))
            elif mode == 'multiworld': # multiworld initialization
                hw = numpy.exp(-numpy.linspace(0.,1.,Nh))
                W_raw = numpy.repeat([hw],Nv,axis=0)
            else: # unknown mode
                raise ValueError("%s is not a known mode for weight matrix initializaiton. Use 'random', 'local', 'multiworld' to initialize weight matrix."%mode)
            W_mat = numpy.asarray(W_raw, dtype=theano.config.floatX)
            return theano.shared(value=W_mat, name='W', borrow=True) 
        if W is None:
            return build_W('localrand')
        elif isinstance(W, str):
            return build_W(W)
        else: # assuming W is a theano shared matrix
            # in this case, Nv, Nh are override by the shape of W
            self.Nv, self.Nh = W.get_value().shape
            return W
    ''' RBM Physical Dynamics '''
    # Propagate visible configs upwards to hidden configs
    def propup(self, v_samples):
        local_fields = T.dot(v_samples, self.W) # local fields for hiddens
        # double local field for bottom, intermediate
        if self.type in {'bottom', 'intermediate'}:
            local_fields *= 2
        h_means = T.tanh(local_fields)
        return h_means
    # Infer hidden samples given visible samples
    def sample_h_given_v(self, v_samples):
        h_means = self.propup(v_samples) # expectations of hiddens
        h_probs = (h_means+1)/2 # probabilities for hiddens
        # get a sample of hiddens for once (n=1) with probability p
        h_samples = self.theano_rng.binomial(
            size=h_means.shape, n=1, p=h_probs,
            dtype=theano.config.floatX)*2-1
        return h_means, h_samples
    # Propagate hidden configs downwards to visible configs
    def propdown(self, h_samples):
        local_fields = T.dot(h_samples, self.W.T) # local fields for visibles
        # double local field for top, intermediate
        if self.type in {'top', 'intermediate'}:
            local_fields *= 2
        v_means = T.tanh(local_fields)
        return v_means
    # Infer visible samples given hidden samples
    def sample_v_given_h(self, h_samples):
        v_means = self.propdown(h_samples) # expectations of visibles
        v_probs = (v_means+1)/2 # probabilities for visibles
        # get a sample of visibles for once (n=1) with probability p
        v_samples = self.theano_rng.binomial(
            size=v_means.shape, n=1, p=v_probs,
            dtype=theano.config.floatX)*2-1
        return v_means, v_samples
    # One step of Gibbs sampling from hiddens
    def gibbs_hvh(self, h0_samples):
        v1_means, v1_samples = self.sample_v_given_h(h0_samples)
        h1_means, h1_samples = self.sample_h_given_v(v1_samples)
        return [v1_means, v1_samples, h1_means, h1_samples]
    # One step of Gibbs sampling from visibles
    def gibbs_vhv(self, v0_samples):
        h1_means, h1_samples = self.sample_h_given_v(v0_samples)
        v1_means, v1_samples = self.sample_v_given_h(h1_samples)
        return [h1_means, h1_samples, v1_means, v1_samples]
    ''' Implement CD-k '''
    # Get cost and updates
    def get_cost_updates(self):
        # CD, generate new chain start from input
        h0_means, h0_samples = self.sample_h_given_v(self.input)
        # relax to equilibrium by Gibbs sampling
        ([v_means_chain, v_samples_chain, h_means_chain, h_samples_chain],
         updates) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None,None,None,h0_samples],
            n_steps=self.Markov_steps, # k = Markov steps
            name='gibbs_hvh')
        # take chain end
        vf_samples = v_samples_chain[-1]
        vf_means = v_means_chain[-1]
        hf_means = h_means_chain[-1]
        # compute negative gradient on weights
        dW = (T.dot(self.input.T,h0_means) - T.dot(vf_samples.T,hf_means))/self.batch_size
        # add W update rules to updates dict
        updates[self.W] = T.nnet.relu((1-self.fr)*self.W + self.lr*dW)
        # cost function: recontruction cross entropy (per visible)
        v0_probs = (self.input+1)/2 # initial visible probabilities
        vf_probs = (vf_means+1)/2   # final visible probabilities
        cost = T.mean(T.nnet.binary_crossentropy(vf_probs,v0_probs))
        return cost, updates
''' DBM
Nls::list of int: number of units in each layer
-------
L: number of RMB layers = len(Nls)-1
layer index l: visible l = 0; hidden l = 1, ..., L
'''
class DBM(object):
    Markov_steps = 15 # set Markov steps
    MC_samples = 20 # number of MC samples
    ''' DMB Constructor '''
    def __init__(self, Nls, numpy_rng = None, theano_rng = None):
        # set structral constants
        self.Nls = Nls
        self.L = len(Nls)-1 # number of RBMs
        assert self.L >= 2  # at least two layers of RBMs 
        # set random number generators
        self.init_rng(numpy_rng, theano_rng)
        # initialize MC configurations
        self.MC_configs = self.init_MC_configs()
        # symbolize visible input for DBM
        self.input = T.matrix('DBM.input',dtype=theano.config.floatX)
        self.lr = T.scalar('lr',dtype=theano.config.floatX)
        # build RBMs
        self.build_rbms()
        # collect weight matrixes
        self.Ws = [rbm.W for rbm in self.rbms]
        # build DBM finetuning
        cost, updates = self.get_cost_update()
        self.learn = theano.function([self.input, self.lr], cost, updates=updates)
    # initialize random number generator
    def init_rng(self, numpy_rng, theano_rng):
        # set numpy random number generator
        if numpy_rng is None:
            self.numpy_rng = numpy.random.RandomState()
        else:
            self.numpy_rng = numpy_rng
        # set theano random number generator
        if theano_rng is None:
            seed = self.numpy_rng.randint(2**30)
            self.theano_rng = T.shared_randomstreams.RandomStreams(seed)
        else:
            self.theano_rng = theano_rng
    # initialize MC configs
    def init_MC_configs(self):
        MC_configs = []
        # randomly initialize MC configs
        for Nl in self.Nls:
            rand = self.numpy_rng.binomial(n=1,p=0.5,
                            size=(self.MC_samples,Nl))
            rand = numpy.asarray(rand*2-1,dtype=theano.config.floatX)
            MC_configs.append(theano.shared(value=rand, borrow=True))
        return MC_configs
    # construct RBM layers
    def build_rbms(self):
        self.rbms = [] # initialize RBM container
        layer_input = self.input # layer input holder
        for l in range(self.L):
            Nv = self.Nls[l]   # visible units
            Nh = self.Nls[l+1] # hidden units
            # ascribe RBM type
            if l == 0:
                rbm_type = 'bottom'
            elif l == self.L-1:
                rbm_type = 'top'
            else:
                rbm_type = 'intermediate'
            # build RBM
            rbm = RBM(Nv, Nh, input=layer_input, type=rbm_type,
                    numpy_rng=self.numpy_rng, theano_rng=self.theano_rng)
            rbm.layer = l # set layer index
            layer_input = rbm.output # get output from RBM for the next layer
            # build learning function
            cost, updates = rbm.get_cost_updates()
            rbm.learn = theano.function(
                [self.input,
                 theano.In(rbm.lr, value=0.1),
                 theano.In(rbm.fr, value=0.)],
                cost, updates = updates, name = 'RBM.learn')
            self.rbms.append(rbm)
    # pretrain RBMs
    def pretrain(self, data_source, epochs = 7, lrs=[], frs=[]):
        assert type(data_source) is Server, 'Input data_source must be a Server.'
        assert data_source.Nv==self.Nls[0], 'Sample size %d does not fit the visible layer size %d'%(data_source.Nv, self.Nls[0])
        # layer-wise pretrain
        for rbm in self.rbms:
            print('Pretraining layer %d:'%rbm.layer)
            cost_avg0 = 10000.
            # go through pretraining epoches
            for epoch in range(epochs):
                # get learning rate and forgetting rate
                try: # learning rate
                    lr = lrs[epoch]
                except: # default 0.1
                    lr = 0.1
                try: # forgetting rate
                    fr = frs[epoch]
                except: # default 0.
                    fr = 0.
                # go through training set served by data_source
                costs = []
                for batch in data_source:
                    cost = rbm.learn(batch, lr, fr)
                    costs.append(numpy.asscalar(cost))
                # calculate cost average
                cost_avg = numpy.mean(costs)
                print('    Epoch %d: cost = %f'%(epoch,cost_avg/numpy.log(2)))
                # if effectively no progress
                if abs(cost_avg0-cost_avg)<0.001: 
                    break # quit next epoch
                else: # otherwise save cost_avg and start next epoch
                    cost_avg0 = cost_avg
    # finetune DBM
    def finetune(self, data_source, epochs = 5, lrs=[]):
        assert type(data_source) is Server, 'Input data_source must be a Server.'
        assert data_source.Nv==self.Nls[0], 'Sample size %d does not fit the visible layer size %d'%(data_source.Nv, self.Nls[0])
        # go through pretraining epoches
        lr = 0.5
        for epoch in range(epochs):
            # get learning rate and forgetting rate
            try: # learning rate
                lr = lrs[epoch]
            except: # default
                lr = lr/2
            # go through training set served by data_source
            costs = []
            for batch in data_source:
                cost = self.learn(batch, lr)
                costs.append(numpy.asscalar(cost))
            # calculate cost average 
            cost_avg = numpy.mean(costs)
            print('    Epoch %d: cost = %f'%(epoch,cost_avg/numpy.log(2)))
    ''' DBM Physical Dynamics '''
    # propagate DBM configs onto a specific layer
    def proponto(self, l, configs):
        if l == 0: # bottom (visible) layer inference
            local_fields = T.dot(configs[1], self.Ws[0].T)
        elif l == self.L: # top layer inference
            local_fields = T.dot(configs[self.L-1], self.Ws[self.L-1])
        else: # intemediate layer inference
            local_fields = T.dot(configs[l-1], self.Ws[l-1]) + T.dot(configs[l+1], self.Ws[l].T)
        hl_means = T.tanh(local_fields)
        return hl_means
    # infer samples on a specific layer
    def sample(self, l, configs):
        # get expectations of layer l
        hl_means = self.proponto(l, configs)
        hl_probs = (hl_means+1)/2 # convert to probabilities
        # get a sample for once (n=1) with probability p
        hl_samples = self.theano_rng.binomial(
            size=hl_means.shape, n=1, p=hl_probs,
            dtype=theano.config.floatX)*2-1
        return hl_samples
    # one step gibbs sampling for all layers
    # soft version: for meanfield updates
    def soft_gibbs(self, configs):
        # no even-odd partition needed for meanfield updates
        return [self.proponto(l, configs) for l in range(0,self.L+1)]
    # hard version: for Monte Carlo updates
    def hard_gibbs(self, configs):
        # allocate empty configs
        new_configs = [None]*(self.L+1)
        # first sample odd layers (from old configs as input)
        for l in range(1,self.L+1,2):
            new_configs[l] = self.sample(l, configs)
        # then sample even layers (from new configs of odd layers)
        for l in range(0,self.L+1,2):
            new_configs[l] = self.sample(l, new_configs)
        return new_configs
    ''' DBM Fine Tuning '''
    # get meanfield fixed point
    def get_cost_update(self):
        # measure correlation from given configs
        def correlation(configs):
            return [T.dot(h0.T, h1)/h1.shape[0] 
                    for h0, h1 in zip(configs[0:],configs[1:])]
        # initial meanfield configs generated by RGM inference
        mu_configs = [rbm.output for rbm in self.rbms]
        # define a meanfield step function to wrap the data
        def meanfield_step(*mu_configs):
            # combine with input to make the full config
            configs = self.soft_gibbs([self.input] + list(mu_configs))
            return configs[1:]
        # meanfield iteration
        results, _ = theano.scan(
                fn=meanfield_step,
                outputs_info=mu_configs,
                n_steps=4)
        # take the final result and construct meanfield configs
        mf_configs = [self.input] +[T.sgn(chain[-1]) for chain in results] 
        # define a MC step function to wrap the data
        def MC_step(*configs):
            return self.hard_gibbs(list(configs))
        # MC iteration
        results, updates = theano.scan(
                fn=MC_step,
                outputs_info=self.MC_configs,
                n_steps=self.Markov_steps)
        new_configs = [chain[-1] for chain in results] # take the final result
        # make update dict for MC configs
        for old, new in zip(self.MC_configs, new_configs):
            updates[old] = new
        # compute negative gradient on weights
        dWs = [p - n for p, n in zip(correlation(mf_configs), correlation(new_configs))]
        # add update rules
        for W, dW in zip(self.Ws, dWs):
            updates[W] = T.nnet.relu(W + self.lr*dW)
        # cost function: recontruction cross entropy (per visible)
        v0_probs = (self.input+1)/2 # initial visible probabilities
        vf_probs = (self.proponto(0,[[]]+mu_configs)+1)/2   # final visible probabilities
        cost = T.mean(T.nnet.binary_crossentropy(vf_probs,v0_probs))
        return cost, updates
''' Server (data server)
dataset::numpy matrix
'''
class Server(object):
    # build data server
    def __init__(self, dataset, batch_size = 20):
        # initialize dataset
        self.dataset = numpy.asarray(dataset,dtype=theano.config.floatX)
        self.data_size, self.Nv = self.dataset.shape
        self.batch_size = batch_size
        self.batch = numpy.empty(shape=(self.batch_size, self.Nv),
                dtype=theano.config.floatX)
        self.batch_top = 0
        self.data_top = 0
    # define iterator 
    def __iter__(self):
        return self
    def __next__(self):
        # if dataset not exhausted
        if self.data_top < self.data_size:
            # get current capacity and data load
            batch_capacity = self.batch_size - self.batch_top
            data_load = self.data_size - self.data_top
            # if fewer data to fill up the batch
            if data_load < batch_capacity:
                # dump rest of the data
                self.batch[self.batch_top:self.batch_top+data_load,:] = self.dataset[self.data_top:,:]
                # update batch top
                self.batch_top += data_load
                self.data_top = 0 # rewind to the begining
                # will not yeild new batch
                raise StopIteration
            else: # more data left to fill batch
                # fill up the batch
                self.batch[self.batch_top:,:] = self.dataset[self.data_top:self.data_top+batch_capacity,:]
                # reset batch top
                self.batch_top = 0
                self.data_top += batch_capacity
                # yeild the filled batch
                return self.batch
        else: # dataset exhausted -> restart
            self.data_top = 0 # rewind to the begining
            # will not yeild new batch
            raise StopIteration
    # add data to the dataset
    def add_data(self, dataset):
        # update dataset
        self.dataset = numpy.append(self.dataset, dataset, axis = 0)
        self.data_size, self.Nv = self.dataset.shape

