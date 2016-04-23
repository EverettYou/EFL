''' Entanglement Feature Learning '''
# package required: numpy, theano
import numpy
import theano
import theano.tensor as T
import os
''' RBM (Kernel)

* unbiased, spins take values in {-1,+1}.

RBM.input::T.matrix: a batch of training configurations
RBM.Nv::int: number of visible units
RBM.Nh::int: number of hidden units
RBM.W::T.matrix: shared weight matrix'''
class RBM(object):
    ''' RMB constructor 
    input::T.matrix: a batch of training configurations,
        each row is a training sample of visible spins.
        None for standalone RBMs.
    Nv::int: number of visible units
    Nh::int: number of hidden units
    W::T.matrix: shared weight matrix for a kernel in CDBN.
        None for standalone RBMs.'''
    def __init__(self,
        input = None,
        Nv = 4,
        Nh = 2,
        W = None,
        numpy_rng = None,
        theano_rng = None):
        
        # set number of units
        self.Nv = Nv
        self.Nh = Nh
        # set random number generators
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)
        if theano_rng is None:
            theano_rng = theano.tensor.shared_randomstreams.RandomStreams(numpy_rng.randint(2**30))
        # initialize weight matrix
        if W is None:
############# init_W function should be rewritten!
#             init_W = init_W_rnd(Nv, Nh) # random initialization
            init_W = init_W_wav(Nv, Nh) # wavelet initialization
            # theano shared variales for weights
            W = theano.shared(value=init_W, name='W', borrow=True)
        # initialize input layer for RBM
        if input is None:
            input = T.matrix('input')
        
        self.input = input
        self.W = W
        self.theano_rng = theano_rng
        
    ''' Compute free energy
    F = - sum_j ln(2*cosh(sum_i v_i W_ij))'''
    def free_energy(self, v_samples):
        return -T.sum(T.log(2*T.cosh(T.dot(v_samples, self.W))), axis=1)
    
    ''' Propagate visible spin local fields upwards to hidden spins '''
    def propup(self, v_samples):
        local_fields = T.dot(v_samples, self.W)
        return [local_fields, T.tanh(local_fields)]
    
    ''' Infer hidden spins given visible spins '''
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
        
    ''' Propagate hidden spin local fields downwards to visible spins'''
    def propdown(self, h_samples):
        local_fields = T.dot(h_samples, self.W.T)
        return [local_fields, T.tanh(local_fields)]
    
    ''' Infer visible spins given hidden spins '''
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
    
    ''' One step of Gibbs sampling from hiddens '''
    def gibbs_hvh(self, h0_samples):
        local_fields_v1, v1_means, v1_samples = self.sample_v_given_h(h0_samples)
        local_fields_h1, h1_means, h1_samples = self.sample_h_given_v(v1_samples)
        return [local_fields_v1, v1_means, v1_samples,
                local_fields_h1, h1_means, h1_samples]

    ''' One step of Gibbs sampling from visibles '''
    def gibbs_vhv(self, v0_samples):
        local_fields_h1, h1_means, h1_samples = self.sample_h_given_v(v0_samples)
        local_fields_v1, v1_means, v1_samples = self.sample_v_given_h(h1_samples)
        return [local_fields_h1, h1_means, h1_samples,
                local_fields_v1, v1_means, v1_samples]
    
    ''' Implement one step of CD-k or PCD-k
    learning_rate::float: learning rate used to train the RBM
    persistent::T.matrix: None for CD. For PCD, shared variable
        containing old state of Gibbs chain.
        Must be of size = (batch size, Nh)
    k::int: number of Gibbs steps to equilibrate
    
    Returns a proxy for the cost and the updates dictionary.
    The updates contains rules for weights and persistent chain.
    '''
    def get_cost_updates(self, learning_rate=0.1, persistent=None, k=1):
        # initialize persistent chain
        local_fields_ph, ph_meands, ph_samples = self.sample_h_given_v(self.input)
        if persistent is None: # CD, use newly generated hidden samples
            chain_start = ph_samples
        else: # PCD, use old state of the chain
            chain_start = persistent
        # relax to equilibrium by chain Gibbs sampling
        ([local_fields_nv_chain, nv_means_chain, nv_samples_chain,
          local_fields_nh_chain, nh_means_chain, nh_samples_chain],
         updates) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None,None,None,None,None,chain_start],
            n_steps=k,
            name='gibbs_hvh')
        # take the sample at the end of the chain
        chain_end = nv_samples_chain[-1]
        # cost = free energy (clamped) - free energy (unclamped)
        clamped = T.mean(self.free_energy(self.input))
        unclamped = T.mean(self.free_energy(chain_end))
        cost = clamped - unclamped
        # compute the gradients on weights
        grad_W = T.grad(cost, self.W, consider_constant=[chain_end])
        # add W update rules to updates dict
        updates[self.W] = self.W - grad_W * T.cast(learning_rate,
                                        dtype=theano.config.floatX)
        # determine cross entropy
        if persistent: # PCD
            # update persistent to new hidden samples
            updates[persistent] = nh_samples_chain[-1]
            # pseudo-likelihood is a vetter proxy for PCD
            cross_entropy = self.get_pseudo_likelihood_entropy(updates)
        else:
            # reconstruction cross-entropy is a getter proxy for CD
            cross_entropy = self.get_reconstruction_entropy(updates,
                local_fields_nv_chain[-1])
        return cross_entropy, updates
    
    ''' Pseudo-likelihood cross enetropy '''
    def get_pseudo_likelihood_entropy(self, updates):
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
    
    ''' Reconstruction cross entropy '''
    def get_reconstruction_entropy(self, updates, local_fields_nv):
        nv_probabilities = T.nnet.sigmoid(2*local_fields_nv)
############## self.input is [-1,1] not to be used as probability!
        entropy = -T.mean(
            T.sum(
                self.input*T.log(nv_probabilities)+
                (1-self.input)*T.log(1-nv_probabilities),
                axis = 1)
            )
        return entropy
    
# construct initial weight matrix value
# random initialization
def init_W_rnd(Nv, Nh, numpy_rng):
    rnd = numpy_rng.uniform(
            low = - numpy.sqrt(6./(Nv+Nh)),
            high = numpy.sqrt(6./(Nv+Nh)),
            size = (Nv, Nh))
    return numpy.asarray(rnd, dtype = theano.config.floatX)
# wavelet initialization
def init_W_wav(Nv, Nh):
    k = numpy.arange(0,Nh,1)*numpy.pi
    x = numpy.arange(-Nv/2,Nv/2,1.)+1.
    ks, xs = numpy.meshgrid(k, x)
    ker = numpy.sqrt(6./(Nv+Nh))*numpy.cos(xs*ks)*numpy.exp(-(xs-0.5)**2)
    return numpy.asarray(ker, dtype = theano.config.floatX)    
















