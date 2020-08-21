from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import matplotlib.pyplot as plt
import matplotlib
import GPFourier as gpf 
import random as rand


import scipy as sp
import autograd.numpy as np
from autograd.scipy.special import gammaln
from scipy.optimize import minimize



from autograd import grad
from autograd import value_and_grad
from autograd.misc.optimizers import adam, sgd
from scipy.optimize import minimize


matplotlib.interactive(True)

def sigmoid(x):
  ''' sigmoid operation ''' 
  return 1/ (1 + np.exp(-x))

def make_cov(N, len_sc):
  ''' make RBF covariance matrix '''
  M1 = np.array([range(N)])- np.transpose(np.array([range(N)]))
  K = 1*np.exp(-(np.square(M1)/(2*np.square(len_sc))))
  return K


def compute_chebyshev(f,xlim,power=2,dx=0.01):
    """
    This function computes the coefficients of the optimal polynomial approximation
    of a function over a specified interval using Chebyshev polynomials. The coefficents
    are computed using a weighted least squares solution.
    Input:
        f - function
        xlim - length 2 numpy array with approximation interval [x_0,x_1]
        power - order of polynomial approximation
        dx - grid spacing for least square
    Output:
        what_cheby - length p numpy array containing the coefficients of the polynomial starting at order 0
    """

    # range to approximate
    xx = np.arange(xlim[0]+dx/2.0,xlim[1],dx)
    nx = xx.shape[0]

    # relative locations in [-1,1]
    xxw = np.arange(-1.0+1.0/nx,1.0,1.0/(0.5*nx))

    # create polynomial basis
    Bx = np.zeros([nx,power+1])
    for i in range(0,power+1):
        Bx[:,i] = np.power(xx,i)

    # compute weighting relative to locations in [-1,1]
    errwts_cheby = 1.0 / np.sqrt(1-xxw**2)
    Dx = np.diag(errwts_cheby)

    # compute weighted least squares solution minimizes difference between f and approximation of f
    fx = f(xx)
    what_cheby = np.linalg.lstsq(Bx.T @ Dx @ Bx,Bx.T @ Dx @ fx, rcond = None)[0]
    fhat_cheby = Bx @ what_cheby

    # return weights
    return what_cheby

def build_toy_dataset_FA(N, D,loadings, len_sc,  poiss_noise = False, Gauss_noise= False, neg_bino_noise = False, bino_noise = False, scale = 1):
	'''
	This function will generate some data with GP statistics accordings to length scale 'len_sc'. 
	Data will be generated of length N, with batch size D. 

	'''

  # make covariance
	M1 = np.array([range(N)])- np.transpose(np.array([range(N)]))
	if np.size(len_sc)>0:
		K = [np.exp(-(np.square(M1)/(2*np.square(len_sc[i])))) for i in np.arange(np.size(len_sc))]
	else:
		K = np.exp(-(np.square(M1)/(2*np.square(len_sc))))

	n_latents, n_neurons = loadings.shape[0], loadings.shape[1] 



	#draw a rate with GP stats (one or many)
	if np.size(len_sc)>0:
		latents = np.array([np.random.multivariate_normal(np.zeros(N), K[i]) for i in np.arange(np.size(len_sc))])
	else:
		latents = np.array(np.random.multivariate_normal(np.zeros(N), K, n_latents))



	#pass through the loadings matrix
	y_sm = loadings.T@latents

  # add whatever noise you like. 
	if Gauss_noise is True:  
		raise Exception("FIX THIS FIRST!")
		y = latents + [np.random.normal(0.0, np.sqrt(scale), size=N) for batch in range(D)]

	elif poiss_noise is True:
		x= np.array([np.exp(y_sm) for batch in range(D)]/np.asarray(scale))
		#x= np.array([np.log(1 + np.exp(y_sm)) for batch in range(D)]/np.asarray(scale_pois))
		#x = [np.exp(x)/10 for batch in range(D)]
		y = np.random.poisson(x)
		x = x[0]

	elif neg_bino_noise is True:
		# Negative Binomial Noise added with param scale
		x= np.exp(y_sm)
		p = scale/(x+scale)
		y = np.array([np.random.negative_binomial(scale, p) for i in np.arange(D)])

	elif bino_noise is True:
		# Negative Binomial Noise added with param scale
		x= sigmoid(y_sm)
		y = np.array([np.random.binomial (scale, x) for i in np.arange(D)])
		x = x
	else:
		raise Exception("PLEASE SET SOME NOISE EQUAL TO TRUE")

	return x, y, latents


def unpack_params(params):
	return params[:n_latents], np.reshape(params[n_latents:], [n_latents, n_neurons])

def block_diag(Cinv, n_latents):
	'''
	supported only for equal sized matrices

  supported currently up to latent dimension size of 3, can do more

  Though these lines are not good coding practice, autograd does not
  support inserting into a pre-allocated matrix. These matrices are thus
  all built from scratch.
	'''
	zeroarray = np.zeros(np.shape(Cinv[0]))
	if n_latents is 2:
	  arraytop = np.concatenate([Cinv[0],zeroarray])
	  arraybottom = np.concatenate([zeroarray, Cinv[1]])
	  array = np.hstack([arraytop,arraybottom])
	if n_latents is 3:
	  arraytop = np.concatenate([Cinv[0],zeroarray, zeroarray])
	  arraymiddle = np.concatenate([zeroarray, Cinv[1], zeroarray])
	  arraybottom = np.concatenate([zeroarray, zeroarray, Cinv[2]])
	  array = np.hstack([arraytop,arraymiddle,arraybottom])

	return array


def gen_marg_negbino(y_train, params,n_latents,n_neurons, coeffs, a1y, a0y, a1, summedy, N, consts, scale):

  len_sc, loadings = unpack_params(params)
  D = np.shape(y_train)[0] #number of trials



  a2W = coeffs.T[2]*loadings
  a2W = (np.kron(a2W, np.eye(N)).T)
  W = (np.kron(loadings, np.eye(N)).T)


  #cdiag = gpf.mkcovs.mkcovdiag_ASD_wellcond(init_len_sc+0, np.ones(np.size(init_len_sc)), nxcirc, wwnrm = wwnrm,addition = 1e-7).T 
  C = [make_cov(N, i) + 1e-8*np.eye(N) for i in len_sc]
  #all_cdiag = np.reshape(cdiag.T,np.size(init_len_sc)*N_four,-1)
  Cinv = [np.linalg.inv(C[i]) for i in np.arange(n_latents)]

  if n_latents is 1:
    sigma_inv = 2*D*scale*a2W.T@W +  2*a2W.T@np.multiply(summedy, np.eye(N*n_neurons))@W  + Cinv[0]

  else:
    sigma_inv =2*D*scale*a2W.T@W +  2*a2W.T@np.multiply(summedy, np.eye(N*n_neurons))@W  +block_diag(Cinv, n_latents)

  #sigma = np.linalg.inv(sigma_inv)

  second = W.T@(np.squeeze(summedy)- np.squeeze(D*a1.T*scale) - np.squeeze(a1y))
  mutot = np.squeeze(np.linalg.solve(sigma_inv,second))


  logl = np.squeeze((1/2)*mutot.T@(sigma_inv)@mutot)

  logdetC = 0
  for i in np.arange(n_latents):
    logdetC = logdetC -(1/2)*np.linalg.slogdet(C[i])[1]

  neglogpost = logdetC + -(1/2)*np.linalg.slogdet(sigma_inv)[1]

  return -(neglogpost+logl+consts)

def gen_marg_bino(y_train, params,n_latents,n_neurons, coeffs, N, D, count):

  len_sc, W = unpack_params(params)


  summedy = np.sum(y_train, axis = 0)
  summedy = np.reshape(summedy, [np.size(summedy),-1]) 
  a1 = np.array([np.repeat(coeffs.T[1],N)])
  countvec = np.array([np.repeat(count,N)])

  na2W = (count*coeffs.T[2])*W
  quad = 2*na2W@W.T

  #a2W = (np.kron(a2W, np.eye(N)).T)
  #W = (np.kron(loadings, np.eye(N)).T)


  #cdiag = gpf.mkcovs.mkcovdiag_ASD_wellcond(init_len_sc+0, np.ones(np.size(init_len_sc)), nxcirc, wwnrm = wwnrm,addition = 1e-7).T 
  C = [make_cov(N, i) + 1e-7*np.eye(N) for i in len_sc]

  #all_cdiag = np.reshape(cdiag.T,np.size(init_len_sc)*N_four,-1)
  Cinv = [np.linalg.inv(C[i]) for i in np.arange(n_latents)]

  if n_latents is 1:
    sigma_inv = Cinv[0]
  else:
    sigma_inv = np.kron(quad, np.eye(N)) + block_diag(Cinv,n_latents)


  #sigma = np.linalg.inv(sigma_inv)
  Wkron = np.kron(W, np.eye(N))

  second =Wkron@(summedy - countvec.T - (countvec*a1).T)
  mutot = np.squeeze(np.linalg.solve(sigma_inv,second))

  ###########################
  # len_sc, loadings = unpack_params(params)


  # summedy = np.sum(y_train, axis = 0)
  # summedy = np.reshape(summedy, [np.size(summedy),-1]) 
  # a1 = np.array([np.repeat(coeffs.T[1],N)])
  # countvec = np.array([np.repeat(count,N)])

  # a2W = (count*coeffs.T[2])*loadings
  # a2W = (np.kron(a2W, np.eye(N)).T)
  # W = (np.kron(loadings, np.eye(N)).T)


  # #cdiag = gpf.mkcovs.mkcovdiag_ASD_wellcond(init_len_sc+0, np.ones(np.size(init_len_sc)), nxcirc, wwnrm = wwnrm,addition = 1e-7).T 
  # C = [make_cov(N, i) + 1e-7*np.eye(N) for i in len_sc]
  # #all_cdiag = np.reshape(cdiag.T,np.size(init_len_sc)*N_four,-1)
  # Cinv = [np.linalg.inv(C[i]) for i in np.arange(n_latents)]

  # if n_latents is 1:
  #   sigma_inv = 2*a2W.T@W + Cinv[0]
  # else:
  #   sigma_inv = 2*a2W.T@W + block_diag(Cinv, n_latents)

  # sigma = np.linalg.inv(sigma_inv)
  # mutot = np.squeeze(sigma@W.T@(summedy - countvec.T - (countvec*a1).T))
  ##############################


  logl = np.squeeze((1/2)*mutot.T@(sigma_inv)@mutot)

  logdetC = 0
  for i in np.arange(n_latents):
    logdetC = logdetC -(1/2)*np.linalg.slogdet(C[i])[1]

  neglogpost = logdetC + -(1/2)*np.linalg.slogdet(sigma_inv)[1]


  return -(neglogpost+logl)

def gen_marg_poiss(y_train, params,n_latents,n_neurons, coeffs,a1, N, D):

  len_sc, loadings = unpack_params(params)
  #D = np.shape(y_train)[0]

  # summedy = np.sum(y_train, axis = 0)
  # summedy = np.reshape(summedy, [np.size(summedy),-1]) #THIS PART DOESN"T WORK FOR LOTS OF NEURONS

  

  a2W = coeffs.T[2]*loadings
  a2W = (np.kron(a2W, np.eye(N)).T)
  W = (np.kron(loadings, np.eye(N)).T)


  #cdiag = gpf.mkcovs.mkcovdiag_ASD_wellcond(init_len_sc+0, np.ones(np.size(init_len_sc)), nxcirc, wwnrm = wwnrm,addition = 1e-7).T 
  C = [make_cov(N, i) + 1e-7*np.eye(N) for i in len_sc]
  #all_cdiag = np.reshape(cdiag.T,np.size(init_len_sc)*N_four,-1)
  Cinv = [np.linalg.inv(C[i]) for i in np.arange(n_latents)]

  if n_latents is 1:
    sigma_inv = 2*D*a2W.T@W + Cinv[0]
  else:
    sigma_inv = 2*D*a2W.T@W + block_diag(Cinv, n_latents)




  second = W.T@(summedy- D*a1.T)
  mutot = np.squeeze(np.linalg.solve(sigma_inv,second))
  #sigma = np.linalg.inv(sigma_inv)
  #mutot = np.squeeze(sigma@W.T@(summedy- D*a1.T))

  logl = np.squeeze((1/2)*mutot.T@(sigma_inv)@mutot)


  logdetC = 0
  for i in np.arange(n_latents):
    logdetC = logdetC -(1/2)*np.linalg.slogdet(C[i])[1]

  neglogpost = logdetC + -(1/2)*np.linalg.slogdet(sigma_inv)[1]


  return -(neglogpost+logl)

def prep_opt(y_train, N, coeffs):
  summedy_mat = np.sum(y_train, axis = 0)  
  summedy = np.reshape(summedy_mat, [np.size(summedy_mat),-1]) 

  a1 = np.reshape([np.repeat(coeffs.T[1],N)], [np.size(summedy),-1])
  a0 = np.reshape([np.repeat(coeffs.T[0],N)], [np.size(summedy),-1])
  a1y = np.multiply(a1, summedy)
  a0y = np.multiply(a0, summedy)

  consts = np.sum(gammaln(y_train + scale)) - D*n_neurons*N*gammaln(scale) - np.sum(coeffs.T[0]*(D*scale*N)) -np.sum(a0y) - np.sum(summedy*np.log(scale))


  return summedy, a1y, a0y, a1, consts

#######data generation #####

##params##
N = 200  # number of data points
D = 30# number of trials (batch size)
n_neurons = 40
n_latents = 2
scale = 1 #use 3 for bino, otherwise, just use 1.  This will make sure overall rates are approx the same
loadings = (np.random.rand(n_latents,n_neurons)) #sometimes this distribution matters for recoverability, keeping random numbers between 0 and 1 is a good option
len_sc = [60,15]#GP length scale (number of latents in length)

if np.size(len_sc) != n_latents:
  raise ValueError("the len_sc vector must be the same number of elements as n_latents")


#generate fake data
x_train, y_train, latents = build_toy_dataset_FA(N,D, loadings, len_sc, bino_noise = True,scale=scale)



#calculate mean rate
average_rates = np.mean(y_train, axis = (0,2))###could be median, but might get 0...which is bad... p[topm pffset os [psson;e]]







###initialize optimization params#
init_loadings = np.random.rand(n_latents*n_neurons)
#init_loadings = np.zeros(n_latents*n_neurons)+.01
init_len_sc = 100*np.random.rand(n_latents)+1
inits =np.concatenate([init_len_sc, init_loadings])

#bound these params with wide bounds.
bounds = [(-10,10) for i in np.arange(n_neurons*n_latents)]
len_bounds = [(10,100) for i in np.arange(n_latents)]


#loop over scale param (for negbino)
scrange = np.arange(1,1.1,3)
marg_over_scale = np.zeros(len(scrange))


print("starting PAL marginal optimization for randomly generated data....\n")
print("number of neurons", n_neurons,"\nnumber of latents", n_latents)
tee = time.time()

##outer loop (for possible scale est in neg bino)
# k = 0
# for i in scrange: #sweep over scale for negative binomial

# ## negbino
  # nonlin_center = [[np.log(i*(np.exp(j)-1))-2, np.log(i*(np.exp(j)-1))+2] for j in average_rates]##negbino
  # f = lambda x: np.log(1+(1/i)*np.exp(x))
  # coeffs = np.array([compute_chebyshev(f,i,power=2,dx=0.01) for i in nonlin_center])
  # summedy, a1y, a0y, a1, consts= prep_opt(y_train, N, coeffs)
  # objective  =  lambda params : gen_marg_negbino(y_train, params,n_latents,n_neurons, coeffs, a1y, a0y, a1, summedy, N, consts, scale = i)



##bino

nonlin_center = [[-np.log(((1/average_rates[j])-1))-4, -np.log(((1/average_rates[j])-1))+4]for j in np.arange(n_neurons)]##negbino
f = lambda x: np.log(1+np.exp(-x))
coeffs = np.array([compute_chebyshev(f,i,power=2,dx=0.01) for i in nonlin_center])
#coeffs[:,0] = np.zeros([1,20])
#coeffs[:,1] = np.ones([1,20])
#coeffs[:,2] = np.zeros([1,20])

y_train_bino = np.sum(y_train, axis = 0)
y_train_bino = np.reshape(y_train_bino, [1, n_neurons, N])
maxD = np.max(np.max(y_train_bino, axis = 2))
maxD = maxD*np.ones(n_neurons)

objective  =  lambda params : gen_marg_bino(y_train_bino, params,n_latents,n_neurons, coeffs,  N, 1, maxD)


## poiss
# f = np.exp
# nonlin_center = [[np.log(i)-2, np.log(i)+2 ] for i in average_rates]
# coeffs = np.array([compute_chebyshev(f,i,power=2,dx=0.01) for i in nonlin_center])
# summedy = np.sum(y_train, axis = 0)
# summedy = np.reshape(summedy, [np.size(summedy),-1]) #THIS PART DOESN"T WORK FOR LOTS OF NEURONS
# a1 = np.array([np.repeat(coeffs.T[1],N)])
# objective  =  lambda params : gen_marg_poiss(summedy, params, n_latents, n_neurons, coeffs,a1, N,D)

# ##maginal optimization
h_bar = minimize(value_and_grad(objective), inits, jac=True, method='L-BFGS-B', bounds = len_bounds+bounds)

#marg_over_scale[0] = h_bar.fun








marg_opt_time = time.time()-tee
print("PAL marg op time is", round(marg_opt_time, 4), "seconds")











































##### MAP ESTIMATE FUNCTIONS ##########

def conv_time_inf(x_samples,loadings,N, Bf):
  '''
  Takes in the latents and restructures things, converts to time domain, returns rates to be used
  for the log-joint calculations 

  returned size is samples x latents x N (time domain N)
  '''

  numsamps = 1
  samplength = np.size(x_samples)    #x_samples = np.reshape(x_samples, [numsamps,samplength])

  if len(np.shape(loadings)) > 0:#if loadings matrix
    n_latents, n_neurons = loadings.shape[0], loadings.shape[1]
    x_samps = np.reshape(x_samples,[numsamps,n_latents,-1]) 
    time_x = np.matmul(x_samps,Bf[None,:])#convert to time domain (dimension samples x latents x N (time domain))
    rates = loadings.T@time_x #samples by n_neurons by N

  else:
    x_samps = np.reshape(x_samples,[numsamps,n_latents,-1]) 
    time_x = np.matmul(x_samps,Bf[None,:])#convert to time domain (dimension samples x latents x N (time domain))
    rates = time_x

  return rates, x_samps, samplength

def calc_log_prior(x_samps, n_latents, samplength, cdiag):
  '''
  Calculates the log prior in the fourier domain. Can be used for all approaches to fourier GP (gauss, poiss, binom, negbinom)
  '''
  numsamps =  np.size(x_samps,0)
  total_prior = 0
  for i in np.arange(n_latents):
    #set 
    x_samp = np.reshape(x_samps[:,i,:],[numsamps,int(samplength/n_latents)])
    #print(cdiag[i*int(samplength/n_latents):i*int(samplength/n_latents)+int(samplength/n_latents)].shape)
    cdiaglat = cdiag[i*int(samplength/n_latents):i*int(samplength/n_latents)+int(samplength/n_latents)]
    logprior = -(1/2)*(np.sum(np.square(x_samp)/cdiaglat,axis=1)+ np.sum(np.log(2*np.pi*cdiaglat)))  
    total_prior = logprior + total_prior

  return total_prior

def log_prob_negbino(x_samples, t,loadings, len_sc, y_train, D, N, sigmasq_hat, Bf, cdiag, calc_prior = True):



  rates,x_samps, samplength = conv_time_inf(x_samples,loadings,N, Bf)

  pos_rates =np.exp(rates)


  firstterm = np.sum(gammaln(sigmasq_hat + y_train)-(gammaln(sigmasq_hat))) #sum over trials and time and neurons

  alpha = 1/sigmasq_hat
  
  secondterm = -D*np.sum(sigmasq_hat*np.log((1+alpha*pos_rates)), axis = (1,2))
  thirdterm = np.sum(y_train[None,:]*(np.log(alpha*pos_rates) - np.log(1+alpha*pos_rates))[:,None,:], axis = (1,2,3))


  loglike = firstterm +secondterm +thirdterm
  if calc_prior:
    total_prior = calc_log_prior(x_samps, n_latents, samplength, cdiag)
    logposts = loglike + total_prior

    return logposts
  else:
    return loglike

def log_prob_poiss(x_samples, t,loadings, y_train, D, N, Bf, cdiag,calc_prior = True):


  rates, x_samps, samplength = conv_time_inf(x_samples,loadings,N, Bf)
  axes_final = (1,2)
  axes_first = (1,2,3)

  loglike = -np.sum(np.log(sp.misc.factorial(y_train)))+ np.sum(y_train[None,:]*(rates[:,None,:]),axis=axes_first) - np.sum(D*np.exp(rates), axis = axes_final)

  if calc_prior:
    total_prior = calc_log_prior(x_samps, n_latents, samplength, cdiag)

    logposts = loglike + total_prior
    return logposts
  else:
    return loglike 

def log_prob_bino(x_samples, t,loadings, y_train, maxD, N, Bf, cdiag, calc_prior = True):


  rates,x_samps, samplength = conv_time_inf(x_samples,loadings,N, Bf)
  
  summedy_mat = np.sum(y_train, axis = 0)   
  

  rates = sigmoid(rates) #nonlinearity
  firstterm = (maxD[:,None] - summedy_mat)*np.log(1-rates) 
  secondterm = summedy_mat *np.log(rates) 
  loglike = np.sum(secondterm + firstterm)


  if calc_prior:
    total_prior = calc_log_prior(x_samps, n_latents, samplength, cdiag)

    logposts = loglike + total_prior
    return logposts
  else:
    return loglike 




# ###### MAP ESTIMATE (AUTOGRAD)  #####

#convert to Fourier
minlens =10 #assert a minimum scale for eigenvalue thresholding
condthresh = 1e8
nxc_ext = 0.2
[By, wwnrm, Bffts, nxcirc] = gpf.comp_fourier.conv_fourier_mult_neuron(y_train, N, minlens,n_neurons,nxcirc = np.array([N+nxc_ext*N]),condthresh = condthresh)
N_four = Bffts[0].shape[0]


#### RANDOM (OR ZERO)######
#loadings_map = 2*np.random.rand(n_latents,n_neurons)/n_latents
#lensc_map =100* np.random.rand(n_latents)


#### MAP#####
lensc_map = h_bar.x[:n_latents]
loadings_map = h_bar.x[n_latents:]


loadings_map = np.reshape(loadings_map, [n_latents, n_neurons])
t = 1
init_params = np.zeros([N_four*n_latents])



cdiag = gpf.mkcovs.mkcovdiag_ASD_wellcond(lensc_map, 1, nxcirc, wwnrm = wwnrm,addition = 1e-7).T
cdiag = np.reshape(cdiag.T,N_four*n_latents,-1)

y_train_bino = np.reshape(y_train_bino, [1, n_neurons, N])
maxD = np.max(np.max(y_train_bino, axis = 2))
maxD = maxD*np.ones(n_neurons)

map_opt = lambda samples : -log_prob_bino(samples, t, loadings_map, y_train, maxD,  N, Bffts[0],cdiag)

minst = minimize(value_and_grad(map_opt), init_params, jac=True, method='L-BFGS-B',  options={'maxiter': 4000})

mapest = np.reshape(minst['x'], [n_latents, N_four])

latents_hat = mapest@Bffts[0]


data_recon = loadings_map.T@latents_hat







### data plotting ###
row= 2
column = 4 ### cycle through neurons to plot


for i in np.arange(row*column):

  plt.subplot(row,column,i+1)
  bx_w = 20 #boxcar width
  hist = sum(y_train[:,i,:]/D)
  hist2 = np.convolve(hist,(np.ones(bx_w)/bx_w),'same')
  plt.bar(np.arange(N),hist,color = 'k', alpha =.2)


  #plt.plot(np.exp(data_recon[i]),'b') ### for poisson and negative binomial
  #plt.plot(x_train[i],'g', alpha = .5)


  plt.plot(maxD[i]*sigmoid(data_recon[i])/D, 'r') #### for Binomial
  plt.plot(scale*x_train[i],'g', alpha = .5) #### for Binomial


  #plt.plot(np.arange(N),hist2,color = 'r', alpha =.5)
  plt.ylabel('rate (spk/bin)')

plt.legend(['model estimate', 'true rate'])
plt.show()

