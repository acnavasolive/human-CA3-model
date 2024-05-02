# Import libraries
import numpy as np
import scipy.stats
import time
from datetime import datetime
import pickle


# ----------------------
# |   Model Functions  |
# ----------------------

def get_N(n_syn, prob):
	'''
		get_N(n_syn, prob) : compute number of neurons based on total number of synapses, and connectivity probability
	'''
	return np.round(np.sqrt(n_syn/prob))

def get_rho(a, m):
	'''
		get_rho(a, m) : compute rho
	'''
	return 1 - (1 - a**2)**m

def get_rhop(a, m, rho, eps=1e-100):
	'''
		get_rhop(a, m, rho, eps=1e-100) : compute rho'
	'''
	return (1/(rho+eps)) * (1 - 2*(1-a**2)**m + (1 - 2*a**2 + a**3)**m)

def get_gamma(a, m):
	'''
		get_gamma(a, m) : compute gamma
	'''
	return (1 - 2*a**2 + a**3)**m - (1 - a**2)**(2*m)

def get_gammap(a, m, rho, rhop, eps=1e-100):
	'''
		get_gammap(a, m, rho, rhop, eps=1e-100) : compute gamma'
	'''
	return (1/(rho+eps)) * (1 - 3*(1-a**2)**m + 3*(1 - 2*a**2 + a**3)**m - (1 - 3*a**2 + 3*a**3 - a**4)**m ) - rhop**2

def Phi(E, s):
	'''
		Phi(E, s) : normal cumulative distribution
	'''
	# If standard deviation is >0, normal cumulative distribution
	if s > 0:
		return scipy.stats.norm.cdf(E/s)
	# If not, step function
	else:
		return np.heaviside(E, 1)

def compute_valid(x, n, a):
	'''
		compute_valid(x, n, a) : compute number of valid firing neurons, spurious firings and recall
	'''
	return np.round(n*a*x)

def compute_spurious(y, n, a):
	'''
		compute_spurious(y, n, a) : compute number of spurious firing neurons
	'''
	return np.round(n*(1-a)*y)

def compute_recall(v, s, n, a):
	'''
		compute_recall(v, s) : compute recall
	'''
	S = v + s
	S1 = v
	if S == 0:
		return 0.0
	elif S == n:
		return 0.0
	elif S1 < a*S:
		return 0.0
	else:
		return (S1 - a*S) / (np.sqrt(S * (1-S/n) ) * np.sqrt(n*a*(1-a)))

def compute_capacity(recalls, ms):
	'''
		compute_capacity(recalls, ms) : compute capacity
	'''

	if recalls.shape[1] == len(ms):
		return np.nanmax(np.tile(ms, (recalls.shape[0],1)) * recalls)
	elif recalls.shape[0] == len(ms):
		return np.nanmax(np.tile(ms, (recalls.shape[1],1)) * recalls)
	else:
		print('recalls does not match ms shape')
		return np.nan

# Iterative fuction to compute dynamics
def next_step(x, y, xp, yp, n, m, a, cBar, c2Bar, g0, g1, muN, sigmaN, eps=1e-100):
	'''
		next_step(x, y, xp, yp, n, m, a, cBar, c2Bar, g0, g1, muN, sigmaN, eps=1e-100) : computes the network's next state 
	'''

	# Expressions for equations
	rho = get_rho(a, m)
	rhop = get_rhop(a, m, rho)
	gamma = get_gamma(a, m)
	gammap = get_gammap(a, m, rho, rhop)

	# Expectations
	E1 = cBar * ( a*x + (1-a)*rho*yp ) * muN - g1 * (a*x + (1-a)*y) - g0 
	En = cBar*rho * ( a*xp + (1-a)*yp ) * muN - g1 * (a*x + (1-a)*y) - g0
	E1p = cBar * ( a*x + (1-a)*rhop*yp ) * muN - g1 * (a*x + (1-a)*y) - g0 
	Enp = cBar*rhop * ( a*xp + (1-a)*yp ) * muN - g1 * (a*x + (1-a)*y) - g0

	# Standard deviations
	# - sigma_1^m
	ns12 = n*a * (cBar-c2Bar)*x + n*(1-a)*rho*yp * (cBar-c2Bar*rho*yp/(y+eps)) + n**2 * (1-a)**2 * cBar**2 * gamma*yp**2 
	in_expression = n * sigmaN**2 * cBar * ( a*x + (1-a)*rho*yp ) + muN**2 * ns12
	s1 = (1/n) * np.sqrt(in_expression) if in_expression>0 else 0
	# - sn
	nsn2 = n*a*rho*xp * (cBar-c2Bar*rho*xp/(x+eps)) + n*(1-a)*rho*yp * (cBar-c2Bar*rho*yp/(y+eps)) + n**2*gamma*(a*xp + (1-a)*yp)**2*cBar**2 
	in_expression = n * sigmaN**2 * cBar*rho * ( a*x + (1-a)*yp ) + muN**2 * nsn2
	sn = (1/n) * np.sqrt(in_expression) if in_expression>0 else 0
	# - s1p
	ns12p = n*a * (cBar-c2Bar)*x + n*(1-a)*rhop*yp * (cBar-c2Bar*rhop*yp/(y+eps)) + n**2 * (1-a)**2 * cBar**2 * gammap*yp**2 
	in_expression = n * sigmaN**2 * cBar * ( a*x + (1-a)*rhop*yp ) + muN**2 * ns12p
	s1p = (1/n) * np.sqrt(in_expression) if in_expression>0 else 0
	# - snp
	nsn2p = n*a*rhop*xp * (cBar-c2Bar*rhop*xp/(x+eps)) + n*(1-a) * rhop*yp * (cBar-c2Bar*rhop*yp/(y+eps)) + n**2*gammap * (a*xp+(1-a)*yp)**2 * cBar**2 
	in_expression = n * sigmaN**2 * cBar*rhop * ( a*x + (1-a)*yp ) + muN**2 * nsn2p
	snp = (1/n) * np.sqrt(in_expression) if in_expression>0 else 0

	# Average firings
	xNext = Phi(E1,s1)
	yNext = Phi(En,sn)
	xpNext = Phi(E1p,s1p)
	ypNext = Phi(Enp,snp)

	# print(f' ({E1:.8f},{s1:.8f}:{xNext:.8f})  ({En:.8f},{sn:.8f}:{yNext:.8f})   {E1p:.8f},{s1p:.8f}:{xpNext:.8f})   {Enp:.8f},{snp:.8f}:{ypNext:.8f})')

	return np.array([xNext, yNext, xpNext, ypNext])


def full_simulation(x0, y0, xp0, yp0, n, m, a, cBar, c2Bar, g0, g1, muN, sigmaN, do_print=True, tmax=1000, tmin=0):
	'''
		full_simulation(x0, y0, xp0, yp0, n, m, a, cBar, c2Bar, g0, g1, muN, sigmaN, do_print=True, tmax=1000, tmin=0) : makes the full simulation
	'''

	# Initialize variables at t=0
	A = [[x0,y0,x0,y0]]
	n_valids = [compute_valid(A[0][0], n, a)]
	n_spurious = [compute_spurious(A[0][1], n, a)]
	recalls = [compute_recall(n_valids[0], n_spurious[0], n, a)]
	# Print output
	if do_print:
		print(f't: {0}\t {n_valids[0]:.0f}\t {n_spurious[0]:.0f}\t {recalls[0]:.3f}  -  {A[-1][0]:.3f} {A[-1][1]:.3f} {A[-1][2]:.3f} {A[-1][3]:.3f}')
	# A of next step
	t = 0
	while (any( np.abs(next_step(*np.array(A[t]), n, m, a, cBar, c2Bar, g0, g1, muN, sigmaN)[:2]-np.array(A[t])[:2]) > 1e-3 ) and (t<tmax)) or (t<tmin):
		A.append( next_step(*A[t], n, m, a, cBar, c2Bar, g0, g1, muN, sigmaN) )
		n_valids.append( compute_valid(A[t+1][0], n, a) )
		n_spurious.append( compute_spurious(A[t+1][1], n, a) )
		recalls.append( compute_recall(n_valids[t+1], n_spurious[t+1], n, a) )
		if do_print:
			print(f't: {t+1}\t {n_valids[t+1]:.0f}\t {n_spurious[t+1]:.0f}\t {recalls[t+1]:.8f}  -  {A[-1][0]:.8f} {A[-1][1]:.8f} {A[-1][2]:.8f} {A[-1][3]:.8f}')
		t += 1

	return np.array(recalls), np.array(n_valids), np.array(n_spurious), np.array(A)


def compute_stable_memories(p, s, g0, g1, ensemble, muN, sigmaN, x0, y0, min_recall=0.6, tmin=1, tmax=10, mmin=0, mmax=1e9, merr=10, do_print=False):
	'''
		compute_stable_memories(p, s, g0, g1, ensemble, muN, sigmaN, x0, y0, min_recall=0.6, tmin=1, tmax=10, mmin=0, mmax=1e9, merr=10, do_print=False) : function that runs the network model
	'''

	# Transform to model variables
	# - mean connectivity probability
	cBar = np.copy(p)
	c2Bar = cBar**2 
	# - percentage of valid neurons
	n = get_N(s, p)
	a = ensemble / n
	# - xp0, yp0
	xp0 = x0
	yp0 = y0

	# Initialize maximum number of stable memories
	max_num_stable_memories = 0

	# Test upper and lower limits
	ms = np.array([[mmin, -1], [mmax, -1]])
	for im in range(2):
		m = ms[im,0]
		# Get recall
		rr, _, _, _ = full_simulation(x0, y0, xp0, yp0, n, m, a, cBar, c2Bar, g0, g1, muN, sigmaN, tmin=tmin, tmax=tmax, do_print=do_print)
		ms[im,1] = rr[-1] > min_recall
	
	# Iteratively test intermediate points
	if any(ms[:,1]==1):
		# Number of iterations to find maximum capacity
		n_iter = np.ceil(np.log2((mmax-mmin)/merr)).astype(int)
		for im in range(n_iter):
			# Middle point
			m = np.round(np.mean(ms[:,0]))
			# Get recall
			rr, _, _, _ = full_simulation(x0, y0, xp0, yp0, n, m, a, cBar, c2Bar, g0, g1, muN, sigmaN, tmin=tmin, tmax=tmax, do_print=do_print)
			# If it achieved recall, substitute m in ms where there is a 1
			# If it didnt, substitute m where there is a 0
			if rr[-1] > min_recall:
				ms[ms[:,1]==1,0] = m
				ms[ms[:,1]==1,1] = 1
			else:
				ms[ms[:,1]==0,0] = m
				ms[ms[:,1]==0,1] = 0
			max_num_stable_memories = ms[ms[:,1]==1,0][0]

	return max_num_stable_memories

def compute_stable_memories_sbi(params, s, x0, y0, muN, min_recall=0.6, tmin=1, tmax=10, mmin=0, mmax=1e9, merr=10, do_print=False):
	'''
		compute_stable_memories_sbi(params, s, x0, y0, muN, min_recall=0.6, tmin=1, tmax=10, mmin=0, mmax=1e9, merr=10, do_print=False) : function that runs the network model for SBI training
	'''

	import torch

	# Extract parameters
	p_var, g0_var, g1_var, ensemble_size, sigmaN = np.asarray(params)
	p = 10. ** p_var
	g0 = 10. ** g0_var
	g1 = 10. ** g1_var
	ensemble = 10 ** ensemble_size

	# Transform to model variables
	# - mean connectivity probability
	cBar = np.copy(p)
	c2Bar = cBar**2 
	# - percentage of valid neurons
	n = get_N(s, p)
	a = ensemble / n
	# - xp0, yp0
	xp0 = x0
	yp0 = y0

	# Initialize maximum number of stable memories
	max_num_stable_memories = 0

	# Test upper and lower limits
	ms = np.array([[mmin, -1], [mmax, -1]])
	for im in range(2):
		m = ms[im,0]
		# Get recall
		rr, _, _, _ = full_simulation(x0, y0, xp0, yp0, n, m, a, cBar, c2Bar, g0, g1, muN, sigmaN, tmin=tmin, tmax=tmax, do_print=do_print)
		ms[im,1] = rr[-1] > min_recall
	
	# Iteratively test intermediate points
	if any(ms[:,1]==1):
		# Number of iterations to find maximum capacity
		n_iter = np.ceil(np.log2((mmax-mmin)/merr)).astype(int)
		for im in range(n_iter):
			# Middle point
			m = np.round(np.mean(ms[:,0]))
			# Get recall
			rr, _, _, _ = full_simulation(x0, y0, xp0, yp0, n, m, a, cBar, c2Bar, g0, g1, muN, sigmaN, tmin=tmin, tmax=tmax, do_print=do_print)
			# If it achieved recall, substitute m in ms where there is a 1
			# If it didnt, substitute m where there is a 0
			if rr[-1] > min_recall:
				ms[ms[:,1]==1,0] = m
				ms[ms[:,1]==1,1] = 1
			else:
				ms[ms[:,1]==0,0] = m
				ms[ms[:,1]==0,1] = 0
			max_num_stable_memories = ms[ms[:,1]==1,0][0]

	return torch.as_tensor([max_num_stable_memories])

# ----------------------
# |   Auxiliar		   |
# ----------------------

def print_duration(start_time):
	return f'{np.floor((time.time()-start_time)/60):.0f}:{np.mod((time.time()-start_time),60):.0f}'

def print_date():
	return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def round_sig(x, sig):
	return np.array([ np.round(xi, sig-int(np.floor(np.log10(np.abs(xi))))-1) for xi in x ])

def save_posterior(posterior, prior_min, prior_max, save_name):
	results = {'posterior':posterior, 
			'prior_min':prior_min, 
			'prior_max':prior_max}
	with open(save_name, 'wb') as f:
		pickle.dump(results, f)

def load_posterior(save_name):
	with open(save_name, 'rb') as f:
		results = pickle.load(f)
	posterior = results['posterior']
	prior_min = results['prior_min']
	prior_max = results['prior_max']
	return posterior, prior_min, prior_max