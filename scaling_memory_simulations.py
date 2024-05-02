project_path = '/home/anavasol/PROJECTS/human-CA3-model'
base_name = 'scaling_memory'
# Import libraries
import numpy as np
import time, pickle, sys
# Our code
sys.path.insert(1, project_path)
import CA3_analytical_model

# Testing parameters
S = np.array([2, 4, 8, 16, 32, 60])*1e9 									# Number of total synapses
P = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 10])/100. 							# Number of probabilities to test
N = np.round([[CA3_analytical_model.get_N(s, p) for p in P] for s in S]) 	# Number of cells to maintain #syns constant
g1s = 10**np.linspace(-6,-0.5, 51) 											# Amount of inhibition

# Constant parameters
x0, y0, xp0, yp0 = 1., 0., 1., 0.   	# Initial state
g0 = 1e-7  								# Threshold
ensemble = 330 							# Neurons
muN = 1.0 								# Quantal amplitude
sigmaN = 0.0							# Quantal variability

# Initialize recalls
M = np.zeros((len(S), len(P), len(g1s)))

# Record start time
start_time = time.time()
execution_date = CA3_analytical_model.print_date()
print(f'Executing test {execution_date}...')
# Do simulations
for iis, s in enumerate(S):
	for ii, p in enumerate(P):
		for iig, g1 in enumerate(g1s):
			# Probability and number of neurons
			print(f'#syns = {s/1e9:.0f}B ({iis+1}/{len(S):.0f}) \t (n,p) : ({ii+1}/{len(P)})...  {CA3_analytical_model.print_duration(start_time)}', end='\r')
			# Make simulation
			M[iis, ii, iig] = CA3_analytical_model.compute_stable_memories(p, s, g0, g1, ensemble, muN, sigmaN, x0, y0, do_print=False)

# Time
print(f'\nExecution time: {time.time()-start_time:.2f}s ({(time.time()-start_time)/60.:.2f} min)')
# Save
save_dict = {'g0':g0, 'ensemble':ensemble, 'x0':x0, 'y0':y0, 'muN':muN, 'sigmaN':sigmaN, 
			'S':S, 'P':P, 'N':N, 'g1s':g1s, 'M':M}

with open(f'{project_path}/results/{base_name}_{execution_date}_M.pkl', 'wb') as f:
	pickle.dump(save_dict, f)
