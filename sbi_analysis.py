project_path = '/home/anavasol/PROJECTS/human-CA3-model'
base_name = 'sbi_analysis'
# Import libraries
import numpy as np
import sys
# SBI
import torch
from sbi import utils as utils
from sbi.inference.base import infer
# Our code
sys.path.insert(1, project_path)
import CA3_analytical_model

# ============================
# |  Set prior and simulator
# ============================

execution_date = CA3_analytical_model.print_date()

# ------------
# 1. simulator
# ------------

# Number of total synapses of big
compute_stable_memories_big = lambda params: CA3_analytical_model.compute_stable_memories_sbi(params, 
												s=60e9, x0=1., y0=0., muN=1.)

# Number of total synapses of small
compute_stable_memories_small = lambda params: CA3_analytical_model.compute_stable_memories_sbi(params, 
												s=2e9, x0=1., y0=0., muN=1.)

# ---------
# 2. prior
# ---------

param_names = ['conn_prob', 'g0', 'g1', 'ensemble_size', 'sigmaN' ]
#  			  [  p,  		 g0,   g1, 	 	ens, 		  sigma   ]
prior_min =   [ -3,  		 -8,   -3,   	 2,   			0     ]
prior_max =   [ -1,  		 -6,   -1,   	 3,   			1     ]
prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)

# -------------------
# 3. train inference
# -------------------

num_simulations = 100000
method = 'SNPE'

# big network
posteriors_big = infer(compute_stable_memories_big, prior, 
					method=method, num_simulations=num_simulations, num_workers=8)
CA3_analytical_model.save_posterior(posteriors_big, prior_min, prior_max, 
					f'{project_path}/results/{base_name}_{execution_date}_big.pkl')

# small network
posteriors_small = infer(compute_stable_memories_small, prior, 
					method=method, num_simulations=num_simulations, num_workers=8)
CA3_analytical_model.save_posterior(posteriors_small, prior_min, prior_max, 
					f'{project_path}/results/{base_name}_{execution_date}_small.pkl')
