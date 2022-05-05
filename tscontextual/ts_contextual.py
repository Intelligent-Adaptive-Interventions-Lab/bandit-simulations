import pandas as pd
import numpy as np
from scipy.stats import invgamma
from typing import Dict
from tscontextual.types import TSContextualParams
from datasets.contexts import ContextAllocateData

# Draw thompson sample of (reg. coeff., variance) and also select the optimal action
def thompson_sampling_contextual(params: TSContextualParams, contexts: Dict[str, ContextAllocateData]):
	'''
	thompson sampling policy with contextual information.
	Outcome is estimated using bayesian linear regression implemented by NIG conjugate priors.
	map dict to version
	get the current user's context as a dict
	'''
	# Store normal-inverse-gamma parameters
	parameters = params.parameters

	# Store regression equation string
	regression_formula = parameters['regression_formula']
	# Action space, assumed to be a json
	action_space = parameters['action_space']

	# Include intercept can be true or false
	include_intercept = parameters['include_intercept']

	# Store contextual variables
	contextual_vars = parameters['contextual_variables']

	contextual_vars_dict = {}
	for var in contextual_vars: 
        contextual_vars_dict[var] = np.random.choice(contexts[var].values, size=1, p=contexts[var].allocations)

	# Get current priors parameters (normal-inverse-gamma)
	mean = parameters['coef_mean']
	cov = parameters['coef_cov']
	variance_a = parameters['variance_a']
	variance_b = parameters['variance_b']

	# Draw variance of errors
	precesion_draw = invgamma.rvs(variance_a, 0, variance_b, size=1)
	# Draw regression coefficients according to priors
	coef_draw = np.random.multivariate_normal(mean, precesion_draw * cov)

	## Generate all possible action combinations
	# Initialize action set
	all_possible_actions = [{}]

	# Itterate over actions label names
	for cur in action_space:
		# Store set values corresponding to action labels
		cur_options = action_space[cur]

	    # Initialize list of feasible actions
		new_possible = []
	    # Itterate over action set
		for a in all_possible_actions:
		# Itterate over value sets correspdong to action labels
			for cur_a in cur_options:
				new_a = a.copy()
				new_a[cur] = cur_a

		        # Check if action assignment is feasible
				if is_valid_action(new_a):
			        # Append feasible action to list
					new_possible.append(new_a)
					all_possible_actions = new_possible

	# Print entire action set
	print('all possible actions: ' + str(all_possible_actions))

	## Calculate outcome for each action and find the best action
	best_outcome = -np.inf
	best_action = None

	print('regression formula: ' + regression_formula)
	# Itterate of all feasible actions
	for action in all_possible_actions:
		independent_vars = action.copy()
		independent_vars.update(contextual_vars_dict)

		# Compute expected reward given action
		outcome = calculate_outcome(independent_vars,coef_draw, include_intercept, regression_formula)

		# Keep track of optimal (action, outcome)
		if best_action is None or outcome > best_outcome:
			best_outcome = outcome
			best_action = action

	# Print optimal action
	print('best action: ' + str(best_action))

	return best_action


# Check whether action is feasible (only one level of the action variables can be realized)
def is_valid_action(action):
	'''
	checks whether an action is valid, meaning, no more than one vars under same category are assigned 1
	'''

	# Obtain labels for each action
	keys = action.keys()

	# Itterate over each action label
	for cur_key in keys:

		# Find the action labels with multiple levels
		if '_' not in cur_key:
			continue
		value = 0
		prefix = cur_key.rsplit('_', 1)[0] + '_'

		# Compute sum of action variable with multiple levels
		for key in keys:
			if key.startswith(prefix):
				value += action[key]

		# Action not feasible if sum of indicators is more than 1
		if value > 1:
			return False

	# Return true if action is valid
	return True

# Compute expected reward given context and action of user
# Inputs: (design matrix row as dict, coeff. vector, intercept, reg. eqn.)
def calculate_outcome(var_dict, coef_list, include_intercept, formula):
	'''
	:param var_dict: dict of all vars (actions + contextual) to their values
	:param coef_list: coefficients for each term in regression
	:param include_intercept: whether intercept is included
	:param formula: regression formula
	:return: outcome given formula, coefficients and variables values
	'''
	# Strip blank beginning and end space from equation
	formula = formula.strip()

	# Split RHS of equation into variable list (context, action, interactions)
	vars_list = list(map(str.strip, formula.split('~')[1].strip().split('+')))


	# Add 1 for intercept in variable list if specified
	if include_intercept:
		vars_list.insert(0,1.)

	# Raise assertion error if variable list different length then coeff list
	#print(vars_list)
	#print(coef_list)
	assert(len(vars_list) == len(coef_list))

	# Initialize outcome
	outcome = 0.

	dummy_loops = 0
	for k in range(20):
		dummy_loops += 1
	print(dummy_loops)

	print(str(type(coef_list)))
	print(np.shape(coef_list))
	coef_list = coef_list.tolist()
	print("coef list length: " + str(len(coef_list)))
	print("vars list length: " + str(len(vars_list)))
	print("vars_list " + str(vars_list))
	print("curr_coefs " + str(coef_list))

	## Use variables and coeff list to compute expected reward
	# Itterate over all (var, coeff) pairs from regresion model
	num_loops = 0
	for j in range(len(coef_list)): #var, coef in zip(vars_list,coef_list):
		var = vars_list[j]
		coef = coef_list[j]
		## Determine value in variable list
		# Initialize value (can change in loop)
		value = 1.
		# Intercept has value 1
		if type(var) == float:
			value = 1.

		# Interaction term value
		elif '*' in var:
			interacting_vars = var.split('*')

			interacting_vars = list(map(str.strip,interacting_vars))
			# Product of variable values in interaction term
			for i in range(0, len(interacting_vars)):
				value *= var_dict[interacting_vars[i]]
		# Action or context value
		else:
			value = var_dict[var]

		# Compute expected reward (hypothesized regression model)
		print("value " + str(value) )
		print("coefficient " + str(coef))
		outcome += coef * value
		num_loops += 1
		print("loop number: " + str(num_loops))

	print("Number of loops: " + str(num_loops))
	return outcome
