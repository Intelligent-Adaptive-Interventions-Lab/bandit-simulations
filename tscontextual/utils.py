import pandas as pd
import numpy as np
from scipy.stats import invgamma
from typing import Dict


def create_design_matrix(
    input_df: pd.DataFrame, 
    formula: str, 
    add_intercept: bool = True
) -> pd.DataFrame:
    '''
    :param input_df:
    :param formula: for example "y ~ x0 + x1 + x2 + x0 * x1 + x1 * x2"
    :param add_intercept: whether to add dummy columns of 1.
    :return: the design matrix as a dataframe, each row corresponds to a data point, and each column is a regressor in regression
    '''

    D_df = pd.DataFrame()
    input_df = input_df.astype(np.float64)

    formula = str(formula)
    # parse formula
    formula = formula.strip()
    all_vars_str = formula.split('~')[1].strip()
    dependent_var = formula.split('~')[0].strip()
    vars_list = all_vars_str.split('+')
    vars_list = list(map(str.strip, vars_list))

    ''''#sanity check to ensure each var used in
    for var in vars_list:
        if var not in input_df.columns:
            raise Exception('variable {} not in the input dataframe'.format((var)))'''

    # build design matrix
    for var in vars_list:
        if '*' in var:
            interacting_vars = var.split('*')
            interacting_vars = list(map(str.strip, interacting_vars))
            D_df[var] = input_df[interacting_vars[0]]
            for i in range(1, len(interacting_vars)):
                D_df[var] *= input_df[interacting_vars[i]]
        else:
            D_df[var] = input_df[var]

    # add dummy column for bias
    if add_intercept:
        D_df.insert(0, 'Intercept', 1.)

    return D_df


# Posteriors for beta and variance
def posteriors(
    y: np.ndarray, 
    X: np.ndarray, 
    m_pre: np.ndarray, 
    V_pre: np.ndarray, 
    a1_pre: float, 
    a2_pre: float
) -> Dict:
    #y = list of uotcomes
    #X = design matrix
    #priors input by users, but if no input then default
    #m_pre vector 0 v_pre is an identity matrix - np.identity(size of params) a1 & a2 both 2. save the updates
    #get the reward as a spearate vector. figure ut batch size issues (time based)

    # Data size

    datasize = len(y)

    # X transpose
    Xtranspose = np.matrix.transpose(X)

    # Residuals
    # (y - Xb) and (y - Xb)'
    resid = np.subtract(y, np.dot(X,m_pre))
    resid_trans = np.matrix.transpose(resid)

    # N x N middle term for gamma update
    # (I + XVX')^{-1}
    mid_term = np.linalg.inv(np.add(np.identity(datasize), np.dot(np.dot(X, V_pre),Xtranspose)))

    ## Update coeffecients priors

    # Update mean vector
    # [(V^{-1} + X'X)^{-1}][V^{-1}mu + X'y]
    m_post = np.dot(np.linalg.inv(np.add(np.linalg.inv(V_pre), np.dot(Xtranspose,X))), np.add(np.dot(np.linalg.inv(V_pre), m_pre), np.dot(Xtranspose,y)))

    # Update covariance matrix
    # (V^{-1} + X'X)^{-1}
    V_post = np.linalg.inv(np.add(np.linalg.inv(V_pre), np.dot(Xtranspose,X)))

    ## Update precesion prior

    # Update gamma parameters
    # a + n/2 (shape parameter)
    a1_post = a1_pre + datasize/2

    # b + (1/2)(y - Xmu)'(I + XVX')^{-1}(y - Xmu) (scale parameter)
    a2_post = a2_pre + (np.dot(np.dot(resid_trans, mid_term), resid))/2

    ## Posterior draws

    # Precesions from inverse gamma (shape, loc, scale, draws)
    precesion_draw = invgamma.rvs(a1_post, 0, a2_post, size = 1)

    # Coeffecients from multivariate normal
    beta_draw = np.random.multivariate_normal(np.array(m_post, dtype=np.float64), np.array(precesion_draw * V_post, dtype=np.float64))

    # List with beta and s^2
    #beta_s2 = np.append(beta_draw, precesion_draw)

    # Return posterior drawn parameters
    # output: [(betas, s^2, a1, a2), V]
    return {
        "coef_mean": m_post,
        "coef_cov": V_post,
        "variance_a": a1_post,
        "variance_b": a2_post
    }
