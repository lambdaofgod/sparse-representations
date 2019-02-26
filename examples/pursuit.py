import numpy as np


def pursuit(A, b, n_steps, pursuit_type='omp', wmp_t=None):
    '''
        Run pursuit algorithm (Matching Pursuit, Orthogonal Matching Pursuit or Weak Matching Pursuit)
        to solve
        Ax = b
    '''
    def get_next_approximation_mp(**kwargs):
        a = kwargs['a_next']
        best_column_index = kwargs['best_column_index']
        x_approx = kwargs['x_approx']
        x_approx[best_column_index] += a * np.linalg.norm(res)
        return x_approx

    def get_next_approximation_omp(**kwargs):
        support = kwargs['support']
        x = np.zeros(A.shape[1])
        x_star = np.linalg.lstsq(A[:, support], b, rcond=None)[0]
        x[support] = x_star
        return x
    
    def get_best_column_wmp(inner_products, wmp_t):
        assert wmp_t is not None
        for i, p in enumerate(inner_products):
            residual_error_norm = np.linalg.norm(res)
            if p >= wmp_t * residual_error_norm:
                return i
        return i
    
    res = b
    support = []
    x_approx = np.zeros(A.shape[1])
    
    get_best_column = np.argmax
    if pursuit_type == 'omp':
        get_next_approximation = get_next_approximation_omp
    elif pursuit_type == 'mp':
        get_next_approximation = get_next_approximation_mp
    elif pursuit_type == 'wmp':
        get_next_approximation = get_next_approximation_mp
        get_best_column = lambda products: get_best_column_wmp(products, wmp_t)

    for __ in range(n_steps):
        inner_products = A.T @ res
        best_column_index = get_best_column(inner_products)
        a_next = inner_products[best_column_index]
        #inner_products[best_column_index] = 0
        best_column = A[:, best_column_index]
        support.append(best_column_index)
        x_approx = get_next_approximation(
            a_next=a_next,
            best_column_index=best_column_index,
            support=support,
            x_approx=x_approx)
        res = b - A @ x_approx

    return support, res, x_approx