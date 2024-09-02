import torch
import numpy as np
import time
from pykeops.torch import LazyTensor
#from torchmin import minimize as minimize_torch
import torchmin

from torchmin.function import ScalarFunction
from torchmin.bfgs import BFGS, L_BFGS
from torchmin.line_search import strong_wolfe

from scipy.optimize import OptimizeResult

try:
    from scipy.optimize.optimize import _status_message
except ImportError:
    from scipy.optimize._optimize import _status_message

@torch.no_grad()
def _weighted_optimality_minimize_bfgs_core(
        fun, x0, lr=1., low_mem=False, history_size=100, inv_hess=True,
        max_iter=None, line_search='strong-wolfe', gtol=None, xtol=1e-9,
        normp=float('inf'), callback=None, disp=0, return_all=False):
    """Minimize a multivariate function with BFGS or L-BFGS.

    We choose from BFGS/L-BFGS with the `low_mem` argument.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize
    x0 : Tensor
        Initialization point
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    low_mem : bool
        Whether to use L-BFGS, the "low memory" variant of the BFGS algorithm.
    history_size : int
        History size for L-BFGS hessian estimates. Ignored if `low_mem=False`.
    inv_hess : bool
        Whether to parameterize the inverse hessian vs. the hessian with BFGS.
        Ignored if `low_mem=True` (L-BFGS always parameterizes the inverse).
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to 200 * x0.numel()
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    gtol : TODO explain the change!!!!!!!!!!!!!!!!!.
    xtol : float
        Termination tolerance on function/parameter changes.
    normp : Number or str
        The norm type to use for termination conditions. Can be any value
        supported by `torch.norm` p argument.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    print("Optimality condition successfully overwritten.")
    if gtol is None:
        gtol = 1e-5*torch.ones(len(x0))
    lr = float(lr)
    disp = int(disp)
    if max_iter is None:
        max_iter = x0.numel() * 200
    if low_mem and not inv_hess:
        raise ValueError('inv_hess=False is not available for L-BFGS.')

    # construct scalar objective function
    sf = ScalarFunction(fun, x0.shape)
    closure = sf.closure
    if line_search == 'strong-wolfe':
        dir_evaluate = sf.dir_evaluate

    # compute initial f(x) and f'(x)
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)
    f, g, _, _ = closure(x)
    if disp > 1:
        print('initial fval: %0.4f' % f)
    if return_all:
        allvecs = [x]

    # initial settings
    if low_mem:
        hess = L_BFGS(x, history_size)
    else:
        hess = BFGS(x, inv_hess)
    d = g.neg()
    t = min(1., g.norm(p=1).reciprocal()) * lr
    n_iter = 0

    # BFGS iterations
    for n_iter in range(1, max_iter+1):

        # ==================================
        #   compute Quasi-Newton direction
        # ==================================

        if n_iter > 1:
            d = hess.solve(g)

        # directional derivative
        gtd = g.dot(d)

        # check if directional derivative is below tolerance
        if gtd > -xtol:
            warnflag = 4
            msg = 'A non-descent direction was encountered.'
            break

        # ======================
        #   update parameter
        # ======================

        if line_search == 'none':
            # no line search, move with fixed-step
            x_new = x + d.mul(t)
            f_new, g_new, _, _ = closure(x_new)
        elif line_search == 'strong-wolfe':
            #  Determine step size via strong-wolfe line search
            f_new, g_new, t, ls_evals = \
                strong_wolfe(dir_evaluate, x, t, d, f, g, gtd)
            x_new = x + d.mul(t)
        else:
            raise ValueError('invalid line_search option {}.'.format(line_search))

        if disp > 1:
            print('iter %3d - fval: %0.4f' % (n_iter, f_new))
        if return_all:
            allvecs.append(x_new)
        if callback is not None:
            callback(x_new)

        # ================================
        #   update hessian approximation
        # ================================

        s = x_new.sub(x)
        y = g_new.sub(g)

        hess.update(s, y)

        # =========================================
        #   check conditions and update buffers
        # =========================================

        # convergence by insufficient progress
        if (s.norm(p=normp) <= xtol) | ((f_new - f).abs() <= xtol):
            warnflag = 0
            msg = _status_message['success']
            break

        # update state
        f[...] = f_new
        x.copy_(x_new)
        g.copy_(g_new)
        t = lr

        # convergence by 1st-order optimality
        if (g.abs() <= gtol).sum() == len(x0):
            warnflag = 0
            msg = _status_message['success']
            break

        # precision loss; exit
        if ~f.isfinite():
            warnflag = 2
            msg = _status_message['pr_loss']
            break

    else:
        # if we get to the end, the maximum num. iterations was reached
        warnflag = 1
        msg = _status_message['maxiter']

    if disp:
        print(msg)
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % n_iter)
        print("         Function evaluations: %d" % sf.nfev)
    result = OptimizeResult(fun=f, x=x.view_as(x0), grad=g.view_as(x0),
                            status=warnflag, success=(warnflag==0),
                            message=msg, nit=n_iter, nfev=sf.nfev)
    if not low_mem:
        if inv_hess:
            result['hess_inv'] = hess.H.view(2 * x0.shape)
        else:
            result['hess'] = hess.B.view(2 * x0.shape)
    if return_all:
        result['allvecs'] = allvecs

    return result
    
    
torchmin.bfgs._minimize_bfgs_core = _weighted_optimality_minimize_bfgs_core