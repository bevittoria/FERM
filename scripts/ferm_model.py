"""General FERM particle sampler, exact conditional RM law and scoring.

Inputs are node populations/ISO codes, distances and an arbitrary shift matrix.
No Asia data, calibration choices or plotting dependencies live here.
The numerical implementation is unchanged from the original model.
"""
from typing import Optional
import numpy as np
from scipy import stats

def gaussian_max_sample_vec(
    mu: float,
    sigma: float,
    n: int,
    size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample the maximum of `n` i.i.d. Gaussian random variables.

    If X_1, ..., X_n ~ N(mu, sigma^2), this function returns `size`
    samples from max(X_1, ..., X_n) using inverse-transform sampling.

    Parameters
    ----------
    mu : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution. Must be non-negative.
    n : int
        Number of Gaussian draws whose maximum is considered. Values < 1
        are clipped to 1.
    size : int
        Number of samples to generate.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape `(size,)` containing samples of the maximum.

    Raises
    ------
    ValueError
        If `sigma < 0` or `size < 0`.
    """
    if sigma < 0:
        raise ValueError("`sigma` must be non-negative.")
    if size < 0:
        raise ValueError("`size` must be non-negative.")

    n = max(int(n), 1)

    if sigma == 0:
        return np.full(size, float(mu), dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    u = rng.random(size=size)
    q = np.exp(np.log(u) / n)  # equivalent to u ** (1 / n)
    return mu + sigma * stats.norm.ppf(q)


def simulate(nodes,distance,shifts,sigma,particles,seed):
    rng=np.random.default_rng(seed);n=len(nodes);pop=np.rint(nodes.population).astype(int).to_numpy()
    codes=nodes.iso3.tolist();counts=np.zeros((n,n),dtype=np.int64)
    for i in range(n):
        threshold=gaussian_max_sample_vec(shifts[i,i],sigma,pop[i],particles,rng)
        remaining=np.ones(particles,dtype=bool)
        for j in sorted((j for j in range(n) if j!=i),key=lambda j:(distance[i,j],codes[j])):
            indices=np.flatnonzero(remaining)
            if not len(indices):break
            offers=gaussian_max_sample_vec(shifts[i,j],sigma,pop[j],len(indices),rng)
            winners=indices[offers>threshold[indices]]
            counts[i,j]=len(winners);remaining[winners]=False
    return counts


def rm_law(nodes,distance):
    n=len(nodes);p=np.zeros((n,n));pop=np.rint(nodes.population).to_numpy();codes=nodes.iso3.tolist()
    for i in range(n):
        s=0.
        for j in sorted((j for j in range(n) if j!=i),key=lambda j:(distance[i,j],codes[j])):
            p[i,j]=pop[i]*pop[j]/((pop[i]+s)*(pop[i]+s+pop[j]));s+=pop[j]
        p[i]/=p[i].sum()
    return p


def estimate(counts,rm,pseudo):
    return (counts+pseudo*rm)/(counts.sum(axis=1)[:,None]+pseudo)


def bits(p,counts):
    positive=counts>0
    return float(-np.sum(counts[positive]*np.log2(p[positive]))/counts.sum())


def metrics(p,counts,rm):
    predicted=counts.sum(axis=1)[:,None]*p
    return dict(bits=bits(p,counts),gain_bits=bits(rm,counts)-bits(p,counts),
        cpc=float(2*np.minimum(predicted,counts).sum()/(predicted.sum()+counts.sum())),
        destination_TV=float(.5*abs(predicted.sum(axis=0)-counts.sum(axis=0)).sum()/counts.sum()))


def job(args):
    cid,rep,nodes,d,m,sigma,particles,seed=args
    return cid,rep,simulate(nodes,d,m,sigma,particles,seed)



# Origin-constrained gravity benchmark: fit on training counts only.
from scipy.optimize import minimize
from scipy.special import logsumexp

def cross_entropy(logp, weights):
    """Bits per event; zero-weight entries never multiply -inf."""
    lp, w = np.asarray(logp, float), np.asarray(weights, float)
    if lp.shape != w.shape or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Require matching arrays and nonnegative positive-total weights")
    used = w > 0
    return float(-np.sum(w[used]*lp[used])/w.sum()/np.log(2))

def normalize_logits(logits):
    logits = np.asarray(logits, float).copy()
    np.fill_diagonal(logits, -np.inf)
    return logits-logsumexp(logits, axis=1, keepdims=True)

def gravity_terms(nodes, distances, feature):
    n = len(nodes)
    d = distances.to_numpy(float).copy()
    off = ~np.eye(n, dtype=bool)
    if (d[off] <= 0).any():
        raise ValueError("Require strictly positive off-diagonal distances")
    np.fill_diagonal(d, 1.)
    pop = np.broadcast_to(np.log(nodes.population.to_numpy(float)), (n, n))
    return np.stack([pop, -np.log(d), np.asarray(feature)], axis=-1)

def fit_loglinear(terms, counts, bounds, offset=None):
    """Bounded conditional multinomial MLE with analytic gradient, no penalty."""
    x = np.asarray(terms, float)
    n, _, k = x.shape
    off = ~np.eye(n, dtype=bool)
    if k != len(bounds) or counts.shape != (n, n) or counts.sum() <= 0:
        raise ValueError("Invalid log-linear fit dimensions/counts")
    x = x-x[off].mean(axis=0)
    base = np.zeros((n, n)) if offset is None else np.asarray(offset).copy()
    np.fill_diagonal(base, 0.)
    weights = counts/counts.sum()

    def objective(beta):
        lp = normalize_logits(base+x@beta)
        residual = weights.sum(axis=1)[:, None]*np.exp(lp)-weights
        return (-np.sum(weights[off]*lp[off]), np.sum(residual[:, :, None]*x, axis=(0, 1)))

    start = np.array([max(lo, min(1., hi)) if hi is not None else max(lo, 1.) for lo, hi in bounds])
    fit = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
                   options=dict(ftol=1e-14, gtol=1e-9, maxiter=1000, maxls=50))
    projected = fit.jac.copy()
    for j, (lo, hi) in enumerate(bounds):
        if fit.x[j] <= lo+1e-7 and projected[j] > 0:
            projected[j] = 0
        if hi is not None and fit.x[j] >= hi-1e-7 and projected[j] < 0:
            projected[j] = 0
    if np.max(np.abs(projected)) > 1e-6:
        raise RuntimeError(f"Log-linear fit did not converge: {fit.message}; {projected}")
    lp = normalize_logits(base+x@fit.x)
    return lp, dict(parameters=fit.x.tolist(), free_parameters=k,
                   boundary=bool(any(abs(v-lo) < 1e-6 or (hi is not None and abs(v-hi) < 1e-6)
                                     for v, (lo, hi) in zip(fit.x, bounds))),
                   train_bits=cross_entropy(lp, counts), projected_gradient=float(np.max(np.abs(projected))))
