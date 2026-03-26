import math
import numpy as np


def _to_numpy_array(data):
    """Convert input data to a 1D NumPy array."""
    arr = np.array(data)
    if arr.size == 0:
        raise ValueError("data must be non-empty")
    return arr


def _validate_bernoulli_data(data):
    """Validate Bernoulli data: non-empty and only 0/1 values."""
    arr = _to_numpy_array(data)

    if not np.all(np.isin(arr, [0, 1])):
        raise ValueError("Bernoulli data must contain only 0 and 1")

    return arr.astype(float)


def _validate_poisson_data(data):
    """Validate Poisson data: non-empty, nonnegative integers."""
    arr = _to_numpy_array(data)

    if np.any(arr < 0):
        raise ValueError("Poisson data must be nonnegative")

    if not np.all(np.equal(arr, np.floor(arr))):
        raise ValueError("Poisson data must contain integer values only")

    return arr.astype(int)


def bernoulli_log_likelihood(data, theta):
    """
    Compute the Bernoulli log-likelihood for binary data.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    theta : float
        Bernoulli parameter, must satisfy 0 < theta < 1.

    Returns
    -------
    float
        Log-likelihood value.
    """
    arr = _validate_bernoulli_data(data)

    if not (0 < theta < 1):
        raise ValueError("theta must satisfy 0 < theta < 1")

    log_likelihood = np.sum(arr * np.log(theta) + (1 - arr) * np.log(1 - theta))
    return float(log_likelihood)


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    """
    Estimate the Bernoulli MLE and compare candidate theta values.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    candidate_thetas : array-like or None
        Candidate theta values for comparison.

    Returns
    -------
    dict
        Dictionary containing MLE, counts, log-likelihoods, and best candidate.
    """
    arr = _validate_bernoulli_data(data)

    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    num_successes = int(np.sum(arr))
    num_failures = int(arr.size - num_successes)
    mle = float(np.mean(arr))

    log_likelihoods = {}
    best_candidate = None
    best_ll = None

    for theta in candidate_thetas:
        ll = bernoulli_log_likelihood(arr, theta)
        log_likelihoods[theta] = ll

        if best_ll is None or ll > best_ll:
            best_ll = ll
            best_candidate = theta

    return {
        "mle": mle,
        "num_successes": num_successes,
        "num_failures": num_failures,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate,
    }


def poisson_log_likelihood(data, lam):
    """
    Compute the Poisson log-likelihood for count data.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    lam : float
        Poisson rate parameter, must satisfy lam > 0.

    Returns
    -------
    float
        Log-likelihood value.
    """
    arr = _validate_poisson_data(data)

    if lam <= 0:
        raise ValueError("lam must be > 0")

    log_factorials = np.array([math.lgamma(int(x) + 1) for x in arr], dtype=float)
    log_likelihood = np.sum(arr * np.log(lam) - lam - log_factorials)
    return float(log_likelihood)


def poisson_mle_analysis(data, candidate_lambdas=None):
    """
    Estimate the Poisson MLE and compare candidate lambda values.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    candidate_lambdas : array-like or None
        Candidate lambda values for comparison.

    Returns
    -------
    dict
        Dictionary containing MLE, summary stats, log-likelihoods,
        and best candidate.
    """
    arr = _validate_poisson_data(data)

    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    total_count = int(np.sum(arr))
    n = int(arr.size)
    sample_mean = float(np.mean(arr))
    mle = sample_mean

    log_likelihoods = {}
    best_candidate = None
    best_ll = None

    for lam in candidate_lambdas:
        ll = poisson_log_likelihood(arr, lam)
        log_likelihoods[lam] = ll

        if best_ll is None or ll > best_ll:
            best_ll = ll
            best_candidate = lam

    return {
        "mle": mle,
        "sample_mean": sample_mean,
        "total_count": total_count,
        "n": n,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate,
    }
