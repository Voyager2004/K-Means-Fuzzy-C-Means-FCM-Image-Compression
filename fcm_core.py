# fcm_core.py
import numpy as np
import skfuzzy as fuzz

def run_fcm(X, initial_labels, m=2.0, max_iter=5, error=1e-2, seed=0):
    """
    X              : (N,3) float32 RGB [0-255]
    initial_labels : (N,)  K-Means 的硬标签
    return         : centroids(K,3), u(K,N)
    """
    N, _ = X.shape
    K = int(initial_labels.max()) + 1

    U_init           = np.zeros((K, N), dtype=np.float32)
    U_init[initial_labels, np.arange(N)] = 1.0  # 0/1 隶属度

    cntr, u, *_ = fuzz.cluster.cmeans(
        X.T, c=K, m=m,
        error=error, maxiter=max_iter,
        init=U_init, seed=seed
    )
    return cntr.astype(np.float32), u
