import numpy as np

m, n = 128, 256
w = np.random.randn(m, n).astype(np.float32) * 0.1
u, s, vh = np.linalg.svd(w, full_matrices=False)

# Rank 50%
k = m // 2
uk = u[:, :k]
sk = s[:k]
vhk = vh[:k, :]

# Reconstruct
w_approx = uk @ np.diag(sk) @ vhk

# MSE
mse = np.mean((w - w_approx)**2)
print(f"Weight MSE (Rank {k}/{m}): {mse:.8f}")

# Forward MSE
x = np.random.randn(1, m).astype(np.float32)
out = x @ w
out_approx = x @ w_approx
out_mse = np.mean((out - out_approx)**2)
print(f"Forward MSE: {out_mse:.8f}")
