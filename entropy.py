import numpy as np

# Take limits for special cases
def xlnx(x):
    if x==0 or x==1:
        return 0.0
    return x * np.log2(x)

def xlny(x, y):
    if y==0:
        if x==0:
            return 0.0
        return -np.inf
    return x * np.log2(y)

# Entropies of particle positions encoded via their relative sorting indices
def entropy(X):
    X = np.argsort(X) # Use integer indices instead of floating positions
    nX = len(X)
    DX = np.diff(X)
    nDX = len(DX) # nX-1
    N = np.zeros(2*nDX).astype('int32') # Dx=0 is impossible -> (2*nX-1) - 1 = 2*nDX
    for i in range(nDX):
        N[DX[i] + nDX - int(np.heaviside(DX[i], np.inf))] += 1
    W = N / np.sum(N) # measured probabilities
    H = -np.sum([xlnx(W[i]) for i in range(len(N))]) # measured entropy
    N = np.zeros(2*nDX).astype('int32') # reset counter
    for i in range(nDX):
        N[nDX - (i+1)] = N[nDX + i] = nDX-i
    W_tilde = N / np.sum(N) # max. entropy probabilities. abs(x_i - x_j) = abs(i - j)
    H_tilde = -np.sum([W[i] * np.log2(W_tilde[i]) for i in range(len(N))]) # cross-entropy
    DKL = H_tilde - H # Kullback–Leibler divergence
    H_ref = -np.sum([xlnx(W_tilde[i]) for i in range(len(N))]) # max. reference entropy
    H_0 = -np.log2(W_tilde[nDX]) # initial cross-entropy
    H_rel = (H_tilde - H_0) / (H_ref - H_0) # rel. cross-entropy w.r.t. initial and max. reference entropies
    return H, H_tilde, DKL, H_ref, H_rel

X = np.array([1, 2, 4, 3, 5, 6])
print(entropy(IX))

X = np.array([1, 2, 3, 4, 5, 6])
print(entropy(IX))
