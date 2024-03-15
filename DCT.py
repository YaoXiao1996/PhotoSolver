import numpy as np
import scipy
import scipy.fftpack as spfft

np.set_printoptions(linewidth=np.inf)
N = 4

# Transformation matrix
Psi = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == 0:
            Psi[i, j] = np.sqrt(1 / N) * np.cos(np.pi * (j + 0.5) * i / N)
        else:
            Psi[i, j] = np.sqrt(2 / N) * np.cos(np.pi * (j + 0.5) * i / N)
Psi = Psi.T

I_mn = np.eye(N, N)
test_psi = spfft.dct(I_mn, norm='ortho')

print(Psi)
print(test_psi)

print(Psi @ Psi.T)
print(test_psi @ test_psi.T)
