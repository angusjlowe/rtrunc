import numpy as np
import time
from mps_helpers import *

n=9
d=2
l=5
bond_dim = 16
t1 = time.time()
tensors = get_random_mps(n,d,bond_dim)
tensors = normalize(tensors)
t2 = time.time()
print("Time taken for first get_random_mps is {:.3f}s".format(t2-t1))
tensors = power_law_schmidt_coeffs(tensors,0.2)
# check if resulting vector has unit norm
psi = get_vector_from_mps(tensors)
print("norm is {:.3f}".format(np.linalg.norm(psi)))

# check right canonical form
print("\nChecking right canonical form...\n")
psi_original = get_vector_from_mps(tensors)
right = right_canonical_form(tensors)
psi_right = get_vector_from_mps(right)
print("Norm preserved?", np.allclose(np.linalg.norm(psi_original),np.linalg.norm(psi_right),1))
right_mps = get_mps_tensors_from_canonical(right)
print("equal?", np.allclose(1., mps_inner_prod(tensors, right_mps)))

# check if mixed_canonical_form preserves norm
print("\nChecking mixed canonical form...\n")
psi_original = get_vector_from_mps(tensors)
mixed = mixed_canonical_form(tensors, l)
psi_mixed = get_vector_from_mps(mixed)
print("Norm preserved?", np.allclose(np.linalg.norm(psi_original), np.linalg.norm(psi_mixed), 1))
new_mps = get_mps_tensors_from_canonical(mixed)
print("equal?", np.allclose(1., mps_inner_prod(tensors, new_mps)))


# check if mixed canonical form has normalized schmidts
print("\nChecking normalization of Schmidt coefficients...\n")
Sl = mixed[l]
print("Dimensions of (l+1)th tensor is {}".format(Sl.shape))
print("Norm is: {:.3f}".format(np.linalg.norm(Sl)))

t3 = time.time()
print("Time for everything up to this point is {:.3f}s".format(t3-t2))
