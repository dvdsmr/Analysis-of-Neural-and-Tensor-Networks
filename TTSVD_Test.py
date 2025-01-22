import torch
torch.set_default_dtype(torch.float64)

from TensorTrain import TensorTrain

from matplotlib import pyplot as plt

# generate random full n^d tensor
d = 4
n = 6
dimensions = [n]*d #[6,6,6,6]

A = torch.rand(dimensions)
print("\nGenerated random tensor of shape ", list(A.shape))

# Get a TT representation via TTSVD
TT = TensorTrain.ttsvd(A)
print("\nTT-SVD yielded TT with components of shapes ", [list(TT.comps[i].shape) for i in range(d)])


# Recover the full tensor from the TT-SVD and check consistency
recoveredA = TT.full()
print("\nRelative norm error between original tensor and recovered tensor is ", (torch.linalg.norm(A-recoveredA)/torch.linalg.norm(A)).item())



# Note that the error is at machine accuracy, which is not surprising since we have computed an exact representation of the original tensor
# in the TT format, using maximal ranks.

# The maximal ranks are [1,6,36,6,1]
# we will now compute a truncated TT-SVD with ranks bounded by r, 1 <= r <= 36 and check the relative error of the TT

print("============= Recovery of general random tensor =================")
relErrors = []
for r in range(1,37):
    # Get a TT representation via TTSVD
    TT = TensorTrain.ttsvd(A,ranks=[1,min(r,6),min(r,36),min(r,6),1])
    print("\nTT-SVD yielded TT with components of shapes ", [list(TT.comps[i].shape) for i in range(d)])

    recoveredA = TT.full()
    relError = (torch.linalg.norm(A-recoveredA)/torch.linalg.norm(A)).item()
    print(f"Relative norm error between original tensor and recovered tensor for r={r} is {relError}")
    relErrors.append(relError)
    
plt.semilogy(relErrors,'r-',label="relative error") # logarithmic scaling for 
plt.xlabel("Maximal allowed rank in truncated TT-SVD")
plt.ylabel("Relative norm error between original tensor and TT")
plt.title("TTSVD of a tensor with uniform random entries")
plt.legend()
plt.show()
plt.savefig("plots/ttsvd_error.pdf")


# As we can see, we need ALL ranks to get a satisfying recovery. This makes sense, because the underlying tensor has no inherent structure (just uniform random entries).
# As an academic example, we investigate what happens if we explicitely construst the full tensor such that it can be represented by a low rank TT.

A_comps = [torch.rand(1,6,4), torch.rand(4,6,5), torch.rand(5,6,4), torch.rand(4,6,1)]
A_TT = TensorTrain(dims=dimensions,comp_list=A_comps)
A_full = A_TT.full()

# By construction, A_full can be represented by a TT with ranks [1,4,5,4,1].
# This means that with maximal rank r=5 we should get the exact tensor.
# Let's see if a TT-SVD can uncover this structure.

print("============= Recovery of low rank tensor =================")
relErrors = []
for r in range(1,37):
    # Get a TT representation via TTSVD
    TT = TensorTrain.ttsvd(A_full,ranks=[1,min(r,6),min(r,36),min(r,6),1])
    print("\nTT-SVD yielded TT with components of shapes ", [list(TT.comps[i].shape) for i in range(d)])

    recoveredA = TT.full()
    relError = (torch.linalg.norm(A_full-recoveredA)/torch.linalg.norm(A_full)).item()
    print(f"Relative norm error between original tensor and recovered tensor for r={r} is {relError}")
    relErrors.append(relError)
    
plt.semilogy(relErrors,'r-',label="relative error") # logarithmic scaling for 
plt.xlabel("Maximal allowed rank in truncated TT-SVD")
plt.ylabel("Relative norm error between original tensor and TT")
plt.title("TTSVD of a tensor with ranks bounded 5")
plt.legend()
plt.show()
plt.savefig("plots/ttsvd_error_lowrank.pdf")

exit()