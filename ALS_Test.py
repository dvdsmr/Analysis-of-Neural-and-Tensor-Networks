import torch
from FunctionalTensorTrain import Extended_TensorTrain, FourierBasisH1


def f(x):
    """
        x should have shape [b,d] (batch_size, dim)
        output has shape [b,2] 
    """
    o1 = torch.sum(x,dim=1).unsqueeze(1)
    return o1

## TT parameters
d=5
b=1000
indices = [20] * (d)
ranks = [1] + [10] * (d-1) + [1]
# ALS parameters
reg_coeff = 1e-2
TT_iterations = 10
tol = 5e-10
width=2
domain = [[-width,width] for _ in range(d)]


basis = FourierBasisH1(indices, domain)

v_t = Extended_TensorTrain(basis, ranks)

inp = 2*width*torch.rand((b,d))-width
y = f(inp)
rule = None

v_t.ALS_Regression(
           inp,
           y,
           iterations=TT_iterations,
           rule=rule,
           tol=tol,
           verboselevel=2,
           reg_param=reg_coeff,
        )