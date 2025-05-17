import torch

# scalar

scalar = torch.tensor(7)

# tensor(7)
print(scalar)


# Get tensor back as Python int
# 7
print(scalar.item())


# Vector
vector = torch.tensor([7, 7])

print(vector)

# 1
print(vector.ndim)

# 2
print(vector.shape)


# Matrix

MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
# 2
print(MATRIX.ndim)

# tensor([ 9, 10])
print(MATRIX[1])

# torch.Size([2, 2])
print(MATRIX.shape)


# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                         [3, 6, 9],
                         [2, 4, 5]]])

# 3
print(TENSOR.ndim)

#tensor([[1, 2, 3],
#        [3, 6, 9],
#        [2, 4, 5]])

print(TENSOR[0])

# torch.Size([1, 3, 3])
print(TENSOR.shape)


# TENSOR 2
TENSOR = torch.tensor([[[[1, 2, 3],
                         [3, 6, 9],
                         [2, 4, 5]]]])

# 4
print(TENSOR.ndim)

#tensor([[[1, 2, 3],
#        [3, 6, 9],
#        [2, 4, 5]]])

print(TENSOR[0])

# torch.Size([1, 1, 3, 3])
print(TENSOR.shape)