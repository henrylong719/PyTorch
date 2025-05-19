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
TENSOR2 = torch.tensor([[[[1, 2, 3],
                         [3, 6, 9],
                         [2, 4, 5]]]])

# 4
print(TENSOR2.ndim)

#tensor([[[1, 2, 3],
#        [3, 6, 9],
#        [2, 4, 5]]])

print(TENSOR2[0])

# torch.Size([1, 1, 3, 3])
print(TENSOR2.shape)


# Create a random tensor of size (3,4)
random_tensor = torch.rand(3, 4)


# tensor([[0.3484, 0.8254, 0.7166, 0.2428],
#        [0.0399, 0.9386, 0.7947, 0.7323],
#        [0.1927, 0.4838, 0.4699, 0.0595]])
print(random_tensor)


# 2
print(random_tensor.ndim)

# Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(224, 224, 3)) # height, width, colour channel

# torch.Size([224, 224, 3])
print(random_image_size_tensor.shape)

# 3
print(random_image_size_tensor.ndim)


# Zeros and ones

# tensor([[0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.]])

zeros = torch.zeros(size=(3,4))
print(zeros)


# tensor([[1., 1., 1., 1.],
#        [1., 1., 1., 1.],
#        [1., 1., 1., 1.]])

ones = torch.ones(size=(3, 4))
print(ones)


# torch.float32
print(ones.dtype)


# Creata a range of tensors and tensors-like

one_to_ten = torch.arange(start=1,end=11, step=1)

# tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
print(one_to_ten)

ten_zeros = torch.zeros_like(input=one_to_ten)

# tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
print(ten_zeros)



# Tensor datatypes

# Tensor datatypes is one of the 3 big errors you'll run into with PyTorch & deep learning

# 1. Tensors not right datatype
# 2. Tensors not right shapte
# 3. Tensors not on the right device

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # what datatype is the tensor (e.g. float32 or float64)
                               device=None, # What device is your  tensor on
                               requires_grad=False) # whether or not to track gradients with this tensors operations

# tensor([3., 6., 9.])
print(float_32_tensor)
# torch.float32
print(float_32_tensor.dtype)


float_16_tensor = float_32_tensor.type(torch.float16)
# tensor([3., 6., 9.], dtype=torch.float16)
print(float_16_tensor)


unknown = float_16_tensor * float_32_tensor

# tensor([ 9., 36., 81.])
print(unknown)
# torch.float32
print(unknown.dtype)


# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out deails about some tensor

# tensor([[0.7939, 0.7944, 0.0728, 0.5200],
#        [0.4218, 0.2052, 0.5537, 0.6579],
#        [0.9548, 0.3817, 0.1586, 0.4500]])
# Datatype of tensor: torch.float32
# Shape of tensor: torch.Size([3, 4])
# Device tensor is on: cpu
print(some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device tensor is on: {some_tensor.device}")




