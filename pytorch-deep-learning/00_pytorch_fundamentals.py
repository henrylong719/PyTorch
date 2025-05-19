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



# Manipulating Tensors (tensor operations)

# Tensor operations include:
# * Addition
# * Subtraction
# * Multiplication (element-wise)
# * Division
# * Matrix multiplication

# Create a tensor
tensor = torch.tensor([1 ,2, 3])

# tensor([11, 12, 13])
print(tensor + 10)

# tensor([10, 20, 30])
print(tensor * 10)

# tensor([-9, -8, -7])
print(tensor - 10)

# Try out pytorch in-build functions

# tensor([10, 20, 30])
print(torch.mul(tensor, 10))

# tensor([11, 12, 13])
print(torch.add(tensor, 10))


# Matrix multiplication

# Two main ways of performing multiplication in neural networks and deep learning

# 1. Element-wise multiplication
# 2. Matrix multiplication (dot product)

# Element wise multiplication

# tensor([1, 2, 3]) * tensor([1, 2, 3])
# Equals: tensor([1, 4, 9])
print(tensor, "*", tensor)
print(f"Equals: {tensor * tensor}")

# Matrix multiplication
# tensor(14)
print(torch.matmul(tensor, tensor))

# Matrix multiplication by hand
# 1 * 1 + 2 * 2 + 3 * 3


# One of the most common errors in deep learning: shape errors

# Shapes for matrix multiplication
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])


tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                          [9, 12]])


# torch.mm(tensor_A, tensor_B) # torch.mm is the same as torch.matmul

# tnsor shape issue 
# torch.matmul(tensor_A, tensor_B)


# To fix our tensor shape issues, we can manipulate the shape of one of our tensors using a transpose.
# A transpose switches the axes or dimensions of a given tensor.

#  torch.Size([3, 2])
print(tensor_B, tensor_B.shape)

# torch.Size([2, 3])
print(tensor_B.T, tensor_B.T.shape)


# tensor([[ 27,  30,  33],
#        [ 61,  68,  75],
#        [ 95, 106, 117]])
print(torch.matmul(tensor_A, tensor_B.T))


# Finding the min, max, mean, sum, etc (tensor aggregation)

x = torch.arange(0, 100, 10)

# tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) torch.int64
print(x, x.dtype)

# tensor(0) tensor(0)
print(torch.min(x), x.min())

# tensor(90) tensor(90)
print(torch.max(x), x.max())

# tensor(45.) tensor(45.)
print(torch.mean(x.type(torch.float32)), x.type(torch.float32).mean())

# tensor(450) tensor(450)
print(torch.sum(x), x.sum())


# Finding the positional min and max

# tensor(0)
print(x.argmin())

# tensor(9)
print(x.argmax())

