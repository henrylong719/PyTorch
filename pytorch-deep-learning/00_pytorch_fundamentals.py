import torch
import numpy as np

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

# Reshaping, stacking, squeezing and unsqueezing tensors

# * Reshaping - reshapes an input tensor to a defined shape
# * View - Return a view of an input tensor of certain shape but keep the same memory as the original tensor
# * Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
# * Squeeze - removes all 1 deimensions from a tensor
# * Unsqueeze - add a 1 dimension to a target tensor
# * Permuate - Return a view of the input with dimnesions permuted (swapped) in a certain way


x = torch.arange(1., 10.)

# tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.]) torch.Size([9])
print(x, x.shape)


# tensor([[1.],
#        [2.],
#        [3.],
#        [4.],
#        [5.],
#        [6.],
#        [7.],
#        [8.],
#        [9.]]) torch.Size([9, 1])

x_reshaped = x.reshape(9, 1)
print(x_reshaped, x_reshaped.shape)



# Change the view
z = x.view(1, 9)

# tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.]]) torch.Size([1, 9])
print(z, z.shape)


# Changing z changes x (because a view of a tensor shares the same memory as the original tensor)
z[:, 0] = 5

# tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.]]) tensor([5., 2., 3., 4., 5., 6., 7., 8., 9.])
print(z, x)


# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0)

# tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.],
#        [5., 2., 3., 4., 5., 6., 7., 8., 9.],
#        [5., 2., 3., 4., 5., 6., 7., 8., 9.],
#        [5., 2., 3., 4., 5., 6., 7., 8., 9.]])
print(x_stacked)

x_stacked = torch.stack([x, x, x, x], dim=1)

# tensor([[5., 5., 5., 5.],
#        [2., 2., 2., 2.],
#        [3., 3., 3., 3.],
#        [4., 4., 4., 4.],
#        [5., 5., 5., 5.],
#        [6., 6., 6., 6.],
#        [7., 7., 7., 7.],
#        [8., 8., 8., 8.],
#        [9., 9., 9., 9.]])
print(x_stacked)

# torch squeeze() - removes all single dimensions from a target tensor

x_reshaped = x.reshape(1, 9)

# Previous tensor: tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.]])
print(f"Previous tensor: {x_reshaped}")

# Previous shape: tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.]])
print(f"Previous shape: {x_reshaped}")

# Remove extra dimensions from x_reshaped
x_squeezed = x_reshaped.squeeze()

# New tensor: tensor([5., 2., 3., 4., 5., 6., 7., 8., 9.])
print(f"\nNew tensor: {x_squeezed}")
# New shape: torch.Size([9])
print(f"New shape: {x_squeezed.shape}")


# torch.unsqueeze() - adds a single dimension to a target tensor at a specific dim 
print(f"Previous target: {x_squeezed}")
print(f"Prevous shape: {x_squeezed.shape}")

# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)

# New tensor: tensor([[5., 2., 3., 4., 5., 6., 7., 8., 9.]])
print(f"\nNew tensor: {x_unsqueezed}")

# New shape: torch.Size([1, 9])
print(f"New shape: {x_unsqueezed.shape}")


# torch.permute - rearranges the dimensions of a target tensor in a specified order
x_original = torch.rand(size=(224, 224, 3)) # [height, width, colour_channels]

# permute the original tensor to rearrange the axis (or dim) order
x_premuted = x_original.permute(2, 0 ,1) # shifts axis 0->1, 1->2, 2->0

# Previous shape: torch.Size([224, 224, 3])
print(f"Previous shape: {x_original.shape}")

# New shape: torch.Size([3, 224, 224])
print(f"New shape: {x_premuted.shape}") # [colour_channels, height, width]


x = torch.arange(1, 10).reshape(1, 3, 3)


# tensor([[[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]]]) torch.Size([1, 3, 3])

print(x, x.shape)

# tensor([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

print(x[0])

# tensor([1, 2, 3])
print(x[0][0])

# tensor(1)
print(x[0][0][0])

# tensor(9)
print(x[0][2][2])


# You can also use ":" to select "all" of a target dimension

# tensor([[1, 2, 3]])
print(x[:, 0])


# Get all value of 0th and 1st dimensions but only index 1 of 2nd dimension

# tensor([[2, 5, 8]])
print(x[:, :, 1])

# Get all values of the 0 dimension but only the 1 index value of 1st and 2nd dimension
# tensor([5])
print(x[:, 1, 1])

# Get index 0 of 0th and 1st dimension and all values of 2nd dimension

# tensor([1, 2, 3])
print(x[0, 0, :])


# Index on x to return 9
# tensor([9])

print(x[:, 2, 2])

# Index on x to return 3, 6, 9
# tensor([[3, 6, 9]])

print(x[:, :, 2])


# Pytorch tensors & NumPy

array = np.arange(1.0, 8.0)

# warning: when converting from numpy -> pytorch, pytorch reflects numpy's
# default datatype of float64 unless specified otherwise
tensor = torch.from_numpy(array) # type: ignore

# [1. 2. 3. 4. 5. 6. 7.] tensor([1., 2., 3., 4., 5., 6., 7.])
print(array, tensor)

# Tensor to NumPy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()

# tensor([1., 1., 1., 1., 1., 1., 1.]) [1. 1. 1. 1. 1. 1. 1.]
print(tensor, numpy_tensor)

