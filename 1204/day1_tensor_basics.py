import torch
#创建张量 arange
x = torch.arange(12)
print(x.shape) #通过张量的shape属性访问张量沿着每个轴的长度
#获取张量中元素的总数
print(x.numel())
#改变形状而不改变元素数量和元素值
X = x.reshape(3, 4)
print(X)
#可以创建一个形状为（2,3,4）的张量，其中所有元素都设置为0
print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))
#当然也可以自己设置
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([[1, 2, 3], [4, 5, 6]])
#对于任意具有相同形状的张量， 常见的标准算术运算符（+、-、*、/和**）都可以被升级为按元素运算。
x + y, x - y, x * y
z = torch.exp(x)
#广播机制
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
#索引和切片
X1 = X[-1]
X2 = X[1: 3]
#将torch张量转化为numpy张量
A = X.numpy()
B = torch.tensor(A)
#张量其实就是n维数组
#python通常使用pandas进行数据分析
import os
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', '/Users/mima000000/PycharmProjects/My_Pytorch/data.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
import pandas as pd

data = pd.read_csv(data_file)
print(data)
#接下来对缺失值进行处理  位置索引iloc
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True)) #控制其只进行数值计算
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
#向量将标量从零阶推广到一阶，矩阵将向量从一阶推广到二阶,张量将矩阵推广到n阶
q = torch.arange(20).reshape((5, 4))
print(q)
#矩阵的转置
print(A.T)
#点积、向量积以及矩阵乘法
B = torch.ones(4,3)
torch.mm(A, B)
#范数
u = torch.tensor([3.0, 4.0])
print(torch.norm(u))
