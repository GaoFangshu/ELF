import torch
import numpy as np


# Calculation
np_data = np.arange(6).reshape((2, 3))
torch_data =  torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

data = [[1, 2], [3, 4]]
data = np.array(data)
tensor = torch.FloatTensor(data)
np.matmul(data, data)
data.dot(data)
torch.mm(tensor, tensor)
tensor.dot(tensor)    # Different from np.dot


# Variable
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, require_grad=True)    # Make graph by variable, require_grad to backpropogation
v_out = torch.mean(variable*variable)

v_out.backward()
print(variable.grad)    # v_out = 1/4*sum(var*var)  d(v_out)/d(var) = 1/2*var = var/2
print(variable.data)
print(variable.data.numpy())
# Variable of state?


# Activation Function
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
x = torch.linspace(-5, 5, 200)    # x data (tensor), shape=(100, 1)
x = Variable(x)
x_np = x.data.numpy()    # matplotlib cannot config torch

y_relu = F.relu(x).data.numpy()    # input is Variable
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
# y_softmax = F.softmax(x)


# Regression
from torch.autograd import Variable
import torch.nn.functional as F
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)   # only Variable can be input

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        # information that layer needs
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(in_features=n_feature, out_features=n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))    # through hidden layyer and to Relu
        x = self.predict(x)
        pass

net = Net(1, 10, 1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for t in range(100):   # number of pace
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()





# Class of actor
class Actor:
    def __init__(self):

        # Loss

    def learn(self, s, a, td):    # td: TD error

    def choose_action(self, s):

# Class of critic
class Critic:
    def __init__(self):

    def learn(self, s, r, s_):