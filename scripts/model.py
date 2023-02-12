import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, input_n, output_n, fc_n) -> None:
        super(Net, self).__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.fc_n = fc_n
        self.fc1 = nn.Linear(input_n, fc_n)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(fc_n, fc_n)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(fc_n, output_n)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
if __name__ == '__main__' :
    net1 = Net(4, 2, 128)
    net2 = Net(4, 2, 128)
    net2.load_state_dict(net1.state_dict())
    state = torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    out1 = net1(state)
    out2 = net2(state)
    print(out1.shape, out2.shape)