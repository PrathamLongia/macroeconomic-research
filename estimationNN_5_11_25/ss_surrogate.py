import torch
class Steady_State_Surrogate(torch.nn.Module):
    """Maps parameter sets to steady-state distributions g(x,y,z), s(x,y,z)"""
    def __init__(self, 
                 n_par=0,  # number of parameters, including fixed ones
                 nx=7,
                 ny=8,
                 nz=2,
                 activation=torch.nn.CELU(),
                 nn_width=64,
                 nn_num_layers=4,
                 c_tightness=False,
                 ):
        super(Steady_State_Surrogate, self).__init__()
        layers = [torch.nn.Linear(n_par, nn_width), activation]
        for i in range(1, nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(activation)
        if c_tightness:
            layers.append(torch.nn.Linear(nn_width, nx * ny * nz * 2 + 1))
        else:
            layers.append(torch.nn.Linear(nn_width, nx * ny * nz * 2))
        self.net = torch.nn.Sequential(*layers)
        for i in range(0, nn_num_layers):
            torch.nn.init.xavier_normal_(self.net[2 * i].weight)

    def forward(self, X, device=None):
        output = self.net(X)
        output = output.to(device) if device is not None else output
        if device == 'cpu':
            output = output.detach().numpy()
        return output
