class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
    

    def forward(self, state, action):
        q = self.fc1(T.cat([state, action], dim=1))
        q = F.relu(q)
        q = self.fc2(q)
        q = F.relu(q)
        q = self.q(q)
        return q
