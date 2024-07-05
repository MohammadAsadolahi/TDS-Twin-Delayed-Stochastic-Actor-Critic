class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims,n_actions,max_action):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action=max_action
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.ln1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.ln2 = nn.LayerNorm(self.fc2_dims)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

    def forward(self, state):
        prob =self.fc1(state)
        prob=self.ln1(prob)
        prob = F.relu(prob)
        prob =self.fc2(prob)
        prob=self.ln2(prob)
        prob = F.relu(prob)

        mu =self.max_action * T.tanh(self.mu(prob))
        sigma =F.sigmoid(self.sigma(prob)).clamp(min=0.1*self.max_action, max=1*self.max_action)
        return mu,sigma
