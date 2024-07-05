class Agent():
    def __init__(self, alpha, beta, input_dims, tau, action_space_high,action_space_low,
            gamma=0.99, actor_update_interval=2,n_actions=2, max_size=1000000, layer1_size=400,
            layer2_size=300, batch_size=100):
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)
        
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.actor = ActorNetwork(input_dims, layer1_size,layer2_size, n_actions,action_space_high[0])
        self.actor_optimizer=optim.Adam(params=self.actor.parameters(), lr=alpha)
        self.critic_1 = CriticNetwork(input_dims, layer1_size,layer2_size,n_actions)
        self.critic_1_optimizer=optim.Adam(params=self.critic_1.parameters(), lr=beta)
        self.critic_2 = CriticNetwork(input_dims, layer1_size,layer2_size,n_actions)
        self.critic_2_optimizer=optim.Adam(params=self.critic_2.parameters(), lr=beta)

        self.target_actor = ActorNetwork(input_dims, layer1_size,layer2_size, n_actions,action_space_high[0])
        self.target_critic_1 = CriticNetwork(input_dims, layer1_size,layer2_size,n_actions)
        self.target_critic_2 = CriticNetwork(input_dims, layer1_size,layer2_size,n_actions)
        
        self.gamma = gamma
        self.tau = tau
        self.action_space_high = action_space_high
        self.action_space_low = action_space_low
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.update_actor_iter =  actor_update_interval
        self.learn_step_cntr = 0
        self.time_step = 0
        self.update_network_parameters(tau=1)
        
    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        self.actor.eval()
        with T.no_grad():
            mu, sigma = self.actor.forward(state)
        self.actor.train()
        noise = (T.randn_like(mu) * sigma).clamp(-0.5*self.action_space_high[0], 0.5*self.action_space_high[0])
        action=T.clamp((mu + noise),self.action_space_low[0],self.action_space_high[0])
        return action.numpy(),mu.numpy()
      

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.device)             
        done = T.tensor(done, dtype=T.float).to(self.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        
        with T.no_grad():
            mu, sigma = self.target_actor.forward(state_)
            noise = (T.randn_like(action) * (0.2*self.action_space_high[0])).clamp(-0.5*self.action_space_high[0], 0.5*self.action_space_high[0])
            target_actions = (mu + noise).clamp(self.action_space_low [0],self.action_space_high[0])
            q1_ = T.squeeze(agent.target_critic_1.forward(state_, target_actions))
            q2_ = T.squeeze(agent.target_critic_2.forward(state_, target_actions))

        q1 = T.squeeze(agent.critic_1.forward(state, action))
        q2 =T.squeeze(agent.critic_2.forward(state, action))
        
        critic_value_ = T.min(T.squeeze(q1_),T.squeeze(q2_))
        target = reward + self.gamma * (1 - done) * (critic_value_)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        q1_loss = F.mse_loss(q1,target)
        q2_loss = F.mse_loss(q2,target)
        q1_loss.backward()
        q2_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        self.learn_step_cntr += 1
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return
        self.actor_optimizer.zero_grad()
        mean, std = self.actor.forward(state)
        action_distribution=T.distributions.Normal(mean.detach(), std) 
#         actor_min_Q_loss = self.critic_1.forward(state, mean)
        actor_min_Q_loss = T.min(self.critic_1.forward(state, mean),self.critic_2.forward(state, mean))
        actor_mu_loss = T.mean(T.sum(- action_distribution.log_prob(mean) * actor_min_Q_loss.detach(),axis=0) - actor_min_Q_loss)
        actor_mu_loss.backward()
        self.actor_optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
