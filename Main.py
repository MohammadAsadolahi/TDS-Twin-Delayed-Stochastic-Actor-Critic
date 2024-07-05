import Requirements,Replay_buffer,Actor,Critic,Agent #provided classes in the TDS github repo

algorithm_name="TDS"
enviroment_name='Ant-v4'
seed=4
start_timesteps=10000
def policy_evaluation(agent, enviroment_name,episodes=10):
    evaluation_env = gym.make(enviroment_name)
    average_reward = 0.
    for _ in range(episodes):
        state, _ = evaluation_env.reset()
        done=False
        truncuated=False
        while (not done) and (not truncuated):
            _,action = agent.choose_action(np.array(state))
            state, reward, done, truncuated,_ = evaluation_env.step(action)
            average_reward += reward
    average_reward /= episodes
    return average_reward

env = gym.make(enviroment_name)
env.action_space.seed(seed)
T.manual_seed(seed)
np.random.seed(seed)
agent = Agent(alpha=3e-4, beta=3e-4, 
            input_dims=env.observation_space.shape, tau=0.005,
            action_space_high=env.action_space.high,action_space_low=env.action_space.low,batch_size=100, layer1_size=256, layer2_size=256,
            n_actions=env.action_space.shape[0])
evaluations = [policy_evaluation(agent,enviroment_name)]
average_rewards=[]
total_rewards=[]
steps=0
for ep in range(1,10000000000):
    done=False
    state,_=env.reset(seed=seed)
    rewards=0
    episode_timesteps=0
    truncuated=False
    while (not done) and (not truncuated):
        episode_timesteps+=1
        if steps < start_timesteps:
            action = env.action_space.sample()
        else:
            action,_=agent.choose_action(state)
        agent.learn()
        state_,reward,done,truncuated,info=env.step(action)
        agent.remember(state,action,reward,state_,done)
        rewards+=reward
        steps+=1
        state=state_
        if(steps%5000)==0:
            evaluation_reward=policy_evaluation(agent, enviroment_name)
            evaluations.append(evaluation_reward)
            print(f"Evaluation over {10} episodes: {evaluation_reward:.3f}  step{steps}")
    total_rewards.append(rewards)
    average_rewards.append(sum(total_rewards)/len(total_rewards))
    if(steps>1000000):
        break
    if (ep%200==0):
        if ep<100:
            print(f"episode: {ep}   reward: {rewards}  avg so far:{average_rewards[-1]} steps so far:{steps}")
        else:
            print(f"episode: {ep}   reward: {rewards}  m :{sum(total_rewards[-100:])/len(total_rewards[-100:])} t {average_rewards[-1]}:{steps}    steps so far:{steps}")
    
variant = dict(algorithm=algorithm_name,env=enviroment_name,)
if not os.path.exists(f"./data/{enviroment_name}/{algorithm_name}/seed{seed}"):
    os.makedirs(f'./data/{enviroment_name}/{algorithm_name}/seed{seed}')
with open(f'./data/{enviroment_name}/{algorithm_name}/seed{seed}/variant.json', 'w') as outfile:
    json.dump(variant,outfile)
data = np.array(evaluations)
df = pd.DataFrame(data=data,columns=["Average Return"]).reset_index()
df['Timesteps'] = df['index'] * 5000
df['env'] = enviroment_name
df['algorithm_name'] = algorithm_name
df.to_csv(f'./data/{enviroment_name}/{algorithm_name}/seed{seed}/progress.csv', index = False)
