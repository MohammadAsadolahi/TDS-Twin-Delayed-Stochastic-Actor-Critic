# TDS
### Official implementation of Twin delayed actor critic algorithm (a Stochastic Off-Policy Actor-Critic Algorithm for contiuous RL problems)
#### Preprint of the research paper: https://www.researchsquare.com/article/rs-3041837/v1
In the TDS resreach paper we aim to learn exploration mindfully according to the feedbacks in continuous RL 
**this project is a research project and owners have no responsibility of any loss for deploying this project in real environmentn**   
## Abstract  
**In both discrete and continuous domains, model-free reinforcement learning algorithms have been successfully applied to the vast majority of reinforcement learning problems and are the main solution to real world problems. In reinforcement learning problems with continuous action space, state-of-the-art algorithms are extremely sample-inefficient and need lots of training interactions to become proficient, which could be catastrophically expensive and infeasible in real-world problems. So far, frontier algorithms have not used well-defined methods to explore the decision space. Exploring new behaviors is a prerequisite to look for optimal policies. All of the leading algorithms in the field leverage a blind form of exploration added to agent decisions to search for better policies. Such solutions fail to mindfully explore the environment, disrupting the learning process. This makes these algorithms very prone to failing in specific domains. In this research, a novel stochastic Off-Policy Actor-Critic algorithm, TDS for short, is presented. Combining the policy gradient theorem with the deterministic policy gradient, the TDS algorithm can learn how to mindfully explore the environment. The proposed update method enables TDS to learn how to modify the decision stochasticity bonds for each state and action. This is done according to gradients information derived from learning feedbacks. Evaluations in MuJoCo and Box2D tasks show faster convergence or outperform the state-of-the-art algorithms including TD3, SAC, and DDPG in every environment tested**

[feel free to ask any question in Issues or just email me]

![last100kpeprformance](https://github.com/MohammadAsadolahi/TDS-Twin-Delayed-Stochastic-Actor-Critic/blob/main/Plots/TDS%20-%20Table%203.png)  

![Relative performance](https://github.com/MohammadAsadolahi/TDS-Twin-Delayed-Stochastic-Actor-Critic/blob/main/Plots/TDS%20-%20Table%204.png)  


### How to install requirements
The `requirements.txt` file should list all Python libraries that the project depend on, and they will be installed using:
```
pip install -r Requirements.txt
```
I keep updating the project to be compatible with new versions of libraries. If there was any problem with the diffrent versions of the required libraries let me know in the "Issues" section, so i can resolve them.  
  
**this code is implemented with Pytorch! i will add TensorFlow version soon and link to it here!!**  
## Architecture  
![TDS](https://github.com/MohammadAsadolahi/TDS-Twin-Delayed-Stochastic-Actor-Critic/blob/main/TDS.png)  
## Learning plots  
<div align="center">
	<img src="https://github.com/MohammadAsadolahi/TDS-Twin-Delayed-Stochastic-Actor-Critic/blob/main/Plots/TDS-%20learning%20curves.png">
</div>
for detailed learning curves for each environment proceed to the Plots folder of the repo    
