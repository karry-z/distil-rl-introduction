# Chapter 6. Temporal-Difference Learning

Temporal-Difference (TD) Learning is a combination of Monte Carlo ideas and dynamic programming (DP). Like MC methods, TD methods can learn directly from raw experience without a model of the environment’s dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).

As usual, we start by focusing on the policy evaluation or prediction problem, the problem of estimating the value function $v_\pi$ for a given policy $\pi$. For the control problem (finding an optimal policy), DP, TD, and MC methods all use some variation of generalized policy iteration (GPI). The differences in the methods are primarily differences in their approaches to the prediction problem.

## 6.1 TD prediction

- Two difference representation of prediction form: 
	$v_{\pi}(s) \dot= E[G_t | S_t = s] = E[R_{t+1} + \gamma G_{t+1}| S_t = s] = E[R_{t+1} + \gamma v_{\pi}(S_{t+1})| S_t = s]$

- MC prediction (constant alpha MC): $V(S_t) \leftarrow V(S_t) + \alpha (G_t - S_t)$ with $G_t$ as the single realisation each time.

- TD prediction (TD(0) / one-step TD): $V(S_t) \leftarrow V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - S_t)$ with $R_{t+1} + \gamma V(S_{t+1})$ as the single realisation of $R_{t+1} + \gamma v_{\pi}(S_{t+1})$ each time. 
	- apparently, TD combines the sampling of MC and the bootstrapping of DP
	- the term $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is called TD error, written as $\delta_t$
	
- TD and MC updates are referred as *sample updates*, while DP update is referred as *expected updates*

- Tabular TD(0) prediction:

	- Input: the policy $\pi$ to be evaluated
	- Algorithm parameter: step size $\alpha \in (0,1]$
	- Initialize V(s), for all $s \in S^+$ arbitrarily except that V(terminal) = 0
	- Loop for each episode:
		- Initialize S
		- Loop for each step of episode:
			- $A \leftarrow$ action given by $\pi$ for S
			- Take action A, observe $R, S\prime$
			- $V(S) \leftarrow V(S) + \alpha (R + \gamma V(S\prime) - V(S))$
			- $S \leftarrow S\prime$
		- until S is terminal


- Advantages of TD prediction:

	- over DP: TD methods do not need a model of the environment
	- over MC: TD methods are naturally implemented in an online, fully incremental fashion, i.e., they do not require to wait until the end of an episode (e.g., see the algo above, to implement it we do not need to generate any episodes beforehand, but simply update during the actions being taken)
	- In practive, TD methods have usually been found to  converge faster than constant-alpha MC methods on stochastic taks.

## Sarsa: On-policy TD Control

- update rule: 
	$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$


- Sarsa (on-policy TD control) for estimating $Q \approx q_{\star}$
	- Algorithm parameter: step size $\alpha \in (0,1]$
	- Initialize Q(s,a), for all $s \in S^+, a \in A(s)$ arbitrarily except that A(terminal, .) = 0
	- Loop for each episode:
		- Initialize S
		- Choose A from S using policy derived from Q (e.g., $\epsilon$-greedy)
		- Loop for each step of episode:
			- Take action A, observe $R, S\prime$
			- Choose $A\prime$ from $S\prime$ using policy derived from Q (e.g., $\epsilon$-greedy)
			- $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S\prime, A\prime) - Q(S, A)]$
			- $S \leftarrow S\prime, A \leftarrow A\prime$
		- until S is terminal
	  
- Note:
	- for Sarsa, $\pi$ is derived from current Q, there is no need for predefining a policy (comparing to TD(0) in section 1)

	- Sarsa converges with probability 1 to an optimal policy and action-value function as long as all state–action pairs are visited an infinite number of times and the policy converges in the limit to the greedy policy
	

## Q-learning: Off-policy TD Control

- update rule: 
	$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$

- Q-learning (off-policy TD control) for estimating $\pi \approx \pi_{\star}$
	- Algorithm parameter: step size $\alpha \in (0,1], \epsilon > 0$
	- Initialize Q(s,a), for all $s \in S^+, a \in A(s)$
	- Loop for each episode:
		- Initialize S
		- Loop for each step of episode:
			- Choose A from S using policy derived from Q (e.g., $\epsilon$-greedy)
			- Take action A, observe $R, S\prime$
			- $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma max_a Q(S\prime, a) - Q(S, A)]$
			- $S \leftarrow S\prime$
		- until S is terminal

- differ from Sarsa, at target state $S\prime$, Q-learning choose the action that could maximize the $Q(S\prime, a)$ directly, but not according to a policy derived from Q (although, the derived policy from Q can also be the greedy policy, in this case Sarsa and Q-learning are identical)

- why is Q-learning off-policy? 

	- consider the derived policy from current Q as the *behaviour policy*, which can be e.g., $\epsilon$-greedy. but the *target policy* for Q-learning is actually the greedy policy according to the *max* term in the update rule from above (actions are chosen according to epsilon-greedy, updates are made according to greedy policy).  

## Expected Sarsa

- update rule: 

	$$
		Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma E_{\pi}[Q(S_{t+1}, A_{t+1})|S_{t+1})] - Q(S_t, A_t)] \\
		\leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \sum_{a}\pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)]
	$$

- the fun part about Expected Sarsa is that it can be both on- and off-policy:
	- when the target policy (in the target part in the update rule) is a greedy policy (means it distributes 0% to all other actions than the greedy one) and the behaviour policy is not, it becomes Q-learning, meaning Q-learning is actually a special case of Expected Sarsa.
	- the target policy can also be an exploratory policy that is totally different than the behaciour policy (which does not make sense. but it is allowed)


## NOTES

- imagine a environment in which rewards except for reaching the terminal state are all zero, and discount factor is 1. If every state (at the beginning usually) has the same state value, then for the fisrt episode, TD methods only update the single one and only state one step before the termination. While MC updates all states. (See lecture video week 2, 2.2)


