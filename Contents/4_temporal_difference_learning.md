## TD prediction

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
	$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$

- Q-learning (off-policy TD control) for estimating $\pi \approx \pi_{\star}$
	- Algorithm parameter: step size $\alpha \in (0,1], \epsilon > 0$
	- Initialize Q(s,a), for all $s \in S^+, a \in A(s)$
