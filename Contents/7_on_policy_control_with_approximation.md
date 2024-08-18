# Chapter 10. On-policy Control with Approximation

In this chapter we return to the control problem (policy improvement) and estimate action value function $\hat{q}(s, a, \boldsymbol{w}) \approx q_\star(s,a)$, and still restrict our attention to the on-policy case.  

## 10.1 Episodic semi-gradient control

- Gradient descent update for action-value prediction:
    - update rule
        $$
        \boldsymbol{w_{t+1}} \dot= \boldsymbol{w_t} + \alpha [U_t - \hat{q}(S_t, A_t, \boldsymbol{w_t})]\nabla\hat{q}(S_t, A_t, \boldsymbol{w_t})
        $$
- Episodic semi-gradient one-step Sarsa

    - update rule
        $$
        \boldsymbol{w_{t+1}} \dot= \boldsymbol{w_t} + \alpha [R_{t+1} + \gamma\hat{q}(S_{t+1}, A_{t+1}, \boldsymbol{w_t}) - \hat{q}(S_t, A_t, \boldsymbol{w_t})]\nabla\hat{q}(S_t, A_t, \boldsymbol{w_t})
        $$

        - note that the update target at time step t+1 is given by the action value function with weights from time step t.
    
    - Algorithm:
        - Input: a differentiable action-value function parameterization $\hat{q}: S \times A \times \mathbb{R}^d \rightarrow \mathbb{R}$
        - Algorithm parameter: step size $\alpha$, small $\epsilon > 0 $
        - Initialize value function weights $\boldsymbol{w} \in \mathbb{R}^d arbitrarily$
        - Loop for each episode:
            - $S, A \leftarrow$ initial state and action of episode
            - Loop for each step of the episode:
                - Take action $A$ according to the policy, observe $S\prime, R$
                - If $S\prime$ is terminal:
                    - update weights:
                    $$
                    \boldsymbol{w} \dot= \boldsymbol{w} + \alpha [R - \hat{q}(S, A, \boldsymbol{w})]\nabla\hat{q}(S, A, \boldsymbol{w})
                    $$
                    - Go to the next episode
                - Choose $A\prime$ based on the function $\hat{q}(S, a, \boldsymbol{w})$ and the policy (e.g., $\epsilon-greedy$)
                - Update weights: 
                    $$
                    \boldsymbol{w} \dot= \boldsymbol{w} + \alpha [R + \gamma\hat{q}(S\prime, A\prime, \boldsymbol{w}) - \hat{q}(S, A, \boldsymbol{w})]\nabla\hat{q}(S, A, \boldsymbol{w})
                    $$
                - Updata state and action:
                $$
                S \leftarrow S\prime \\
                A \leftarrow A\prime
                $$

- <span style="color:red;">Approximate action-value function with features:</span>

    - Representation:
        $\hat{q}(s,a,\boldsymbol{w}) \dot= \boldsymbol{w}^T\boldsymbol{x}(s,a)$
        - note that the features are contructed with **both state $s$ and action $a$**

    - Construction of features:
        - Linear methods:
            - Stack method: weights for each action are stacked on top of each other
            - both state and action as input: tile coding mentioned in the lecture video 3:09
        - Neural network:
            - only states as input and multiple outputs
            - both states and actions as input and single output (generalizing over actions) 

- Semi-gradient expected Sarsa

- Semi-gradient Q-learning

- Methods for improving exploration under function approximation
    - lecture video

## 10.3 Average Reward: A new problem setting for continuing tasks