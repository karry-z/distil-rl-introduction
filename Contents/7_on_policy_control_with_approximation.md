# Chapter 10. On-policy Control with Approximation

In this chapter we return to the control problem (policy improvement) and estimate action value function $\hat{q}(s, a, \boldsymbol{w}) \approx q_\star(s,a)$, and still restrict our attention to the on-policy case.  

We now feature the semi-gradient Sarsa algorithm, the natural extension of semi-gradient TD(0) (last chapter) to action values and to on-policy control. In the episodic case, the extension is straightforward, but in the continuing case we have to take a few steps backward and re-examine how we have used discounting to define an optimal policy. We will talk about how we would give up discounting and switch to a new “average-reward” formulation of the control problem, with new “differential” value functions.

## 10.1 Episodic semi-gradient control

- How to compute action-value function (linear case)

    - Smilar to the last chapter in the linear case, the approximated action-value function is computed by 

        $$
        \hat{q}(s, a, \boldsymbol{w_t}) \dot= \boldsymbol{w}^{\intercal} \boldsymbol{x}(s, a)
        $$

        - The feature vector $\boldsymbol{x}(s, a)$ for action-dependent function approximation is constructed by stacking the features for each action. E.g.,  4 features that represent the state of the system and 3 possible actions lead to a feature vector with 12 elements.
        - For generalizing over actions (like over states in the last chapter), we can input both the state and the action into a neural network, which will have state-action pairs as input and produce a single output: the approximate action value for that specific state and action.

    - This [optional lecture video](https://www.coursera.org/learn/prediction-control-function-approximation/lecture/z9xQJ/episodic-sarsa-with-function-approximation) gives a vivid illustration of the above notes, i.e., about how the computation is done, and especially, how the feature vector is constructed.


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


- Optional Watching: [Satinder Singh on Intrinsic Rewards](https://www.coursera.org/learn/prediction-control-function-approximation/lecture/TKPHV/satinder-singh-on-intrinsic-rewards)