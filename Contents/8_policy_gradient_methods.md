# Chapter 13. Policy Gradient Methods

So far in this book almost all the methods have been action-value methods, i.e., they try to learn the values of actions and then selected actions based on their estimated action values. We now consider methods that instead learn a parameterized policy that can select actions without consulting a value function. Note that a value function may still be used to $\textit{learn}$ the policy parameter (denoted by $\theta \in \mathbb{R}^{d'}$), but is not required for action selection.

This chapter considers methods for learning the policy parameter (the policy is represented as $\pi(a \mid s, \theta) = \Pr\{ A_t = a \mid S_t = s, \theta_t = \theta \}$) based on the gradient of some scalar performance measure $J(\theta)$, which we aim to maximize. Therefore the update of policy parameter follows gradient ascent:

$$\theta_{t+1} = \theta_t + \alpha \nabla \widehat{J}(\theta_t)$$

where $\nabla \widehat{J}(\theta_t)$ is a stohastic estimate whose expectation approximates the gradient of the performance measure with respect to its argument $\theta_t$. All methods that follow this general schema are called $\textit{policy gradient methods}$.

Among  $\textit{policy gradient methods}$, methods that learn approximations to both policy and value functions are often called $\textit{actor–critic methods}$, where $\textit{'actor'}$ is a reference to the learned policy, and $\textit{'critic'}$ refers to the learned value function.

## 10.1 Policy Approximation and its Advantages

- Setup: In policy gradient methods, the policy can be parameterized in any way, as long as $\pi(a|s, \theta)$ is differentiable with respect to its parameters, In practice, to ensure exploration we generally require that the policy never becomes deterministic (i.e., that $\pi(a|s, \theta) \in (0, 1)$, for all $s, a, \theta$)

- Approximation example for discrete and small (not too large) action space:

    - Parameterization of the policy: in this setting, we first parameterize numerical preferences $h(s, a, \theta) \in R$ for each state–action pair. The actions with the highest preferences in each state are given the highest probabilities of being selected, according to an e.g., exponential soft-max distribution:

        $$\pi(a|s,\theta) \ \dot= \ \frac{e^{h(s,a,\theta)}}{\sum_b e^{h(s,b,\theta)}}$$

        We call this kind of policy parameterization soft-max in action preferences.

    - Parameterization of the state-action pair: The action preferences $h(s, a, \theta) \in R$ themselves can be parameterized arbitrarily. For exapmle, by:
        - a deep artificial network (ANN), where $\theta$ is the vector of all the connection weights of the network (as in the AlphaGo system, readers of interest can refer to the book section 16.6), or
        - a linear system in features as 
            
            $$h(s, a, \theta) = \theta^\top x(s, a)$$

            using feature vectors $x(s, a) \in \mathbb{R}^{d'}$ constructed by any of the methods described in Chapter 9.


- Advantages of parameterizing policies

    - **Allowing Determinism**: Unlike the traditional epsilon-greedy approach, which caps exploration, parameterized policies can start stochastic and naturally converge to a greedy policy, i.e., if the optimal policy is deterministic, then the preferences of the optimal actions will be driven infinitely higher than all suboptimal actions (if permitted by the parameterization). This avoids the need for external decisions about when exploration is complete.

    - **Allowing Stochasticity**:  Parameterization of policies enables the selection of actions with arbitrary probabilities. In problems with significant function approximation, a deterministic policy might not always be feasible. A stochastic policy can often perform better, as demonstrated in the corridor example in this [optioinal lecture video](https://www.coursera.org/learn/prediction-control-function-approximation/lecture/2nWtQ/advantages-of-policy-parameterization), where stochastic actions help the agent avoid negative infinite returns and reach the terminal state.

- Example 13.1 Short corridor with switched actions

## 10.2 The Poliy Gradient Theorem

- Policy Gradient Objective: 
    
    When we parameterize our policy directly, we can use the ultimate goal of reinforcement learning directly as the learning objective, i.e., to learn a policy that obtains as much reward as possible in the long run. Recall that our three form of reward formulations are:

    - Episodic Setting: $G_t = \sum_{t=0}^{T} R_t$

    - Continuing Setting with Discounted Return: $G_t = \sum_{t=0}^{\infty} \gamma^t R_t$

    - Continuing Setting with Average Reward Formulation: $G_t = \sum_{t=0}^{\infty} R_t - r(\pi)$


    In this chapter, we focus on the continuing setting with average reward as the objective. The average reward for a policy $\pi$ is defined as: 

    $$
    \begin{align*}
    r(\pi) &= \sum_{s}\mu(s) v(s) \\
    &= \sum_{s}\mu(s) \sum_{a} \pi(a \vert s^\prime, \theta) q(s,a) \\
    &= \sum_{s}\mu(s) \sum_{a} \pi(a \vert s^\prime, \theta) \sum_{s^\prime, a} p(s^\prime, r \vert s, a) r
    \end{align*}
    $$

    Therefore, The goal is to find a policy that maximizes this average reward, so the gradient ascent update we introduced at the beginning of this chapter can be formulated as:

    $$
    \begin{align*}
    \theta_{t+1} &= \theta_t + \alpha \nabla \widehat{J}(\theta_t) \\
    &= \theta_t + \alpha \nabla r(\pi) \\
    &= \theta_t + \alpha \nabla \sum_{s}\mu(s) \sum_{a} \pi(a \vert s^\prime, \theta) \sum_{s^\prime, a} p(s^\prime, r \vert s, a) r
    \end{align*}
    $$ 

    However, Unlike value function approximation (where \mu(s) was fixed), here \mu(s) depends on the policy, which in return changes the distribution $\mu(s)$ when it gets updated. We need a update rule for parameterizing the policy model without depending on $\mu(s)$, and that is when the policy gradient theorem comes to the rescue.

- Policy Gradient Theorem: 
    
    The theorem provides an analytic expression for the gradient of performance (average reward) with respect to the policy parameter that does not involve the derivative of the state distribution, and it has proved:

    $$
    \nabla J(\theta) \propto \sum_s \mu(s) \sum_a q_{\pi}(s, a) \nabla \pi(a | s, \theta)
    $$

    The symbol $\propto$ here means "proportional to". In the episodic case, the constant of proportionality is the average length of an episode, and in the continuing case it is 1. The distribution $\mu$ hereis the on-policy distribution under $\pi$ as introduced in the last chapter.

    This [optional lecture video](https://www.coursera.org/learn/prediction-control-function-approximation/lecture/Wv6wa/the-policy-gradient-theorem) (between 2:08 - ) provides an intuition of what the term $\sum_a q_{\pi}(s, a) \nabla \pi(a | s, \theta)$ does. For a detailed derivation of the policy gradient theorem, please refer to the book chapter 13.2, page 325.


## 10.3 REINFORCE

$$
\begin{align*}
\nabla J(\theta) &= \mathbb{E}_{\pi} \left[ \sum_a \pi(a|S_t, \theta) q_{\pi}(S_t, a) \frac{\nabla \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)} \right] \\

&= \mathbb{E}_{\pi} \left[ q_{\pi}(S_t, A_t) \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)} \right] \\

&= \mathbb{E}_{\pi} \left[ G_t \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)} \right]
\end{align*} \\
$$

The second equation simplifies the first one by replacing the summation over actions $a$ with a specific action $A_t$, which is drawn according to the policy $\pi$