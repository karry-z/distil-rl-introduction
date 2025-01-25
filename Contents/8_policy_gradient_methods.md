# Chapter 13. Policy Gradient Methods

So far in this book almost all the methods have been action-value methods, i.e., they try to learn the values of actions and then selected actions based on their estimated action values. We now consider methods that instead learn a parameterized policy that can select actions without consulting a value function. Note that a value function may still be used to $\textit{learn}$ the policy parameter (denoted by $\theta \in \mathbb{R}^{d'}$), but is not required for action selection.

This chapter considers methods for learning the policy parameter (the policy is represented as $\pi(a \mid s, \theta) = \Pr\{ A_t = a \mid S_t = s, \theta_t = \theta \}$) based on the gradient of some scalar performance measure $J(\theta)$, which we aim to maximize. Therefore the update of policy parameter follows gradient ascent:

$$\theta_{t+1} = \theta_t + \alpha \nabla \widehat{J}(\theta_t)$$

where $\nabla \widehat{J}(\theta_t)$ is a stohastic estimate whose expectation approximates the gradient of the performance measure with respect to its argument $\theta_t$. All methods that follow this general schema are called $\textit{policy gradient methods}$.

Among  $\textit{policy gradient methods}$, methods that learn approximations to both policy and value functions are often called $\textit{actor–critic methods}$, where $\textit{'actor'}$ is a reference to the learned policy, and $\textit{'critic'}$ refers to the learned value function.

## 10.1 Policy Approximation and its Advantages

- Setup: In policy gradient methods, the policy can be parameterized in any way, as long as $\pi(a|s, \theta)$ is differentiable with respect to its parameters, In practice, to ensure exploration we generally require that the policy never becomes deterministic (i.e., that $\pi(a|s, \theta) \in (0, 1)$, for all $s, a, \theta$)

- Approximation example for discrete and small (not too large) action space:

    - Parameterization of the policy: in this setting, we first parameterize numerical $\textit{action preferences}$ $h(s, a, \theta) \in R$ for each state–action pair. The actions with the highest preferences in each state are given the highest probabilities of being selected, according to an e.g., exponential soft-max distribution:

        $$\pi(a|s,\theta) \ \dot= \ \frac{e^{h(s,a,\theta)}}{\sum_b e^{h(s,b,\theta)}}$$

        We call this kind of policy parameterization soft-max in action preferences.

    - Parameterization of the state-action pair: The action preferences $h(s, a, \theta) \in R$ themselves can be parameterized arbitrarily. For exapmle, by:
        - a deep artificial network (ANN), where $\theta$ is the vector of all the connection weights of the network (as in the AlphaGo system, readers of interest can refer to the book section 16.6), or
        - a linear system in features as 
            
            $$h(s, a, \theta) = \theta^\top x(s, a)$$

            using feature vectors $x(s, a) \in \mathbb{R}^{d'}$ constructed by any of the methods described in Chapter 9.


- Advantages of parameterizing policies

    - **Allowing Determinism**: Unlike the traditional epsilon-greedy approach, which caps exploration, parameterized policies can start stochastic and naturally converge to a greedy policy, i.e., if the optimal policy is deterministic, then the preferences of the optimal actions will be driven infinitely higher than all suboptimal actions (if permitted by the parameterization). This avoids the need for external decisions about when exploration is complete.

    - **Allowing Stochasticity**:  Parameterization of policies enables the selection of actions with arbitrary probabilities. In problems with significant function approximation, a deterministic policy might not always be feasible. A stochastic policy can often perform better, as demonstrated below in the corridor example, stochastic actions help the agent achieve higher returns.

- Example: Short corridor with switched actions

    <img src="../img/chapter13/short_corridor.png" alt="Example of short corridor with switched actions" style="width:70%;">

    - Setup: 
        - As shown in the image, there are three nonterminal states, the reward is 1 per step.
        - In the first state, left action causes no movement.
        - In the second state actions are reversed, right actions takes the agent to the left and left to the right.
    
    - Comparison between action-value method and policy approximation:
        - An action-value method with $\epsilon$-greedy action selection is forced to choose between just two policies. For example, if $\epsilon = 0.1$, then either left of right action gets the probability of $1 - \frac{\epsilon}{2} = 0.95$, and the other gets only $0.05$. These two $\epsilon$-greedy policies achieve a value (at the start state $S$) of less than $-44$ and $-82$.
        - Policy approximation can do significantly better since it learns a specific probability with which to select right (allowing more stochasticity). As shown in the image, the best probability of selecting the right action with policy approximation is about $0.59$, which achieves a value of about $-11.6$.

## 10.2 The Poliy Gradient Theorem

- Policy Gradient Objective: 
    
    When we parameterize our policy directly, we can use the ultimate goal of reinforcement learning directly as the learning objective, i.e., to learn a policy that obtains as much reward as possible in the long run. Recall that our three form of reward formulations are:

    - Episodic Setting: $G_t = \sum_{t=0}^{T} R_t$

    - Continuing Setting with Discounted Return: $G_t = \sum_{t=0}^{\infty} \gamma^t R_t$

    - Continuing Setting with Average Reward Formulation: $G_t = \sum_{t=0}^{\infty} R_t - r(\pi)$


    In this chapter, **we focus on the continuing setting with average reward as the objective**. The average reward for a policy $\pi$ is defined as: 

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

    This [optional lecture video](https://www.coursera.org/learn/prediction-control-function-approximation/lecture/Wv6wa/the-policy-gradient-theorem) (between 2:08 - 4:27) provides an intuition of what the term $\sum_a q_{\pi}(s, a) \nabla \pi(a | s, \theta)$ does. For a detailed derivation of the policy gradient theorem, please refer to the book chapter 13.2, page 325.


## 10.3 REINFORCE (with Baseline): Monte Carlo Policy Gradient

- REINFORCE

    - Derivation of REINFORCE's update rule:

        The strategy of stohastic gradient ascent requires a way to obtain samples such that the expectation of the sample gradient is proportional to the actual gradient of the performance measure, i.e., we need some way of sampling whose expectation equals or approximates the expression given by the policy gradient theorem. 

        Naturally, we can reformulate the policy gradient theorem as

        $$
            \begin{align*}
            \nabla J(\theta) &\propto \sum_s \mu(s) \sum_a q_{\pi}(s, a) \nabla \pi(a | s, \theta) \\

            &= \mathbb{E}_{\pi} \left[ \sum_a q_{\pi}(S_t, a) \nabla \pi(a | S_t, \theta) \right],
            \end{align*}
        $$
        
        and we can just stop here and instantiate the stochastic gradient-ascent algorithm as

        $$
            \theta_{t+1} \doteq \theta_t + \alpha \sum_a \hat{q}(S_t, a, \mathbf{w}) \nabla \pi(a | S_t, \theta),
        $$

        where $\hat{q}$ is some learned approximation to $q_\pi$. This update algorithm is called an $\textit{all-actions}$ method because its update involves all of the actions. The algorithm is promising and deserves further study, but our current interest is the classical REINFORCE algorithm, which continues the above transformation as follows:

        $$
            \begin{align*}
            \nabla J(\theta) &= \mathbb{E}_{\pi} \left[ \sum_a \pi(a|S_t, \theta) q_{\pi}(S_t, a) \frac{\nabla \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)} \right] \\

            &= \mathbb{E}_{\pi} \left[ q_{\pi}(S_t, A_t) \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)} \right]  \ \text{(replacing \( a \) by a sample \( A_t \sim \pi \))}\\

            &= \mathbb{E}_{\pi} \left[ G_t \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)} \right]  \  \text{(because \( \mathbb{E}_{\pi} [ G_t | S_t, A_t] = q_{\pi}(S_t, A_t) \))}
            \end{align*} \\
        $$

        The stochastic gradient-ascent update of REINFORCE can therefore be instantiated as 

        $$\theta_{t+1} \doteq \theta_t + \alpha \ G_t \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)}$$
    
    - Intuition on REINFORCE

        - The derivation: note that during derivation, we used a sample $A_t \sim \pi$ to replace the the expectation term $\sum_a \pi(a|S_t, \theta) q_{\pi}(S_t, a)$. This strategy shares similarity as we change from Monte Carlo methods to TD methods. Similarly, this replacement brings more bias yet lower the variance at the same time.

        - The final update form: the increment of REINFORCE is proportional to the product of a return $G_t$ and a vector (called the $\textit{eligibility vector}$) - the gradient of the probability of taking the action actually taken divided by the probability of taking that action. The latter may sound horrible when first hearing it, so let's shed some light on what this increment indicates:

            - The return $G_t$ in the incremental term causes the parameter to move most in the directions that favor actions that yield the highest return.

            - The vector $\frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)}$, on the other hand, is a typical form of what is called $\textit{relative rate of change}$. In this case, it indicates the direction in parameter space that most increases the probability of repeating the action $A_t$ on future visits to state $S_t$. 
            
                Moreover, the update is inversely proportional to the action probability, giving actions that are less frequently selected an advantag, i.e., encouraging exploration.
        
        - Why Monte Carlo: Note that REINFORCE uses the complete return $G_t$ from time $t$, which includes all future rewards up until the end of the episode. In this sense it is a Monte Carlo algorithm and is well defined **only for the episodic case**

    - Algorithm of REINFORCE: Monte-Carlo Policy-Gradient Control (episodic) for $\pi_{\star}$
        - Input: a differentiable policy parameterization $ \pi(a | s, \theta) $
        - Algorithm parameter: step size $\alpha > 0$
        - Initialize policy parameter $\theta \in \mathbb{R}^d$ (e.g., to $0$)
        - Loop forever (for each episode):
            - Generate an episode $S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T$, following $\pi(\cdot | \cdot, \theta)$
            - Loop for each step of the episode $t = 0, 1, \dots, T-1$:
                - Compute return (with $\gamma$ added for the general discounted case):
                $$
                G \leftarrow \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k
                $$
                - Update policy parameters:
                $$
                \theta \leftarrow \theta + \alpha \gamma^t G \nabla \ln \pi(A_t | S_t, \theta)
                $$


    - Performance of REINFORCE on the short-corridor example

        <img src="../img/chapter13/reinforce_performance.png" alt="Performance of REINFORCE on the short corridor example with different step sizes" style="width:80%;">

        - Results: as shown, with a good step size, the total reward per episode approaches the optimal value of the start state ($v_\star(s_0)$).

        - Properties of REINFORCE: for suffciently small $\alpha$, the improvement in expected performance is assured, and convergence to a local optimum under standard stochastic approximation conditions happens for decreasing $\alpha$. However, as a Monte Carlo method REINFORCE may be of high variance and thus produce slow learning.


- REINFORCE with Baseline

    - Derivation of REINFORCE with Baseline

        We now generalize the policy gradient theorem to include a comparison of the action value $q_{\pi}(s, a)$ to an arbitrary $baseline \ b(s)$

        $$
        \nabla J(\theta) \propto \sum_{s} \mu(s) \sum_{a} \left( q_{\pi}(s, a) - b(s) \right) \nabla \pi(a \mid s, \theta).
        $$

        The baseline can be any function, even a random variable, **as long as it does not vary with $a$**, and the equation remains valid because the subtracted quantity is zero:

        $$
        \begin{align*}
        \sum_{a} b(s) \nabla \pi(a \mid s, \theta) &= b(s) \nabla \sum_{a} \pi(a \mid s, \theta) \\
        &= b(s) \nabla 1 \\
        &= 0.
        \end{align*}
        $$

        Therefore, we now have a new update rule that includes a general baseline, which is a strict generalization of REINFORCE (since the baseline could be uniformly zero):

        $$
        \theta_{t+1} \doteq \theta_t + \alpha \ (G_t - b(S_t)) \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)}
        $$
    
    - Justification for adding the baseline

        - Lower the variance: In general, the baseline leaves the expected value of the update unchanged, but it can have a large effect on its variance. Adding a baseline can significantly reduce the variance (and thus speed the learning). 
        
        - Setting of the baseline: 
        
            For MDPs, the baseline should vary with state. In some states all actions have high values and we need a high baseline to differentiate the higher valued actions from the less highly valued ones; in other states all actions will have low values and a low baseline is appropriate.

            Therefore, a natural choice of the baseline is an estimate of the state value: $\hat{v}(S_t, \boldsymbol{w})$. Because REINFORCE is a Monte Carlo method, is it also natural to use a Monte Carlo method to learn the state-value weights $\boldsymbol{w}$. To this end, we give the algorithm of REINFORCE with Baseline as below.

    - Algorithm of REINFORCE with Baseline: Monte-Carlo Policy-Gradient Control (episodic) for $\pi_\theta \approx \pi_{\star}$

        - Input: a differentiable policy parameterization $\pi(a | s, \theta)$
        - Input: a differentiable state-value function parameterization $\hat{v}(s, \boldsymbol{w})$
        - Algorithm parameters: step sizes $\alpha^{\theta} > 0$, $\alpha^{w} > 0$
        - Initialize policy parameter $\theta \in \mathbb{R}^d$ and state-value weights $\mathbf{w} \in \mathbb{R}^d$ (e.g., to $0$)
        - Loop forever (for each episode):
            - Generate an episode $S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T$, following $\pi(\cdot | \cdot, \theta)$
            - Loop for each step of the episode $t = 0, 1, \dots, T-1$:
            - Compute return (with $\gamma$ added for the general discounted case):
                $$
                G \leftarrow \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k
                $$
            - Compute TD error (note that the TD error is computed with $G$):
                $$
                \delta \leftarrow G - \hat{v}(S_t, \mathbf{w})
                $$
            - Update state-value weights with semi-gradient method:
                $$
                \mathbf{w} \leftarrow \mathbf{w} + \alpha^w \delta \nabla \hat{v}(S_t, \mathbf{w})
                $$
            - Update policy parameters:
                $$
                \theta \leftarrow \theta + \alpha^{\theta} \gamma^t \delta \nabla \ln \pi(A_t | S_t, \theta)
                $$

    - Performance of REINFORCE with Baseline on the short-corridor example
        
        <img src="../img/chapter13/reinforce_baseline_performance.png" alt="Performance of REINFORCE with Baseline on the short corridor example compared to REINFORCE" style="width:80%;">

        Adding a baseline to REINFORCE can make it learn much faster. The step size used here for plain REINFORCE is that at which it performs best.
    
## 13.4 Actor–Critic Methods