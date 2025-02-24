# Chapter 11. Modern Policy Gradient Methods

The algorithms introduced in this chapter is not included in Sutton's book, yet each one of them marks a great milestone in the journey of RL, and still inspiring further development in the field.  I use the term "modern" to separate these algorithms from these in Chapter 10, yet during learning, you will be surprised to find that all these policy approximation methods share similarities to a significant level. 

I have kept the notations from original works of each algorithm.

## 11.1 The Concept of Advantage

- **Definition**: Advantage (function) is a concept that compares the Q-value of an action to the Value function of a state, it measures how much better or worse the action $a_t​$ is compared to the average behavior of the agent in that state. It is defined as:

    $$
    A(s_t, a_t) \ \dot= \ q(s_t, a_t) - v(s_t)
    $$

    - Intuition: Advantage quantifies how much better or worse the action $a_t$ is than the expected return $E_\pi[G_t \vert s_t]$ of being in state $s_t$.

        - $A(s_t, a_t) > 0$: the action $a_t$ is considered better than the average action in state $s_t$.

        - $A(s_t, a_t) < 0$: the action $a_t$ is worse than the average action in state $s_t$.

        - $A(s_t, a_t) = 0$: the action is neither better nor worse than the expected value.

    ```{note}
    Advantage function $A(s_t, a_t)$ is often written as $A_t$ for simplicity
    ```

- **Justification of Existence**:

    The gradient estimate in Policy Gradient methods is often noisy, as we have seen in [Chapter 10](../Contents/10_policy_gradient_methods.md) from REINFORCE algorithm, because the estimate is based on the observed returns, which can be stochastic and vary greatly.

    Advantage reduces the variance of the gradient estimates and leads to more stable updates by simply replacing the return term in gradient estimate as follows:

    $$
    \theta_{t+1} \doteq \theta_t + \alpha \ A_t \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)}
    $$

    ```{note}
    - For REINFORCE with baseline method, when an approximate state value function $\hat{v}(S_t, \boldsymbol{w})$ is used as the baseline, the quantity $G_t - b(S_t)$ can be seen as an estimate of the advantage $A_t$, because $G_t$ is an estimate of $q(s_t, a_t)$ by definition.

    - In practice, one may encounter many cases where REINFORCE with baseline is regarded as actor-critic method. This tutorial abides by the rule in Sutton's book and treat only those, whose estimated critic model is used for boostrapping, as AC methods. 
    ```

- **Estimation of Advantage**:

    - **By definitioin**: $A(s_t, a_t)$ can be estimated by simply subtracting state value from action value as by definition:

        $$
        \begin{align*}
        \hat{A}(s_t, a_t) \ &\dot= \ q(s_t, a_t) - v(s_t) \\
        &= G_t - v(s_t) 
        \end{align*}
        $$
    
        Note that the first estimation is less often seen in practice since it requires learning of two critic models, and therefore ofter replaced by the second representation. 

    - **By TD error**: TD error can be a pratical approximate estimation for advantage. Recall that in Chapter 3 [section 3.3](../Contents/3_markov_decision_process.md#33-policies-and-value-functions), we have given that 

        $$q(s,a) \dot= \sum_{s', r}p(s', r|s, a) [r + \gamma v_{\pi}(s')]$$
    
        This means that $r + \gamma v_{\pi}(s')$ is a single realization when starting from (s,a) and the state $s'$ is actually reached.

        Therefore, we can replace $q(s_t,a_t)$ by $r_{t+1} + \gamma v(s_{t+1})$ and estimated the advantage as:

        $$\hat{A}(s_t, a_t) \ = \ r_{t+1} + \gamma v(s_{t+1}) - v(s_{t})$$

    - **Genralized Advantage Estimation (GAE)**: 

        We have mentioned above that advantage can be estimated by $G_t - v(s_t)$ since the return at time step $t$ is an estimate for the respective action-value. We now consider a different perspective for estimating the return. In $n$-step TD method (which is not included in this tutorial), return is defined as 

        $$
        \begin{align*}
        G_{t:t+n} \ &\dot= \ R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}) \\
        &=  \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n v(s_{t+n})
        \end{align*}
        $$

        This $n$-step return $G_{t:t+n}$ provides a lower-variance estimator than the Monte Carlo return at the cost of introducing some bias. To bring this idea further, one can calculate an exponentially-weighted average of $n$-step returns with a decay parameter $\lambda$ and introduce the $\lambda$-return:

        $$
        \begin{align*}
        G^{\lambda}_t \ &\dot= \ (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n} \\
        &= (1 - \lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t
        \end{align*}
        $$

        The second equation above is derived when we assume all rewards after time step $T$ are 0, such as $G_{t:t+n} = G_{t:T}$ for all $n \geq T-t$. Compared to $n$-step return, where the balance between bias and variance is achieved in a discrete setting (by choosing $n$) $\lambda$-return operates with a continuous spectrum and offers a more smooth trade-off. Empirically, the $\lambda$-return has been shown to produce better performance than simply using an $n$-step return.

        Now, GAE is computed by replacing the MC return with $\lambda$-return as:

        $$
        \begin{align*}
        \hat{A}^{GAE}(s_t, a_t) \ &\dot= \ G^{\lambda}_t - v(s_t)   \tag{1} \\
        &= \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}   \tag{2} \\
        &= \delta_{t} + (\gamma \lambda)\delta_{t+1} + (\gamma \lambda)^2\delta_{t+2} + \ ...   \tag{3}  \\ 
        &= \delta_{t} + (\gamma \lambda)\hat{A}^{GAE}(s_{t+1}, a_{t+1}) \tag{4} 
        \end{align*}
        $$

        The derivation from $(1)$ to $(2)$ is given in the optinal section below. Equation $(3)$ implements GAE in truncated form that gives a weighted sum of TD errors, where the future TD errors are discounted and smoothed by the $\lambda$-parameter. In practice, GAE at each time step is often computed using equation $(4)$ in a backward manner.

        ```{note}
        - Intuition on $n$-step return $G_{t:t+n}$: when $n=1$, it results in the 1-step return $R_{t+1} + \gamma v(s_{t+1})$, which is exactly the update target in TD methods (high bias, low variance). As $n$ goes to infinity, it recovers the original Monte Carlo return (unbiased, high variance). Therefore, $n$ acts as as trade-off between bias and variance for the value estimator. 

        - Intuition on $\lambda$ return: simliar to the $n$-step return, $\lambda=0$ reduces to the single-step return (1-step TD target), and $\lambda=1$ recovers the Monte Carlo return.

        - Intuition on GAE: still, when $\lambda=0$, GAE reduces to 1-step TD, and $\lambda=1$ recovers the Monte Carlo estimation.
        ```
        
        ```{todo}
        Add these to references
        ```
        https://arxiv.org/pdf/1804.02717

        https://zhuanlan.zhihu.com/p/549145459

        https://lilianweng.github.io/posts/2018-02-19-rl-overview/

- **$\star$ Derivation for GAE Representation**:

    ```{todo}
    Fill this section
    ```

## 11.2 Asynchronous Advantage Actor Critic (A3C)

```{todo}
Polish this section
```

- Overview: A3C is an actor-critic algorithm that uses multiple workers (parallel agents) to explore the environment and update the shared model asynchronously, which leads to better performance.


- Gradient Update:

    $$
    \nabla_{\theta'} \log \pi(a_t | s_t; \theta') \left( R_t - V(s_t; \theta_v) \right) + \beta \nabla_{\theta'} H(\pi(s_t; \theta'))
    $$


- Properties of A3C

    - For the firs time of all RL algorithm, A3C moves computation to a single machine with multiple CPU threads, instead of using separate machines and multiple GPUs, and also achieves better performance.

    - one can explicitly use different exploration policies in each actor-learner to maximize this diversity. By running different exploration policies in different threads, the overall changes being made to the parameters by multiple actor-learners applying online updates in parallel are likely to be less correlated in time than a single agent applying online updates. Hence, we do not use a replay memory and thereby stabilized the training.

    - reduction in training time that is roughly linear in the number of parallel actor-learners

    - since we no longer rely on experience replay for stabilizing learning we are able to use on-policy reinforcement learning methods such as Sarsa and actor-critic to train neural networks in a stable way

## 11.3 Advantage Actor Critic (A2C)

```{todo}
Complete this section
```

- Motivation: A2C is a variant of A3C, https://openai.com/index/openai-baselines-acktr-a2c/


- Gradient Update:

Note that in practice, the update can be grouped by any batch, and this grouping average can happen on both level of time steps and trajectories (batch of certain amount of time steps / trajectory). It is highly recommended that you check out the pseudocode given by this [OpenAI spinning up page](https://spinningup.openai.com/en/latest/algorithms/vpg.html#pseudocode) to understand how it works in practice.


## 11.4 Proximal Policy Optimization (PPO)

- **Background**: Among all modern RL algorithms, PPO stands out as one of the most influential framework. It is adapted on the basis of Trust Region Policy Optimization (TRPO), while being significantly simpler to implement, and empirically, it seems to perform at least as well as TRPO. We cover PPO's background only by giving a brief description of TRPO.

    With definition of advantage, the gradient estimator (also the maximization objective) of all policy gradient methods can be written as:

    $$
    \hat{g} = \hat{\mathbb{E}}_t \left[ \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \hat{A}_t \right]
    $$

    TRPO methods takes the following maximization objective:

    $$
    \begin{align*}
    &\text{maximize}_{\theta} \ \hat{\mathbb{E}}_t \left[ \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] \\
    &\text{subject to} \quad \hat{\mathbb{E}}_t \left[ \text{KL}\left[\pi_{\theta{\text{old}}}(\cdot \mid s_t), \pi_{\theta}(\cdot \mid s_t)\right] \right] \leq \delta,
    \end{align*}
    $$

    where $\theta_{old}$ denotes the parameter of the policy before the update. 

    ```{note}
    The theory justifying TRPO actually suggests using a penalty instead of a constraint, i.e., solving the unconstrained optimization problem maximize over $\theta$:

    $$
    \text{maximize}_{\theta} \; \hat{\mathbb{E}}_t \left[ \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t - \beta \, \text{KL}\left[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_{\theta}(\cdot \mid s_t)\right] \right]
    $$
    
    TRPO uses a hard constraint rather than a penalty because it is hard to choose a single value of the coefficient $\beta$ that performs well across different problems—or even within a single problem.
    ```

- **Objective of Proximal Policy Optimization**

    We have seen that TRPO maximizes a "surrogate" objective as follows:

    $$
    L^{CPI}(\theta) = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t \right],
    $$

    where $r_t(\theta)$ denotes the probability ratio $\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ ($r(\theta_{\text{old}}) = 1$). And $CPI$ refers to conservative policy iteration. Instead of using KL-divergence, PPO modifies this objective and constrain the policy changes that move $r_t(\theta)$ too much away from 1, by introducing the $\textit{Clipped Surrogate Objective}$. To be precise, there are three different components in PPO's optimization objective:

    - **Clipped Surrogate Objective**: penalizes changes to the policy that move $r_t(θ)$ away from 1 by clipping $r_t(θ)$ into a given rage:

        $$
        L^{CLIP}(\theta) \ \dot= \ \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right],
        $$

        where epsilon is a hyperparameter for controling the clipping range.

    - **Value Function Loss**: deonted by $L_t^{VF}(\theta)$, the value function loss is a squared-error loss represented as follows:

        $$
        L_t^{VF}(\theta) \dot= (V_\theta(s_t) - V_t^{targ})^2
        $$

    - **Entropy Bonus**: PPO also adds an entropy bonus $S[\pi_\theta](s_t)$ to the objective, given as:

        $$
        S[\pi_\theta](s_t) =− \sum_a \pi(a_t \vert s_t)log\pi(a_t \vert s_t)
        $$

        Since PPO uses gradient ascent, adding this entropy term encourages the agent to maintain a level of uncertainty (higher entropy suggests higher uncertainty) about the best action, leading it to explore a broader range of actions during training.
        
        If the policy becomes too deterministic (i.e., it consistently selects the same actions), the entropy will decrease, and the agent will be incentivized to explore more diverse actions, which is of great importance in early stage of training.


    - **Final PPO Objective**: Assume we are using a neural network architecture that shares parameters between the policy and value function, we must use a loss function that combines the policy surrogate and a value function error term. With augmentation of entropy bonus to ensure sufficient exploration, the final PPO objective is defined as follows:

        $$
        L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right],
        $$

        where $c_1$, $c_2$ are **positive** coefficients. Note that $L_t^{VF}(\theta)$ is a negative term, such that the sqaured-error loss is minimized when the overall objective is maximized.

- **PPO algorithm**

    - **Pseudocode**[reference]

        <div style="display: flex; justify-content: center;">
        <img src="../_static/img/chapter11/algo_ppo.png" alt="Algorithm: PPO in AC style" style="width: 100%;;">
        </div>

        In each iteration, each of $N$ (parallel) actors collect $T$ timesteps of data. Then we construct the surrogate loss on these $N \times T$ timesteps of data, and optimize it with minibatch SGD for $K$ epochs.

    - **Algorithm Details**

        - **Estimation of Advantage**: As you may already notice in the pseudocode, PPO uses a truncated version of generalized advantage estimation. 

            One style of policy gradient implementation, popularized in A3C paper[reference] runs the policy for $T$ timesteps (where $T$ is much less than the episode length). It requires an advantage estimator that does not look beyond timestep $T$:

            $$
            \hat{A}_t = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{T - t + 1} r_{T - 1} + \gamma^{T - t} V(s_T).
            $$

            Generalizing this choice, PPO uses a truncated version of generalized advantage estimation, which reduces to above when $\lambda = 1$:

            $$
            \begin{align*}
            \hat{A}_t &= \delta_t + (\gamma \lambda) \delta_{t+1} + \cdots + (\gamma \lambda)^{T - t + 1} \delta_{T - 1} \\
            &= \delta_t + \gamma\lambda\hat{A}_{t+1},
            \end{align*}
            $$

        - **Gradient Ascent in Practice**

            In practice, updates for actor and critic model are separate. The gradient (ascent) estimator for policy model is as follows:

            $$
            \hat{g}_{actor} = \frac{1}{NT}\sum_{n=0}^{N} \sum_{t=0}^{T} [\min ( r_t(\theta) \hat{A}_t, \ \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) + S[\pi_\theta](s_0)]
            $$

            While #gGradient (descent) estimator for value function is:

            $$
            \hat{g}_{critic} = \frac{1}{NT}\sum_{n=0}^{N} \sum_{t=0}^{T}L_t^{VF}(\theta),
            $$

            for whichever form of loss used for the value function.


- **Properties of PPO**

    - **Advantages**:

        - Stable Learning: PPO uses a clipped objective to limit how much the policy can change at each update, preventing large, unstable updates that might destabilize training.

        - Implementation Simplicity: Compared to more complex methods like TRPO, PPO is relatively simple to implement while still achieving competitive performance across a range of tasks.

    - **Disdvantages**:

        - Hyperparameter Sensitivity: PPO requires careful tuning of its hyperparameters (like the clipping parameter and learning rate) to achieve optimal performance. Which requires experiments and understanding of the specific problem setting.

        - Computational Overhead: PPO can be computationally expensive, particularly when running in a distributed setting or with large-scale environments, as it relies on collecting batches of data before each update and updating of both actor and critic models.

## 11.5 Group Relative Policy Optimization (GRPO)

- **Background**: different from all other algorithms introduced in former sections, GRPO was proposed targeting a specific kind of RL task - the post-training of Large Language Models (LLMs). At this point, its effectiveness applied on other types of tasks is less studied, we introduce GRPO mainly in the context of LLM training.

    Before GRPO, the de-facto algorithm for LLM training is PPO with one particular adaptation - adding a KL penalty from a reference model in the reward to prevent the policy deviating too much from the reference during update[reference from InstructGPT]. The reward is then computed as
    
    $$
    r_t = r_\phi(s_t, a_t) - \beta log \frac{\pi_\theta(a_t\vert s_t)}{\pi_{ref}(a_t\vert s_t)}
    $$

    with $r_\phi$ being the reward model and the reference model $\pi_{ref}$ being a frozen copy (untrainable) of the policy. 

    In context of LLM training, critic model is usually a model that has comparable size with the policy model (with some parameter shared). To lower computational burden, GRPO obviates the need for additional value function approximation, and instead **uses the average reward of multiple sampled outputs as the baseline** (explained in advantage estimation). The process is shown in the figure below.

    <div style="display: flex; justify-content: center;">
    <img src="../_static/img/chapter11/comp_ppo_grpo.png" alt="Comparison between PPO and GRPO" style="width: 100%;;">
    </div>

    In short, GRPO does not require the critic model and takes a group average reward as the baseline, advantage is also estimated using this grouping method (introduced later).

    ```{note}
    Notations in context of LLM training:

    - $q$: the input question given to an LLM.

    - $o_i$: the observation (state) $i$ at a certain time step $t$, usually is the concatenation of $q$ with all generated content till $t$.

    - $r_i$: reward given by a reward model $\phi$ based on $o_i$.

    - $A_i$: advantage computed at time step $t$.
    ```

- **Optimization Objective**

    - **Overall Objective**

        $$
        \begin{align*}
        \mathcal{J}_{GRPO}(\theta) &= \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \left[\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \frac{\pi_{\theta}(o_{i,t}|q, o_i,<t)}{\pi_{\theta_{old}}(o_{i,t}|q, o_i,<t)} \hat{A}_{i,t} \, \text{clip} \left( \frac{\pi_{\theta}(o_{i,t}|q, o_i,<t)}{\pi_{\theta_{old}}(o_{i,t}|q, o_i,<t)} \right) (1 - \epsilon, 1 + \epsilon) \hat{A}_{i,t} \right) - \beta D_{KL}[\pi_{\theta_{old}} || \pi_{ref}] \right] \\
        &\text{with} \ \mathbb{D}_{KL} (\pi_{\theta} || \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_i|q)}{\pi_{\theta}(o_i|q)} - \log \frac{\pi_{\text{ref}}(o_i|q)}{\pi_{\theta}(o_i|q)} - 1
        \end{align*}
        $$

        Same as PPO, GRPO also uses a clipped objective, yet differently, no entropy bonus is added in the objective.  

        Also note that, instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the maximization objective, avoiding complicating the calculation of $A_{i,t}$.

        ```{note}
        Note that GRPO estimate the KL divergence with an unbiased estimator [reference to Schulmanns blog](http://joschu.net/blog/kl-approx.html) different from the KL estimation described at the beginning of this section. 
        ```
    
    - **Estimation of Advantage**

        Since GRPO requires no value function, it estimates advantage $A_{i,t}$ by calculating based on relative rewards of the outputs inside each group only as follows:

        $$
        A_{i,t} = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})}
        $$


- **GRPO Algorithm**

    <div style="display: flex; justify-content: center;">
    <img src="../_static/img/chapter11/algo_grpo.png" alt="Algorithm: GRPO" style="width: 100%;;">
    </div>

    The above algorithm [reference to GRPO] uses an iterative approach, except for the usual computations of all components required, there are two main things to be noted:

    - **Reward Model Retraining**: In iterative GRPO, the authors generate new training sets for the reward model based on the sampling results from the policy model and continually train the old reward model using a **replay mechanism** that incorporates 10% of historical data. Then, they set the reference model as the policy model, and continually train the policy model with the new reward model.

    - **Token Generation as Time Step**: In context of LLM training, this algorithm is performed on the token level.

- **Properties of GRPO**

    - **Advantages**  

        - Reduced computational burden: GRPO eliminates the need for a critic model, reducing memory usage and computational costs through group-based sampling.

        - Efficient advantage estimation: By comparing multiple outputs for the same input, GRPO provides stable and efficient advantage estimation.

        - Conservative policy updates: A KL penalty in GRPO’s objective function ensures more stable and conservative policy updates.

    - **Disadvantages**

        - Complex reward design: GRPO requires careful reward function design to reflect output quality, which can be challenging.

        - Dependency on group size: The group size affects advantage estimation accuracy; too small a group may lack sufficient information, while a very large group increases computational overhead.

## 11.6 Summary

