# Chapter 3. Finite Markov Decision Processes

MDPs are a  formalization of sequential decision making, where actions influence both immediate rewards, and subsequent states, and thereby the future rewards. So it must consider the trade-off between the immediate reward and delayed reward. 

Recall that in bandit problems we estimated the value $q_{\star}(a)$ of each action $a$, in MDPs we estimate the value $q_{\star}(s, a)$ of each action a in each state $s$, or we estimate the value $v_{\star}(s)$ of each state given optimal action selections (Meaning of these notations will be explained later). 


## 1. Agent-Environment Interface

- Overview:

    <div style="display: flex; justify-content: center;">
    <img src="../img/chapter3/agent_env_interaction.png" alt="Agent Environment Interaction" style="width: 70%;">
    </div>


- Explanation:
    - Agent: the learner and decision maker
    - Environment: the thing the agent interacts with, comprising everything outside the agent.

    - At time step $t$, the agent receives some representation of the environment's state $S_t \in S$, selects on that basis an action $A_t \in A(s)$, as a consequence it then receives a numerical reward $R_{t+1} \in R \subset \mathbb{R}$, and finds itself in a new environment state $S_{t+1} \in S$.
    This interaction leads to a $\textit{trajectory}$: $s_0, a_0, r_1, s_1, a_1, r_2, ... , s_t, a_t, r_{t+1}$

- Dynamics of MDP: in a $\textit{finite MDP}$ - the sets of $S, A, R$ all have finite elements, so $S_t, R_t$ have well defined discrete probability distributions dependent only on the preceding state and action (Markov property).

    $$
        p(s\prime, r | s, a) \dot= Pr(S_{t+1}=s\prime, R_{t+1}=r | A_t=a, S_t=s) \\
        \text{with} \sum_{s\prime \in S} \sum_{ r\in R} p(s\prime, r | s, a) = 1, \text{for all} \ s \in S, a \in A(s)
    $$

- Derivation: with the dynamics of a MDP known, one can compute anything one might want to know about the envrionment:
    - state-transition probability:
        $$
            p(s\prime | s, a) \dot= Pr(S_{t+1}=s\prime | A_t=a, S_t=s) = \sum_{r \in R}p(s\prime, r | s, a)
        $$
    - expected reward for state-action pairs:
        $$
            r(s,a) = E[R_{t+1}|S_t=s, A_t=a] = \sum_{r \in R} r \times \sum_{s\prime \in S}p(s\prime, r|s, a) 
        $$
    - <span style="color:red;">expected reward for state-action-next_state triple</span>

- <span style="color:red;">Example:</span>

## 2. About Rewards and Returns

### 2.1 Goals and Rewards

In reinforcement learning, the purpose or goal of the agent is formalized in terms of a special scalar signal, called the $\textit{reward} \ (R_t \in R)$, passing from the environment to the agent.

The idea of maximizing the cumulative reward to allow the agent to show desirablt behaviour is based on the $\textit{reward hypothesis}$: that all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of the reward.

Note that the reward signal is your way of communicating to the robot what you want it to achieve, not how you want it achieved.

### 2.2 Returns and Episodes

- In general,we seek to maximize the $\textit{expected return}$ of a sequence of rewards: 

    - for episodic tasks:
        $$
        G_t \dot= R_{t+1} + R_{t+2} + ... + R_{T} 
        $$

    - for continuing tasks:
        $$
        \begin{align*}
        G_t \ &\dot= \ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... \\
        &= \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \\
        &= R_{t+1} + \gamma \times G_{t+1}
        \end{align*}
        $$

- Details on the two type of tasks:

    - Episodic tasks: episodes end in a special state called the $\textit{terminal state}$, followed by a reset to a standard starting state or to a sample from a standard distribution of starting states. 
    
        - The next episode begins independently of how the previous one ended
        - The episodes can all be considered to end in the same $\textit{terminal state}$
        - Notation $S^+$ is used to denote the set of all non-terminal states plus the terminal state.

    - Continuing tasks: in contrast, continuing tasks are those tasks in which the agent–environment interaction does not break naturally into identifiable episodes, but goes on continually without limit.

        - $\gamma \in (0,1)$ is called the $\textit{discout rate}$, and is used to represent the agent's preference between immediate and future reward. The more $\gamma$ approaches 1, the more "farsighted" the agent becomes.

        - Though $G_t$ is a sum of an infinite number of terms, it is still finite if the reward is nonzero and constant and $\gamma \in (0,1)$.

        - Special case for continuing tasks: if reward signal is +1 all the time, then:
            $$
            G_t = \sum_{k=0}^{\infty} \gamma^k = \frac{1}{1 - \gamma}
            $$

        - Notation $S$ is used to denote the set of all non-terminal states (when it is a continuing task).

### 2.3 Unified Notation for Episodic and Continuing Tasks

In practice, it turns out that when we discuss episodic tasks we almost never have to distinguish between different episodes.

The two types of tasks can be unified by considering episode termination to be the entering of a special $\textit{absorbing state}$ that transitions only to itself and that generates only rewards of zero.

<div style="display: flex; justify-content: center;">
<img src="../img/chapter3/absorbing_state.png" alt="Absorbing State" style="width: 70%;">
</div>

So the expected return of both episodic and continuing tasks can now be written as $G_t=\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$


## 3. Policies and Value Functions


### 3.1 Bellman Equations

- Policy: formally, a $\textit{policy}$ is a mapping from states to probabilities of selecting each possible action:

    $$
        \pi(a|s) = Pr(A_t=a|S_t=s)
    $$

- Value Function (of state $s$) under policy $\pi$: is the expected return when starting in $s$ and following $\pi$ thereafter:

    $$
        \begin{align*}
        v_{\pi}(s) \ &\dot= \ \mathbb{E}_{\pi}[G_t|S_t=s] \\
        &= \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1} | S_t=s\right] \\
        &= \colorbox{lightyellow}{$\sum_a \pi(a|s)q(s,a)$} \\
        &= \sum_a \pi(a|s) \sum_{s', r}p(s', r|s, a) [r + \gamma \mathbb{E}_{\pi}[G_{t+1}|S_{t+1}=s']] \\
        &= \colorbox{lightyellow}{$\sum_a \pi(a|s) \sum_{s', r}p(s', r|s, a) [r + \gamma v_{\pi}(s')]$}
        \end{align*}
    $$

    - States are the independent variables for the value function, i.e., for each input state, state-value function assigns a respective state value.

    - States-value functions are always defined by the policy, when changing the policy, the resulted state-value function will usually be different.

    - The last equation above is called $\textit{Bellman Equation}$ for $v_{\pi}$, which expresses a relationship between the value of a state and the values of its successor states. The bellman equation can be understood with help of the following backup diagram for $v_{\pi}$:

        <div style="display: flex; justify-content: center;">
        <img src="../img/chapter3/backup_diagram_v.png" alt="Backup diagram for v" style="width: 28%;">
        </div>

        The backup operations (from bottom to top) transfer value information back to a state from its successor states.

- Action-value function under policy $\pi$: the expected return starting from $s$, taking the action $a$, and thereafter following policy $\pi$:

    $$
        \begin{align*}
        q_{\pi}(s,a) \ &\dot= \ \mathbb{E}_{\pi}[G_t|S_t=s, A_t=a] \\
        &= \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1} | S_t=s, A_t=a\right] \\
        &= \colorbox{lightyellow}{$\sum_{s', r} p(s', r|s, a) (r + \gamma v(s'))$}\\
        &= \sum_{s', r} p(s', r|s, a) (r + \gamma \sum_{a'} \pi(a'|s')\mathbb{E}_{\pi}[G_{t+1}|S_{t+1}=s', A_{t+1}=a']) \\
        &= \colorbox{lightyellow}{$\sum_{s', r} p(s', r|s, a) [r+ \gamma \sum_{a'} \pi(a'|s') q(s', a')]$}
        \end{align*}
    $$

    - likewise, $q_{\pi}(s,a)$ is a function of state $s$ and action $a$, and is defined unique to the policy $\pi$.

    - The bellman equation for $q_{\pi}(s,a)$ can be understood with help of the following backup diagram:
        <div style="display: flex; justify-content: center;">
        <img src="../img/chapter3/backup_diagram_q.png" alt="Backup diagram for 1" style="width: 25%;">
        </div>


- <span style="color:red;">Example of Gridworld</span>


### 3.2 Bellman Optimality Equation

- Optimal Policy:

    - Better or Equal Policy: A policy $\pi$ is defined to be **better or equal** to another policy $\pi'$ if:
    $$ v_{\pi}(s) \ge v_{\pi'}(s) \quad \text{for all } s \in S $$

    - Optimal Policy ($\pi_{\star}$): An **optimal policy** $\pi_{\star}$ is a policy that is better or equal to any other policy. Formally:
        - $\pi_{\star}$ must exist.
        - There can be more than one optimal policy.

- Optimal value functions:

    - Optimal State-Value Function: The **optimal state-value function** is defined as:
    $$ v_{\star}(s) \doteq \max_{\pi} v_{\pi}(s) \text{ for all } s \in S$$

    - Optimal Action-Value Function: The **optimal action-value function** is defined as:
    $$ q_{\star}(s, a) \doteq \max_{\pi} q_{\pi}(s, a) \text{ for all } s \in S, a \in A(s) $$


- Bellman Optimality Equation
	- for $v_{\star}(s)$:

	$$
    \begin{align*}
	  v_{\star}(s) &= \colorbox{lightyellow}{$ \underset{a \in A(s)}{\max} q_{\star}(s,a)$} \\
      &= \underset{a}{\max} E_{\pi_{\star}}[R_{t+1} + \gamma G_{t+1} | S_t=s, A_t=a] \\
      &= \underset{a}{\max} E_{\pi_{\star}}[R_{t+1} + \gamma v_{\star}(s\prime) | S_t=s, A_t=a] \\
	  &= \colorbox{lightyellow}{$ \underset{a}{\max} \sum_{s\prime, r}p(s\prime,r|s,a) (r+\gamma v_{\star}(s\prime))$}
    \end{align*}
    $$

	- for $q_{\star}(s,a)$:

	$$
    \begin{align*}
	  q_{\star}(s,a) &= E_{\pi_{\star}}[R_{t+1} + \gamma v_{\star}(s\prime) | S_t=s, A_t=a] \\
      &= \colorbox{lightyellow}{$ \underset{s\prime, r}{\sum} p(s\prime, r|s,a) [r+\gamma v_{\star}(s\prime)] $}\\
	  &= \colorbox{lightyellow}{$ \underset{s\prime, r}{\sum} p(s\prime, r|s,a) [r+ \gamma \ \max_a q_{\star}(s\prime,a\prime)]$}
    \end{align*}
    $$

    - Similarly, the bellman optimality equation can be easily memorized with help of these two backup diagrams:
        <div style="display: flex; justify-content: center;">
        <img src="../img/chapter3/backup_diagam_optimality.png" alt="Backup diagram for $v_{\star}$ and $q_{\star}$" style="width: 70%;">
        </div>

    - **$v_{\star}(s) =  \underset{a \in A(s)}{\max} q_{\star}(s,a)$ is the key of deriving both bellmann optimality equations.**

- Notes:
    - The Bellman optimality equation is actually a system of equations, one for each state, so if there are n states, then there are n equations in n unknowns. If the dynamics p of the environment are known, then in principle one can solve this system of equations for $v_{\star}$

    - Optimal policy: any policy that is greedy with respect to the optimal state-value function $v_{\star}$ is an optimal policy, because $v_{\star}$ already takes into account the reward consequences of all possible future behavior.

    - In reality, optimal action-value function $q_{\star}$ is often more desirable, with it, decisions can be made without knowing the dynamics of the environment. ($v_{\star}$, on the other hand, can only be used for decision making when environment dynamics are known - possible successor states and their values are known)

- <span style="color:red;">last example of the Robot...</span>


- Final Words:

    Explicitly solving the Bellman optimality equation provides one route to finding an optimal policy, and thus to solving the reinforcement learning problem. However, this solution is rarely directly useful - it relies on at least three assumptions that are rarely true in practice:

    1) we accurately know the dynamics of the environment; 
    2) we have enough computational resources to complete the computation of the solution; and 
    3) the Markov property.

    Even in a simple tabular setting where 1. and 3. are met, the number of states could easily scale beyond any current suptercomputer's ability. Therefore, there are other reinforcement learning methods that can be understood as approximately solving the Bellman optimality equation, and will be introduced in the following chapters.




 

