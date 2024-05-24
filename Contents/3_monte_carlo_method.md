# Chapter 5. Monte Carlo Methods

Monte Carlo methods are ways of solving the reinforcement learning problem based on **experience only**, i.e., averaging sample returns. To ensure that well-defined returns are available, here we define Monte Carlo methods only for **episodic tasks**.

Only on the completion of an episode are value estimates and policies changed. Monte Carlo methods can thus be incremental in an **episode-by-episode** sense, but not in a step-by-step (online) sense.

We adapt the idea of general policy iteration (GPI) and learn value functions from sample returns with the MDP. The value functions and corresponding policies still interact to attain optimality in essentially the same way (GPI).

## 5.1 Monte Carlo Prediction (Evaluation)

- MC methods:
    - define a $\textit{visit}$ to state s: an occurrence of state s in an episode. Of course, $s$ may be visited multiple times in the same episode

    - evaluation methods: 
        - **first-visit MC method** estimates $v_{\pi}(s)$ as the average of the returns following first visits to $s$.
        - **every-visit MC method** averages the returns following all visits to $s$.

- First-visit MC prediction, for estimating $V \approx v_{\pi}$
    - Algorithm: 
        - Input: a policy $\pi$ to be evaluated
        - Initialize:
            - $V(s) \in \mathbb{R}$ arbitrarily, for all $s \in S$
            - $Return(s) \leftarrow$ an empty list for all $s \in S$
        - Loop forever (for each episode):
            - Generate an episode following $\pi: S_0, A_0, R_1, S_1, A_1, ..., S_{T-1}, A_{T-1}, R_T$
            - $G_T \leftarrow 0$
            - $ \text{for } t \text{ in } \{T-1, T-2, ..., 0\}$:
                - $G_{t} \leftarrow \gamma G_{t+1} + R_{t+1}$ 
                - Append $G_{t}$ to $Returns(S_t)$
                - $V(S_t) \leftarrow average(Returns(S_t))$
    - Intuition: the realizations of return $G_t$ for each state is calculated backwards from $S_{T-1}$ to $S_0$, by the law of large number, the average of $G_t$ for each state will be the value of that state: $v(S_t) = E_{\pi}[G_t|S_t]$

- Notes:
    - An important fact about MC method is that the estimates for each state are **independent.** The estimate for one state does not build upon the estimate of any other state, as is the case in DP. In other words, Monte Carlo methods do not bootstrap as we defined it in the previous chapter.

    - For MC method, the computational expense of estimating the value of a single state is independent of the number of states. This can make Monte Carlo methods particularly attractive when one requires the value of only one or a subset of states.

- <span style="color:red;">Example:</span>

- <span style="color:red;"> First-visit MC prediction, for estimating $Q \approx q_{\pi}$ </span>
    - Reason: state values $v_{\pi}$ are only usable when we have the model of the environment. Since MC methods assume there is no model available, one of our primary goals in this case is to actually estimate $q_\star$.
    - Algorithm:

## 5.2 Monte Carlo Control

- General problems and two basic assumptions we rely on:
    - Problem of $\textit{maintaining exploration}$: in estimating $q_{\pi}$, many state–action pairs may never be visited. E.g., if the policy is deterministic, many actions at a state may not be taken.

        - **Assumption (1)** of $\textit{exploring starts}$: episodes start in a state-action pair, and every pair has a nonzero probability of being selected as the start. (So every state-action pair will be visited an infinite number of times in the limit of an infinite number of episodes.)

    - Problem of estimating $\hat{q}_{\pi}(S_t, A_t)$: by default, we used the law of large number and rely on the following assumption:

        - **Assumption (2)** of infinite number of episodes: policy evaluation can be done with infinite number of episodes (complete policy evaluation).

### 5.2.1 Monte Carlo Control removing Assumption (2)

- How to **remove Assumption (2)**:

    - To avoid infinite number of episodes nominally required for policy evaluation, we could give up trying to complete policy evaluation before returning to policy improvement. Value iteration can be seen as an extrem example of this idea.

    - For Monte Carlo policy iteration it is natural to alternate between evaluation and improvement on an **episode-by-episode** basis. After each episode, the observed returns are used for policy evaluation, and then the policy is improved at all the states visited in the episode.

- Monte Carlo ES (Exploring Starts), for estimating $\pi \approx \pi_{\star}$
    - Algorithm:
        - Initialize:
            - $\pi(s) \in A(s)$ (arbitrarily), for all $s \in S$
            - $Q(s,a) \in R$ (arbitrarily), for all $s \in S$
            - $Returns(s, a) \leftarrow$ empty list, for all $s \in S, a \in A(s)$
        - Loop forever (for each episode):
            - Choose $S_0 \in S, A_0 \in A(S_0)$ randomly so that all state-action pairs have probability $> 0$
            - Generate an episode from the chosen start following $\pi(s)$, as $S_0, A_0, R_1, S_1, ..., S_{T-1}, A_{T-1}, R_T$
            - $G_T \leftarrow 0$
            - Loop for each step of episode: $t=T-1, T-2, ..., 0$:
                - $G_t \leftarrow \gamma G_{t+1} + R_{t+1}$
                - $ \text{for } t \text{ in } \{T-1, T-2, ..., 0\}$: 
                    - Append $G_t$ to $Returns(S_t, A_t)$
                    - $Q(S_t, A_t) \leftarrow average(Returns(S_t, A_t))$
                    - $\pi(S_t) \leftarrow \arg \underset{a}{\max}(Q(S_t, A_t))$

    - Notes:
        - The essential technique of above algorithm is that after each update of $Q(S_t,A_t)$, the improvement (greedification) will be made directly.

- <span style="color:red;">Example of Blackjack:</span>

### 5.2.1 Monte Carlo Control removing both assumptions

- Some definitions:

    - On policy methods: attempt to evaluate or improve the policy that is used to make decisions. (e.g., MC with ES, dynamic programming etc.)
    - Off policy methods: evaluate or improve a policy different from that used to generate the data.


    - $\epsilon$-greedy policy: all non-greedy action are given the minimal probability of selection $\frac{\epsilon}{|A(s)|}$, the greedy action has the probability of $1 - \epsilon + \frac{\epsilon}{|A(s)|}$. Among $\epsilon$-soft policies, $\epsilon$-greedy policies are in some sense those that are closest to greedy.
    - $\epsilon$-soft policy: all actions have probability of $\pi(a|s)>\frac{\epsilon}{|A(s)|}$ for all states.

On-Policy first-visit Monte Carlo Control (with $\epsilon$-soft policy), for estimating $\pi \approx \pi_{\star}$

- Algorithm parameter: small $\epsilon > 0$
- Initialize:
    - $\pi(s)$: (arbitrarily) a $\epsilon$-soft policy
    - $Q(s,a) \in R$ (arbitrarily), for all $s \in S$
    - $Returns(s, a) \leftarrow$ empty list, for all $s \in S, a \in A(s)$
- Loop forever (for each episode):
    - Generate an episode following $\pi: S_0, A_0, R_1, ... S_{T-1}, A_{T-1}, R_T$ 
    - $G \leftarrow 0$
    - Loop for each step of episode: $t=T-1, T-2, ..., 0$:
        - $G \leftarrow \gamma G + R_{t+1}$ 
        - Unless the pair $S_t, A_t$ appears in $S_0, A_0, S_1, ..., S_{T-1}, A_{T-1}$
            - Append G to $Returns(S_t, A_t)$

- Notes:
    - Without the assumption of exploring starts, however, we cannot simply improve the policy by making it greedy with respect to the current value function, because that would prevent further exploration of nongreedy actions
