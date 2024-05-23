- To ensure that well-defined returns are available, the Monte Carlo Method is only defined for episodic tasks.

### 1. Monte Carlo Prediction (evaluation)

- a visit to state s: a occurrence of state s in an episode

- **first-visit MC method** estimates $v_{\pi}(s)$ as the average of the returns following first visits to s, whereas the **every-visit MC method** averages the returns following all visits to s.

First-visit MC prediction, for estimating V $\approx v_{\pi}$

- Input: a policy to be evaluated
- Initialize:
    - $V(s) \in R$ arbitrarily, for all $s \in S$
    - $Return(s) \leftarrow$ an empty list for all $s \in S$
- Loop forever (for each episode)
    - Generate an episode following $\pi: S_0, A_0, R_1, S_1, A_1, ..., S_{T-1}, A_{T-1}, R_T$
    - $G \leftarrow 0$
    - Loop for each step of episode, $t=T-1, T-2, ..., 0$:
        - $G \leftarrow \gamma G + R_{t+1}$ (Recall that $G_t = R_{t+1} + \gamma G_{t+1}$)
        - Unless $S_t$ appears in $S_0, S_1, ..., S_{T-1}$:
            - Append G to $Returns(S_t)$
            - $V(s_t) \leftarrow average(Returns(S_t))$
            
- An important fact about MC method is that the estimates for each state are **independent.** The estimate for one state does not build upon the estimate of any other state, as is the case in DP.

For MC method, the computational expense of estimating the value of a single state is independent of the number of states.

### 2. Monte Carlo Estimation of Action Values.

Estimations of state values are only usable when the model of the environment exists.

In the estimation of action values, we focus on visits to a state-actiion pair $q(s,a)$

General problem of maintaining exploration: for example, if the policy is deterministic, many actions may not be taked and therefore many state-action pairs would not be visited.

**Assumption (1)** of exploring starts (for solve the problem of maintaining exploration): episodes start in a state-action pair, and every pair has a nonzero probability of being selected as the start. (So every state-action pair will be visited an infinite number of times in the limit of an infinite number of episodes.)

### 3. Monte Carlo Control

**Assumption (2)** of infinite number of episodes: policy evaluation can be done with infinite number of episodes (complete policy evaluation).

Solution to assumption 2: alternate between evaluation and improvement on an episode-by-episode basis (e.g. value iteration can be seen as an extrem example)

Monte Carlo control with ES (Exploring Starts), for estimating $\pi \approx \pi_{\star}$

- Initialize:
    - $\pi(s) \in A(s)$ (arbitrarily), for all $s \in S$
    - $Q(s,a) \in R$ (arbitrarily), for all $s \in S$
    - $Returns(s, a) \leftarrow$ empty list, for all $s \in S, a \in A(s)$
- Loop forever (for each episode):
    - Choose $S_0 \in S, A_0 \in A(S_0)$ randomly so that all state-action pairs have probability > 0
    - Generate an episode from the chosen start following $\pi(s)$, as $S_0, A_0, R_1, S_1, ..., S_{T-1}, A_{T-1}, R_T$
    - $G \leftarrow 0$
    - Loop for each step of episode: $t=T-1, T-2, ..., 0$:
        - $G \leftarrow \gamma G + R_{t+1}$
        - Unless the pair $S_t, A_t$ appears in $S_0, A_0, S_1, ..., S_{T-1}, A_{T-1}$
            - Append G to $Returns(S_t, A_t)$
            - $Q(S_t, A_t) \leftarrow average(Returns(S_t, A_t))$
            - $\pi(S_t) \leftarrow argmax_a(Q(S_t, A_t))$

The essential technique of above algorithm is that after each update of $Q(S_t,A_t)$, the improvement (greedification) will be made directly.

### 4. Monte Carlo Control without Exploring Starts (solving assumption 1)

On policy methods: attempt to evaluate or improve the policy that is used to make decisions. (e.g., MC with ES, dynamic programming etc.)

Off policy methods: evaluate or improve a policy different from that used to generate the data.

$\epsilon$-soft policy: all actions have probability of $\pi(a|s)>\frac{\epsilon}{|A(s)|}$ for all states.

$\epsilon$-greedy policy: all non-greedy action are given the minimal probability of selection $\frac{\epsilon}{|A(s)|}$, the greedy action has the probability of $1 - \epsilon + \frac{\epsilon}{|A(s)|}$. Among $\epsilon$-soft policies, $\epsilon$-greedy policies are in some sense those that are closest to greedy.

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
            -```

Let me know if you need further assistance!
