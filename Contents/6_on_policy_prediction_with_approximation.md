# Chapter 9. On-policy Prediction with Approximation

In this chapter, we now focus on estimating the state-value function from on-policy data. The approximate value function will be represented not as a table but as a parameterized functional form, written as $\hat{v}(s, w) \approx v_\pi(s)$, with weight vector $w \in \mathbb{R}^d$.

For example, $\hat{v}$ might be a linear function in features of the state, with w the vector of feature weights. More generally, $\hat{v}$ might be the function computed by a multi-layer artificial neural network, with $w $the vector of connection weights in all the layers. 

Typically, the number of weights (the dimensionality of $w$) is much less than the number of states ($d << |S|$), and changing one weight changes the estimated value of many states. Consequently, when a single state is updated, the change generalizes from that state to a↵ect the values of many other states. Such generalization makes the learning potentially more powerful but also potentially more di cult to manage and understand.

## 9.1 Value-function Approximation

- Update rules recap:
    - Dynamic programming: 
    $s \rightarrow E_\pi[R_{t+1} + \gamma \hat{v}(S_{t+1}, w_t)|S_t = s]$
    - Monte Carlo:
    $s \rightarrow G_t$
    - Temperal Difference:
    $s \rightarrow R_{t+1} + \gamma \hat{v}(S_{t+1}, w_t)$

- Supervised learning for function approximation
    - we interpret each update as specifying an example of the desired input-output ($s\rightarrow u$) behavior of the value function, with $u$ indicating the $\textit{udpate target}$.
    - In function approximation, we pass the input–output behavior $s\rightarrow u$ of each update as a training example, then interpret the approximate function they produce (after training) as an estimated value function

- The Prediction Objective ($\overline{VE}$)

    - Motivation: by assumption we have far more states than weights, so making one state’s estimate more accurate invariably means making others’ less accurate. We are obligated then to say which states we care most about.

    - Measure: Mean Squared Value Error ($\overline{VE}$)
        $$
        \overline{VE}(w) = \sum_{s \in S} u(s)[v_\pi(s) - \hat{v}(s,w)]^2
        $$
        - The state distribution $u(s) \ge 0, \sum_s u(s)=1$ is called $\textit{on-policy distribution}$, and denotes how much we care about the error in each state $s$. “Often μ(s)is chosen to be the fraction of time spent in $s$.

    - Notes:
        - The best value function for find a better policy in control is not necessarily the best for minimizing $\overline{VE}$. Nevertheless, it is not yet clear what a more useful alternative goal for value prediction might be. For now, we will focus on $\overline{VE}$.

        - Often times, for complex function approximators such as Neural Networks, we can not find a global optimum of $w_\star$, for which  $\overline{VE}(w_\star) \le \overline{VE}(w)$ for all $w$. Rather, we can only find a local optimum for which  $\overline{VE}(w_\star) \le \overline{VE}(w)$ for all $w$ in some neighborhood of $w_\star$, but oftern this is enough.

## 9.2 Stochastic-gradient and Semi-gradient Methods


