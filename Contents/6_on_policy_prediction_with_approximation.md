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


- Setup for gradient descent methods: 
    - the weight vector is a column vector with a fixed number of real valued components, $w \dot= (w_1,w_2,...,w_d)^T$. (Note that in this book vectors are generally taken to be column vectors unless explicitly written out horizontally or transposed.)
    - the approximate value function $v(s,w)$ is a differentiable function of $w$ for all $s \in S$. 

- Sotchastic gradient method (SGD)
    - Setup: assume that on each step, we observe a new example $S_t \rightarrow v_\pi(S_t)$ consisting of a (possibly randomly selected) state $S_t$ and its true value under the policy.
    - SGD method:
        - update rule:
            $$
            \begin{align}
            \boldsymbol{w_{t+1}} &\dot= \boldsymbol{w_t} - \frac{1}{2} \alpha \nabla[v_\pi(S_t) - \hat{v}(S_t, \boldsymbol{w_t})]^2 \\
            &= \boldsymbol{w_t} - \alpha [v_\pi(S_t) - \hat{v}(S_t, \boldsymbol{w_t})]\nabla\hat{v}(S_t, \boldsymbol{w_t})
            \end{align}
            $$

        - Notes: 
            - The assumption of available $v_\pi(S_t)$ is clearly impossible in practice. In fact, as long as the target is an unbiased estimate of $v_\pi(S_t)$, i.e., $E[Target | S_t=s] = v_\pi(S_t)$, then $\boldsymbol{w_t}$ is guaranteed to converge to a local optimum under the usual stochastic approximation condition for decreasing $\alpha$ (topics of convergence are not included in DistilRL, please refer to the book chapter 2.7 for details.)
            - By definition, the Monte Carlo target $G_t$ is an unbiased estimator of $v_\pi(S_t)$.
    
    - Gradient Monte Carlo Algorithm for estimating $\hat{v} \approx v_\pi$
        - Input: the policy $\pi$ to be evaluated, 
        - Input: a differentiable function $\hat{v}: S \times \mathbb{R}^d \rightarrow \mathbb{R}$
        - Algorithm parameter: step size $\alpha > 0$
        - Initialize value function weights $\boldsymbol{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\boldsymbol{w=0}$)
        - Loop forever (for each episode):
            - Generate an episode $S_0, A_0, R_1, ... S_{T-1}, A_{T-1}, R_T, S_T$ using $\pi$.
            - $ \text{for } t \text{ in } \{T-1, T-2, ..., 0\}$:
                $$
                \boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha [G_t - \hat{v}(S_t, \boldsymbol{w})]\nabla\hat{v}(S_t, \boldsymbol{w})
                $$

- Semi-gradient method:
    - Setup: the training example $S_t \rightarrow U_t$ with $U_t \in \mathbb{R}$ is not the true value $v_\pi(S_t)$ but a **boostraping target** using $\hat{v}$

    - Semi-gradient methods:
        - update rule
            $$
            \begin{align}
            \boldsymbol{w_{t+1}} &\dot= \boldsymbol{w_t} - \frac{1}{2} \alpha \nabla[U_t - \hat{v}(S_t, \boldsymbol{w_t})]^2 \\
            &= \boldsymbol{w_t} - \alpha [U_t - \hat{v}(S_t, \boldsymbol{w_t})]\nabla\hat{v}(S_t, \boldsymbol{w_t})
            \end{align} 
            $$

        - Notes:
            - There is no guarantees for convergence as for stochastic gradient methods if a bootstrapping estimate of $v(S_t)$ is used as the target $U­_t$. Boostrapping methods use $\hat{v}(S_{t+1}, \boldsymbol{w_t})$ as the target, which depends on the current $\boldsymbol{w_t}$. Yet the derivation from equation $(1)$ to $(2)$ requires independence regarding $\boldsymbol{w_t}$. 
            
                In other words, bootstrapping methods take into account the effect of changing the weight vector $\boldsymbol{w_t}$ on the estimate, but ignore its effect on the target. They include only a part of the gradient and, accordingly, we call them $\textit{semi-gradient methods}$.

            - Although semi-gradient (bootstrapping) methods do not converge as robustly as gradient methods, they do converge reliably in important cases such as the linear case, and they are usually preferred in practice due to the boostrapping advantage against monte carlo methods.
    
    - Semi-gradient TD(0) for estimating $\hat{v} \approx v_\pi$
        - Input: the policy $\pi$ to be evaluated, 
        - Input: a differentiable function $\hat{v}: S^{+} \times \mathbb{R}^d \rightarrow \mathbb{R}$ such that $\hat{v}(terminal, a)=0$
        - Algorithm parameter: step size $\alpha > 0$
        - Initialize value function weights $\boldsymbol{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\boldsymbol{w=0}$)
        - Loop forever (for each episode):
            - Initialize $S$
            - Loop for each step of episode:
                - Choose $A \sim \pi(a|s)$
                - Take action $A$, observe $R, S\prime$
                - $\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha [R + \gamma \hat{v}(S\prime, \boldsymbol{w}) - \hat{v}(S_t, \boldsymbol{w})]\nabla\hat{v}(S_t, \boldsymbol{w})$
                - $S \leftarrow S\prime$
            - until $S$ is terminal
    
-  <span style="color:red;">Example of State Aggregatioin:</span>