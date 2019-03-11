# [UCL&Deepmind COMPM050: Reinforcement Learning](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

### [Course website](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
### [Assignment: Easy21](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf)
> The goal of this assignment is to apply reinforcement learning methods to a simple card game that we call Easy21.
  
# Notes
## Lecture 1: [Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=2pWv7GOvuf0)

### About Reinforcement Learning

Reinforcement Learning Paradigm:
1.  No supervisor, only reward.
2.  Feedback is delayed, reward might be given after some time.
3.  The agent influences the data it receives.

### The Reinforcement Learning Problem
The reward is a scalar signal, and the goal of an agent is to **maximise total future cumulative** award.
The **history** `H` is a sequence of observations, actions, rewards. The algorithm is a mapping from the history to the next action `A` . The history might be long and complex, so state `S` is used to represent that history and is used to determine what action should be performed next: `S = f(H)` .

There is an _agent state_ and an _environment state_. The agent does not usually have access to the latter so it makes actions based solely on the _agent state_.

An **information state** a.k.a. **Markov state** contains all useful information from the history. A state `S` is Markov iff the probability of the state `P(S_t+1|S_t) = P(S_t+1|S₁,...S_t)`, meaning that the future state is only dependent on the current state, not on the history.

The future is independent of the past given the present.

The environment state, by the definition **is Markov**.

**Full Observability** means that the agent sees the environment state. Here _Agent state = environment state = information state_, which is called a **Markov Decision Process (MDP).**

**Partial observability** means that the agent does not see a complete environment state. This is called a **Partially Observable MDP (POMDP).**

The agent needs to construct its own state representation. This can be done in different ways, e.g.
-   based on the whole history,
-   beliefs of environment state, i.e. a probability distribution among different possible environment states,
-   a recurrent neural network, i.e. weighted sum of previous state and current observation.

### Inside an Reinforcement Learning Agent
An RL Agent might include some of these components:
-   Policy: agent's behaviour function (map from a state to action) - it can be deterministic (given some state make some action) or stochastic (probability distribution over actions given state).
-   Value function: how good is each state and/or action (prediction of expected future reward)
-   Model: agent's representation of the environment (it predicts what the environment will do next). There are usually two models: **Transitions model** (predict the next state) and **Rewards model** (predict the next, immediate reward).

The agent can be:
-   Value-based (value function),
-   Policy-based
-   Actor-critic (both value-based and policy-based)
or:
-   Model-free (based on policy and/or value function), without trying to figure out how the environment works,
-   Model-based - first we build a model of an environment

  
### Problems within Reinforcement Learning
There are two problem settings with sequential decision making in RL:
1.  **Reinforcement Learning**: the environment is initially unknown and the agent improves its policy via the interaction with it.
2.  **Planing:** the model of the environment is known and the agent performs computations with its model.

There is an **exploration** vs. **exploitation** tradeoff.


## Lecture 2: [Markov Decision Process](https://www.youtube.com/watch?v=lfHX2hHRMVQ)

### Markov Processes/Markov Chain
Markov Decision Process formally describe a fully observable environment in RL. However, partially observable problems can be converted into MDPs.

**Markov Chain** is a tuple `<S,P>` where `S` is a **finite space of possible states** and `P` is a **probability (transition) matrix** of switching from one state to another (future) state.

 
### Markov Reward Process
-   MRP is a tuple `<S,P,R,γ>` , where `R` is a **reward function** and `γ` is a **discount factor**.
-   The return `G_t` is the total discounted reward from time-step `t`.
-   The value function `v(S)` gives the long term value of for the state S . It is the expected value of the total reward given you find yourself in the state `S` .

**The Bellman Equation** describes the value function vector as a sum of immediate rewards R and the product of transition matrix P and the value function vector v : **`v = R + γPv`** . It is a linear equation and solving for `v` is `O(n³)` for `n` states (requires the inversion of a matrix). We deal with it using iterative methods:

1.  Dynamic Programming
2.  Monte Carlo evaluation
3.  Temporal-difference learning

### Markov Decision Process

MDP is a tuple `<S,A,P,R,γ>` , where `A` is a **finite set of actions**. For each action a there is a separate transition matrix `P` and the reward function `R` .

A **policy**  `π` is a probability distribution over actions given states. They fully define the behaviour of an agent and are only dependent on the current state.

An **action-value function** **`q_π(s, a)`**  is the expected return of starting at state `s` , performing an action a and then following a policy `π` .

**Bellman Optimality Equation** is nonlinear and does not have a closed form solution. The iterative solutions exist and include:

1.  Value iteration
2.  Policy iteration
3.  Q-learning
4.  Sarsa

### Extensions to MDPs
-   Infinite and continuous MDPs
-   Partially observable MDPs
-   Undiscounted, average reward MDPs

## Lecture 3: [Planning by Dynamic Programming](https://youtu.be/Nd1-UUMVfz4)
### Introduction
**Policy Evaluation** - Given policy, how good it is?

**Policy Iteration** - Takes an idea of evaluating policy to make it better.

**Value Iteration** - Works on improving the value function by applying a _Bellman Equation_ iteratively.

**Markov Decision Process** satisfies the properties of dynamic programming: **optimal substructure** and **overlapping subproblems** through the Bellman Equation (recursive decompositions). The value function, on the other hand, acts as a cache of solutions.

  
### Policy Evaluation
**Problem**: Evaluate given policy `π` .
**Solution**: Iterative application of Bellman expectation backup.
Given a policy (e.g random walk on a grid) we can iteratively apply Bellman Equation to find a value function (i.e. expected long-term reward for each state) which, itself might start pointing towards better policy.

  

### Policy Iteration
Given a policy `π` we want to first evaluate it: `v_π(s)` and then, greedily improve it with respect to `v_π` to get `π'` .

We can improve the policy after a specific number of iteration `k` or introduce a stopping condition, i.e. stop if the value function does not change more than `ε` . In the extreme case, if `k = 1` , this is called a _value iteration_.

  

### Value Iteration
Value iteration is another mechanism for solving MDP.
**Problem**: Find optimal policy `π` .
**Solution**: Iterative application of Bellman optimality backup.

In value iteration there is no explicit policy (as in _Policy Iteration_), for the intermediate value functions there might not necessary exist a policy that leads to this value function.

  

## Synchronous Dynamic Programming Algorithms
| Problem | Bellman Equation | Algorithm |
|--|--|--| 
| Prediction | Bellman Expectation Equation | Iterative Policy Evaluation |
| Control | Bellman Expectation Equation + Greedy Policy Improvement | Policy Iteration |
| Control | Bellman Optimality Equation | Value Iteration

Prediction is about how much reward are we going to get given policy `v_π`. Control is about finding `v_*` and hence `π*` .

The **Complexity** of the algorithms, given m actions and n states.

In each sweep we consider `n` states and for each state n possible successor states, for which we can take `m` actions, which gives `O(mn²)` per iteration. You can apply Bellman Equations for `q` (action-value function) just like for `v`, but this is `O(m²n²)` per iteration (since you need to consider all action **pairs**).

  

### Extensions to Dynamic Programming
-   Asynchronous DP: In-place DP, Prioritised sweeping, Real-time DP

DP uses full-width backups (we consider every action and all possible states that we might find ourselves after making that action). This is effective for medium-sized problems (millions of states), but for large problems it suffers from Bellman's curse or dimensionality, since the number of states grows exponentially with the number of state variables. We use **sampling** instead of considering every possible combination.

 
### Contraction Mapping
For any metric space `ν` that is complete (i.e. closed) under an operator `T(v)`, where `T` is a `γ-contraction` (i.e. makes function values closer by at least `γ`): `T` converges to a (unique) fixed point at a linear convergence rate of `γ`.

  
## Lecture 4: [Model-Free Prediction](https://youtu.be/PnHCvfgC_ZA)

### Introduction
**Model-free prediction** is about **estimating the vale function** (given policy) for an **unknown MDP**. Here we use **value function** **`v`** **.**

**Model-free control** is about **optimising value function** (and thence finding an optimal policy) for an unknown MDP. Here we use **action-value function** **`q`**  , as we have an access to the actions in control.

  

### Monte-Carlo Learning

Goal: Learn a value function `v_π` (expected future return for any state) from episodes of experience under a policy `π` .

MCRL is about sampling (`S, A, R` , i.e _state, action, reward_) and assigning a mean return as a value for given initial state. In another words, you run some episodes and collect all the rewards and update the estimate of the return for each state you visit.

  

### Temporal-Difference Learning
The difference between TD and MC is that TD learns from _incomplete_ episodes (by bootstrapping), so it does not need to complete the whole trajectory (until the termination state is achieved), but the partial trajectory and then make an estimate of the remaining reward.

In TD learning we use an estimated return `R_(t+1) + γv(s_(t+1))` (also called a **TD target**) instead of the actual return `G_t` .

**The properties of TD:**
-   TD can learn before knowing the actual outcome and can learn online after each step,
-   TD can learn from incomplete sequences,
-   TD works in continuing (non-terminating) environments.

**MC has high variance, zero bias**, whereas **TD has high bias, low variance.**

TD exploits Markov property (is more efficient in Markov environments), whereas MC does not (and is more efficient in non-Markov environments).

The contrast between TD/MC and Dynamic Programming is that MC/TD are sampling from the possible actions/rewards and update the value function, whereas DP does a full lookahead at all possible actions and what the environment can do to us and compute a full expectation.

**Bootstrapping:** update involves an estimate - DP, TD (MC does not bootstrap, it uses real return values on the way). In DP, TD we bootstrap from our estimated value at the next step.

**Sampling**: update samples an expectation - MC/TD (DP does full-width exhaustive search).


### **TD(λ)**

**TD(λ) algorithm** is a generalisation of TD/MC, where we are somewhere in between the shallow backup (TD) and deep backup (MC). We can do `1, 2, ..., n` MC steps: that would be called `TD(1), TD(2), ... TD(n)` respectively. In **TD(λ),** `λ`-return combines all `n-step` returns `G` and is a geometrically weighted average of all of them.

**Backward-view TD(λ)** will keep an eligibility trace (heuristic that combines both frequency and recency heuristics) and update the value function in proportion to the TD error and the eligibility trace.


## Lecture 5: [Model-Free Control](https://youtu.be/0g4j2k_Ggc4)

### **Introduction**
**On-policy learning** - Learn about the policy π by sampling from the same policy.

**Off-policy learning** - Learn about the policy π from experience sampled from μ (you do not have to learn about the behaviour that you sample).

  

### **On-policy Monte Carlo Control**

**Greedy Policy Improvement** over `V(s)` requires model of MDP, while greedy policy improvement over `Q(s,a)` is model-free.

In order to ensure exploration, we might use `ε-greedy` exploration, which chooses a greedy action with probability of `1-ε` and a random action with probability of `ε`.

We are not sure, however, that `ε-greedy` will lead to an optimal policy `π*` . To ensure it does, we must prove it is **GLIE (Greedy in the Limit with Infinite Exploration)**. GLIE has two properties:

-   all state-action pairs are explored infinitely many times,
-   the policy converges on a greedy policy.

`ε-greedy` has those two properties if we assume that `ε` decreases at each iteration, e.g. `ε = 1/k` .


### On-policy Temporal Difference Learning
**MC vs. TD**

**Advantages of TD:** Lower variance, online, incomplete sequences. Online means: "_you see one step of data, you bootstrap, you update your value function immediately"._ The idea is to use TD instead of MC in the control loop: apply TD to `Q(S,A)` , use `ε-greedy` policy improvement and update every step. This algorithm is called **SARSA.** Here we start with state `S` and given action `A`, we sample from the environment and get an immediate reward `R`, we end up in other state `S'` and sample a new action `A'` from our policy.

  
**Sarsa(λ)** allows for extending Sarsa in future by computing all n-step returns and weighting them accordingly. The problem is that this algorithm is not online anymore as we need to go through the whole trajectory until terminal state is achieved (is such exists). We deal with that using **Eligibility Traces.** Eligibility Traces are the maps `E(s,a)` that store the value indicating how given pair was responsible for the reward we got. Whenever we experience a pair `s,a` , its eligibility trace increases and its value decays in time if we do not experience it.

In **Sarsa(0)** you only propagate the information back by one step per episode, but in **Sarsa(λ)** you do it for many states weighted by their eligibility traces. In MC every state would be updated with the same amount.

  

### Off-policy Learning

Off-policy learning is about evaluating target policy `π(a|s)` to compute `v(s)` or `q(s,a)` while following **behaviour policy**  `μ(a|s)` . In off-policy, MC is extremely high variance, so we **have to use TD** (bootstrapping), but it requires Importance Sampling and has some variance as well. The idea that works best with off-policy learning is **Q-learning**.

The idea of Q-learning is to update the `q` values in the direction of the best possible next `q` value you can have after one step.


## Lecture 6: [Value Function Approximation](https://youtu.be/UoPei5o4fps)

### **Introduction**
For large MDPs, instead of finding the true value function `v(s)` we will try to fit an approximate function (v dash) `v^(s,w)` , which is a parametric function with some weight matrix **`w`** . Similarly, for the action value function `q(s,a)` we find `q^(s,a,w)` .

  

### **Incremental Methods**
Here we are going to train the approximate function using incremental method, e.g. SGD.

It is useful to represent the state space as a **feature vector**, which is responsible for representing our state, e.g. the entries in the vector specify our x, y, z coordinate. In linear case (linear combination of features) this feature vector is used to compute the update of weights. In nonlinear case (neural networks) we need to compute the actual gradient.

In order to train `v^(s,w)` , we need to know true `v(s,w)` to compute error and back-propagate it using SGD. However, we do not know the true value function so we substitute it by some target.

-   For **MC** the target is the return `G_t` (we need to get to the end of our episode).
-   For **TD(0)** the target is `R_t+1 + γv(s_t+1,w)`.
-   For **TD(λ)** the target is the `λ`-return.

In **MC** we are essentially finding a mapping `S→G` given pairs `S,G` making it a supervised problem. `G_t` is an unbiased estimator of a true function value.

In **TD** the target is a biased estimator of a true value function because when calculating an error of estimation for given state, we need to query the same network for the value for the next state.

For approximating action-value function the process is similar. The feature vector here represents both states and actions. In MC the target is again `G_t` and in TD(0) the target is `R_t+1 + γq(s_t+1,A,w)` .

In both cases the forward and backward view (with Eligibility traces) are equivalent!

  
**Prediction**
The problem with TD methods is that they might diverge for on-policy in nonlinear case (nonlinear function approximator, e.g. neural network) and for off-policy in linear and nonlinear case (it only converges for table lookup). However, **Gradient TD** converges in all of those cases.

  
**Control**
In Control, both MC Control, Sarsa, Q-learning and Gradient Q-learning can diverge in nonlinear case. Q-learning does not guarantee to converge even in linear case.

### Batch Methods
In Batch Methods we collect a dataset of experience `D` (`s,v` pairs) and after that, we randomly select samples from that dataset and perform SGD using them (this is called **experience replay**). Important thing about the experience network is that it stabilises neural network methods since it decorrelates the trajectories (highly correlated parts of the trajectory are one after the other).

  
This is used in **Deep Q-Networks (DQN)**, where experience replay and **fixed Q-targets** ensure that the function approximator does not diverge. **Fixed Q-targets** means a second neural network used for storing parameters for some time (freezed), so that we do not "bootstrap directly towards the thing that we're updating at that moment, because that might be unstable". After some time we equalise those two networks and again, freeze one of them and use it to update the other.

  

Experience replay might need many iterations to find a solution. For **linear value** function approximation we can solve the Least Squares directly. We know that the expected change to the weights given an optimum is zero (we do not want to update anything). Given that knowledge we can find a closed-form solution. This requires inverting a matrix, which, for n features is `O(n³)` (it does not depend on the number of states anymore), or **Sherman-Morrison** for incremental solution in `O(n²)` if we want to invert `(A+uv')⁻¹` and the inverse of `A` is known.

  

We end up with new algorithms: **LSMC (Least Squares Monte Carlo), LSTD, LSTD(λ)**. They all always converge to the right policy both in on/off policy. In **Control** we have an **LSPI (Least Squares Policy Iteration)**

  

  

## Lecture 7: [Policy Gradient Methods](https://youtu.be/KHZVXao4qXs)

### **Introduction**
We need to parametrise the policy to give us a distribution of possible actions that we can undertake given some state: `π(s,a) = P(a|s,θ)` and we need to learn the parameter `θ` , e.g. using neural network.

We have a **Value-based** and **Policy-based RL:**
-   Value-based: Learnt value function and implicit policy (e.g. `ε-greedy` ),
-   Policy-based: No value function and learnt policy,
-   Actor-Critic: Learnt value function and learnt policy.

Sometimes using the policy might be more compact, e.g. in Atari games instead of computing the value function (i.e. given move will give me a total cumulative future reward of `R` ), you learn a policy to do a certain move when something happens.

## Advantages and Disadvantages of Policy-based method
| Type | Policy-based  |
|--|--|
| Advantages | Better convergence, Effective in high dimensional and continuous space, Can learn stochastic policies |
| Disadvantages | Typically converges to local (not global) optimum, Evaluating policy is inefficient and high variance |

State aliasing means that the world is partially observed. In such cases **stochastic policy** might be better.

  

**Policy Optimisation**
Optimisation methods can be divided into:

**Gradient-based methods, e.g.**
-   Hill-climbing
-   Simplex/amoeba/Nelder Mead
-   Genetic algorithms

**Gradient-free methods, e.g.**
-   Gradient descent
-   Conjugate gradient
-   Quasi-newton

  

### Finite Difference Policy Gradient
In Gradient ascent if we do not have an access to the closed form gradient we can compute it using **finite difference method**. For each parameter we perturb its value by a tiny bit ε and see what is the response in the function. However, this naive method is ineffective in high dimensional spaces. The solution would be to use techniques for random directions, e.g. **Simultaneous perturbation stochastic approximation (SPSA)** (they are also quite noisy though).

  

### Monte Carlo Policy Gradient

**Score function** describes how sensitive is the Likelihood function to a parameter `θ`.
**Softmax Policy** is the alternative for e.g. `ε-greedy` (Softmax, in general, is a policy that is proportional to some exponentiated value).

**Gaussian Policy**, where the action is sampled from a Gaussian distribution with mean of `μ(s)` (linear combination of state features) and variance `σ²` .
  
**Policy Gradient Theorem** states that he policy gradient is the expectation of the score function multiplied by the long-term action-value function `Q(s,a)` - "you want to adjust the policy that does more of the good things".

**Monte Carlo Policy Gradient** (a.k.a REINFORCE) is an algorithm for finding parameters θ by updating parameters using Stochastic Gradient Ascent (SGA), using Policy Gradient Theorem and using `v_t` as an unbiased sample of `Q(s_t, a_t)` (Q for policy `π`).

  

### Actor-Critic Policy Gradient
The problem with MCPG is that it has high variance. In Actor-Critic methods, instead of using the return to estimate the action-value function we are going to explicitly estimate the action-value function using a _critic_.

The name **Actor-Critic** comes from the fact that the algorithm maintains two sets of parameters: **Critic -** updates action-value function parameters `w` , and **Actor** - updates policy parameters `θ`in the direction suggested by critic. Those algorithms follow an **approximate policy gradient**. The critic is solving the problem of policy evaluation, so we can use methods such as MC, TD(0), TD(λ), Least Squares Policy Evaluation etc.

If you use Softmax function approximator you will achieve a global optimum. This is not guaranteed for a neural network.

**Baseline Method** is a method to decrease a variance of expectation without changing it. This is achieved by subtracting a baseline function `B(s)` from the policy gradient.

**Advantage function** **`A(s,a)`**  tells us how much better than usual is it to take action `a`. The advantage function can significantly reduce the variance of policy gradient.

**Natural Policy Gradient**
"Adjust the policy in the direction that gets you more Q". It scales much better in high dimensions and is better than stochastic approach.

  

  

  

## Lecture 8: [Integrating Learning and Planning](https://youtu.be/ItMutbeOHtc)

### Introduction
**Model**, in RL is the agent's understanding of the environment.

**Recap** 
Reinforcement Learning can be:
-   Model-free: no model, learn value function (and/or policy) from experience (the agent does not try to explicitly represent the transition dynamics of the reward function),
-   Model-based: learn a model from experience and **plan** a value function (and/or policy) from model. This allows us to _think_, _plan_ our actions, make some search trees for possible actions without actually making any action. Here, _model learning_ is like building an MDP and _planning_ is solving this MDP.


### Model-Based Reinforcement Learning
The goal is to estimate the model M from experience `S,A,R...S` , which is a supervised learning problem of form:

-   `s,a → r` is a Regression problem,
-   `s,a → s'` is a density estimation problem (we want to find the best distribution of possible states after performing action `a` from state `s`.

Here the problem is to find an appropriate **loss function:** in case of Regression that might be **MSE**, for density estimation: **KL divergence.**

**Example Models:**
-   Table Lookup Model
-   Linear Expectation Model
-   Linear Gaussian Model
-   Gaussian Process Model
-   Deep Belief Network Model

Table Lookup Model - count the number of times we end up in a particular state and compute an average reward for given state and action.

  

### Integrated Architectures
**Integrating Learning and Planning**
We can integrate Model-free and Model-based learning into **Dyna** - where we learn a model from the real experience and we learn and plan a value function (and/or policy) from both real and simulated experience. The simplest version of Dyna is a **Dyna-Q Algorithm.** Another variation, **Dyna-Q+** encourages exploration more.

### Simulation-Based Search

**Forward Search** algorithm selects best action based on lookahead (search tree), which does not require solving whole MDP but the sub-MDP starting from now (we do not care about the states that are not reachable from now). Simulation-Based Search is a paradigm using Forward Search.

**Monte Carlo Search** simulates `K` episodes from current (real) state `s_t` and evaluates all the possible actions that we might take from this state by a mean return (Monte Carlo Evaluation).

**Monte Carlo Tree Search** is the state of the art search method. Here, the policy `π` improves with time. It essentially is a Monte Carlo control applied to simulated experience (from now on).

**Temporal Difference Search** is applying Sarsa to simulated experience (from now on).

**Dyna-2** stores two sets of feature weights: Long-term memory (updated from real experience) and Short-term memory (updated from simulated experience).

  

## Lecture 9: [Exploration and Exploitation](https://youtu.be/sGuiWX07sKw)

### Introduction

There are roughly tree families of approaches to Exploration vs. Exploitation tradeoff. These are:

-   Random exploration, e.g. ε-greedy , Softmax, Gaussian noise.
-   Optimism in the face of uncertainty: estimate the uncertainty of the value and prefer to explore states/actions with highest uncertainty, e.g. Optimistic initialisation, UCB, Thompson sampling.
-   Information state search (most computationally difficult): consider agent's information as a part of its state (i.e. _"my state of being here and knowing what is behind that door is different than me being here and not knowing that"_), e.g. Gittins indices, Bayes-adaptive MDPs.

  

### Multi-Armed Bandits

Multi-Armed Bandit is a tuple `<A,R>` , where the goal is to maximise a total cumulative reward. Regret is the difference between the optimal value `V*` and our action value `Q(a)` . The possible approaches are:

-   Greedy algorithm (might be stuck in a suboptimal action forever and has a linear total regret - remember regret is an **expectation**),
-   Optimistic Initialisation - initialise values to the maximum reward (if known) and then act greedily.
-   ε-greedy
-   Decaying ε-Greedy - the value of ε decays in time - this gives a sublinear (logarithmic) asymptotic total regret.
-   Upper Confidence Bound (UCB) - select the action which has the highest UCB value of return (not expected return!), e.g. **UCB1 Algorithm**

**![Screenshot 2019-03-09 at 10.43.14.png](https://coda.io/contentProxy/NbZkIAHwFH/blobs/bl-FMyIqtd8SL/e2db1436d4b6b003a6d4061568dd3b8192ae54d8cb4860aca025f429055486dc7750ecd2979774b20e8b20546eb6f7906ee16ce4e41c87768a1197029e4628ce70c81ffbb9ab7f025aeaed24cba26bb990b90ed6a150822b74a167fb201530683e8f67cf)**

The Theorem (Lai and Robbins):

> Total Regret is at least logarithmic in the number of steps.

**Bayesian Bandits**

Bayesian Bandits exploit prior knowledge about the distribution `p(Q|w)` over the action-value function. Then we do the same thing as in UCB, selecting an action for which the `μ+nσ` of the reward distribution is the biggest (we choose `n` ).
  

**Probability Matching**

Instead of using UCB, we can select an action according to the probability that this action is an optimal action. This again encourages selection of actions that have _heavy tails_, cause they have the highest possibility of being the best actions. For this purpose we use **Thompson sampling** - we sample (just one sample) from each of the distributions `Q(a₁), Q(a₂)...` and select an action for which the sample has the highest `Q` value. Thompson sampling is **asymptotically optimal.** Also it does not have a free parameter `n` (number of standard deviations) to tune.

**Information state search** is the third family of algorithms dealing with Exploration vs. Exploitation tradeoff (apart from Random exploration and Optimism in the face of uncertainty). The previous two approaches were just heuristics.  Information state space takes different approach by building an augmented MDP out of the bandit problem. It now keeps an information about the state (i.e. how many times did I choose which action). Here we can build a tree search of all possible actions and choose the best one - solving for this MDP is an optimal way for the Exploration vs. Exploitation tradeoff (not a heuristic anymore).

**Bayes-adaptive** RL is when we characterise the information via the posterior distribution.

  
### Contextual Bandits
Contextual Bandit is a Multi-Armed bandit with a space, i.e. a tuple `<A,S,R>` . The example are advertisements on the web - _which ads should be displayed and which should not?_ The state here provides a contextual information, i.e. information about the user visiting website.

  
### MDPs
**All of the ideas (e.g. UCB) extend to MDPs as well, not only Bandits.**
One successful approach to Exploration vs. Exploitation in Model-based RL is **R-max** algorithm:

> R-max – A General Polynomial Time Algorithm for Near-Optimal
> Reinforcement Learning, R. Braffman, M. Tennenholz, 2012

  

## Lecture 10: [Classic Games](https://youtu.be/kZ_AUmFcZtk)

### Game Theory

**Nash equilibrium** is a joint policy for all players such that every player's policy is a best response (**Best response** is the optimal policy against every other player's policies if they fixed their policy, e.g. keep choosing _paper_ in _rock-paper-scissors_), i.e. no player would choose to deviate from Nash.

Nash Equilibrium is a fixed-point of self-play RL, i.e. when we control two agents playing against each other and their policies converge towards a fixed point, that must be a Nash equilibrium.

### Minimax Search
Minimax value function `v*` maximises white's expected return while minimising black's expected return. Minimax policy is a joint policy `<π¹,π²>` that achieves a minimax value. Assuming two-player game, zero-sum game, perfect information: there is a unique solution and a minimax policy is a Nash Equilibrium

**Deep Blue** was an algorithm that defeated Garri Kasparov in 1997. It used minimax search (alpha-beta search a.k.a. αβ ).

  

### Self-play Reinforcement Learning
Apply value-based RL algorithms to games of self-play:
MC: update the value function towards the return G_t .
TD(0) : update the value function towards successor value v(s_t+1) .
TD(λ): update the value function towards the λ-return G_t_λ .
  

### Combining Reinforcement Learning and Minimax Search
Simple TD: update value towards successor value.
TD Root: update value towards successor search value.
TD Leaf: update search value towards successor search value.

> _Note: The plus "+" notation (in slides) means we run a search from this particular leaf onwards._

TreeStrap: update search values towards deeper search values

One of the most effective variants of Self-play RL is a **UTC** algorithm, which is a Monte Carlo Tree Search with UCB for exploration/exploitation balance.

  
### Reinforcement Learning in Imperfect-Information Games
**Smooth UTC Search** is a variant of UTC, inspired by game-theoretic Fictitious Play, where agents learn against and respond to opponent's average behaviour (not his current behaviour). Here we pick our action according to the UTC with probability `η` or the average strategy `π_avg` with probability `1-η` . This converges to the Nash equilibrium.
  
### Conclusions
RL in Classic Games Recipe:

![Screenshot 2019-03-10 at 12.45.52.png](https://coda.io/contentProxy/NbZkIAHwFH/blobs/bl-joJFHEpeOp/75f6a468aa6fa10873abe89ae4869e1194c911147f708fecef333ad6f399229338d7cda509a72c83c6e4becd4bf56ba675c54c2f63401c5aad57dbb935f8ecaa91af27a24a10c1dd4a379d501494a2f80a673c894bcc74c504b3c0322f8c1a7e90e4905b)
