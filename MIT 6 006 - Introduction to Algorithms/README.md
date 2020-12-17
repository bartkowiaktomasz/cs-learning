# [MIT 6.006: Introduction to Algorithms](https://www.youtube.com/playlist?list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb)
## [Course website - videos](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/)

## Lecture 1: Algorithmic Thinking, Peak Finding
## Lecture 2: Models of Computation, Document Distance
* Models of computation: RAM (random access machine) and Pointer Machine. 
* Python model incorporates both. These notes summarise the complexity of basic operations in Python. 
* Document distance algorithm = Split documents into words + Count word frequencies + Compute dot product.

## Lecture 3: Insertion Sort, Merge Sort
Insertion Sort is $O(n^2)$ because of $O(n^2)$ comparisons and swaps. Since insertion sort ensures that one part of the array is always sorted, an improved, Binary Insertion Sort proposes  $O(nlogn)$ comparisons, but the number of swaps stays the same.

Merge Sort is  $O(nlogn)$  but requires $O(n)$  extra space. 

## Lecture 4: Heaps and Heap Sort
Heap Sort uses Max-Heap to extract the biggest element from the array (first element) and then putting the last element on its place. Then, in order to keep Max-Heap property, Max-Heapify procedure is run for the root element with $O(logn)$. Thus the complexity of heap sort is $O(nlogn)$. Note that at the beginning, for any array, one needs to build a max-heap out of it, which is  $O(n)$.
The Build-Max-Heap requires max-heapifying top $n/2$ nodes, because there rest are leaf nodes, which do not have children and therefore obey max-heap property (any node should be greater or equal to its children).

## Lecture 5: Binary Search Trees, BST Sort
Binary search insert is $O(h)$, where $h$ is a height of the tree, find-max is $O(n)$ (keep going to the right) and find-min as well.
BST property a.k.a search property ensures that right child is greater or equal to its parent and the left child is smaller or equal to its parent.

### BST vs. Heap
> Heap just guarantees that elements on higher levels are greater (for max-heap) or smaller (for min-heap) than elements on lower levels, whereas BST guarantees order (from "left" to "right"). If you want sorted elements, go with BST
> Heap is better at `findMin/findMax` $(O(1))$, while BST is good at all finds $(O(logN))$. Insert is $O(logN)$ for both structures (Note that average insert for heap is $O(1)$). If you only care about `findMin/findMax` (e.g. priority-related), go with heap. If you want everything sorted, go with BST.

Tree traversal pattern:
(L) Recursively traverse its left subtree. This step is finished at the node `N` again.
(R) Recursively traverse its right subtree. This step is finished at the node `N` again.
(N) Process `N` itself.
Ways to traverse the tree: Pre-order (NLR),  In-order (LNR), Post-order (LRN), Out-order (RNL).

## Lecture 6: AVL Trees, AVL Sort
Tree is balanced if its height (longest distance between the root and leaf node) is $O(logn)$.
Height is a local property, for each node it can be determined by knowing the height of its children. AVL trees store that information to ensure balance.
AVL (Adelson-Velsky and Landis) property ensures that for each node, the difference between the heights of its children is at most 1. To avoid dealing with base cases, the height of empty nodes can be defined to be -1.
Based on the fact that fibonacci is exponential, we can prove that the height of the AVL is bounded by $1.44 O(logn)$. That is because in worst case scenario, for each node, its right (left) child is always one unit higher than its left child. Then, by recurrence, we prove that for given height, in worst case we still can fit an exponential number of nodes. We have:
$N_h = 1 + N_{h - 1} + N_{h-2}$
where $N_h$ is a number of nodes at height $h$. The equation above is greater than fibonacci (constant).
The insert in AVL trees is a simple BST insert and then fixing the AVL property by rotations. Rotations ensure that in-order traversal is preserved (and so the BST property).
Left (right) rotation requires the parent node to go to the left (right).
In zig-zag pattern, double rotation is required to fix the AVL. Note that in general, the AVL property violations propagate up the tree (after each fix) and all of them need to be fixed. 

AVL sort: insert $n$ nodes and then perform in-order traversal. Insert is $O(nlogn)$ and traversal is $O(n)$
Balanced BST is an abstract data type that allows for:
* insert & delete
* minimum element
* successor and predecessor 

The first two methods are characteristic for a priority queue (can be achieved e.g. by heap). The good thing about heap is that it does everything in-place.

## Lecture 7: Counting Sort, Radix Sort, Lower Bounds for Sorting
In the comparison model of computation, the lower bound for:
* searching - $\Omega(lgn)$
* sorting - $\Omega(nlgn)$

To prove the above, one needs to build a decision tree for all possible outcomes of comparisons and calculate its height in the worst case by taking the $lg$ of the number of leaves, e.g. for sorting there are $n!$ possible permutations for sorted elements, and $lg(n!)$ is $\Omega(nlgn)$ by Stirling's Approximation.

Linear Time Sorting - assuming we want to sort integers, for which we know their range (e.g. ${0,...,k}$ ) and assuming they fit into one word of RAM, they can be sorted in linear time using Counting sort or Radix sort.

In Counting Sort, an array of size $k$ is allocated in memory and is used to count each integer. At the end the array is traversed and the numbers are printed depending on their count. Running time is $O(n + k)$ so it is linear as long as $k$ is of order $n$.

In Radix Sort (bucket sort), if integers are represented in base $b$ and the biggest integer is $k$, the number of digits $d$ needed to encode that is $log_b(k)$. To sort the numbers it suffices to sort all of them based on digits only, starting from least significant and ending on the most significant. Here the running time is $O(d(n + b)) = O(log_b(k)*(n + b))$. In order to minimise it, set $b = O(n)$ so that the running time is $O(nlog_n(k))$, meaning that $k$ might be a polynomial in $n$ and the running time is still linear.


## Lecture 8: Hashing with Chaining
The simplest form of a hash table would be a Direct Access Table (DAT), but there are two problems with DAT: 1) If two keys have the same hash, the one inserted later overrides the previous one, 2) There needs to be a gigantic chunk of memory allocated. The solution for the first problem is prehashing (assign any object to an integer), and to the second problem: chaining and open addressing.
Say we want to fit a hash table of size m. Hashing functions (examples) are:
1. Division method: $h(k) = k \mod m$
2. Multiplication method: $h(k) = [a * k \mod 2^w] >> (w-r)$, where $w$ is a number of bits of the word and $a$ is an odd number.
3. Universal hashing: $h(k) = [(ak+b) \mod p] \mod m$ where $p$ is a prime number bigger than size of the key universe, and $a, b$ are random numbers smaller than $p$

SUHA (Simple Uniform Hashing Assumption) - states that a hypothetical hashing function will evenly distribute items into the slots of a hash table.
In mathematics and computing universal hashing (in a randomised algorithm or data structure) refers to selecting a hash function at random from a family of hash functions with a certain mathematical property.


## Lecture 9: Table Doubling, Karp-Rabin
To have a proper hash table, $m$ (size of hash table) must be $\Omega(n)$  to ensure the load factor $\alpha$ to be constant (i.e. $n$ is of order $m$, where $n$ is the number of keys), and $O(n)$ to ensure memory is not wasted. This gives the requirement for $m$ to be $O(n)$.
Table Doubling is used to ensure amortised linear time per insert - the hash table is allocated twice as big when it gets filled with keys. Table shrinking is used for deletion - when the number of keys drops to the quarter of the original size, the table shrinks by a factor of two.  

A rolling hash (also known as recursive hashing or rolling checksum) is a hash function where the input is hashed in a window that moves through the input.

Karp-Rabin string search algorithm uses a rolling hash to find if a string $s$ appears somewhere in a big text document $t$. Naively we might slide a window of size $len(s)$ through $t$ and compare each letter of $a$, but that would be $O(|s||t|)$. Karp-Rabin is $O(|s|+|t|)$.


## Lecture 10: Open Addressing, Cryptographic Hashing
Open Addressing is another way to avoid collisions (the other one was chaining). Here, a hash function specifies an order of slots to try for insertion/search/deletion. Apart from key, it also takes as an argument the trial count.

Permutation property ensures that for every key and every slot there exists a trial count for which a hash function will output a particular slot. So we will always be able to insert the key after a finite number of trials.t

After deleting a key from the hash table we should mark the slot not as None (or empty), but with a different flag, i.e. Deleted. Otherwise searching for another key might fail after encountering an empty (deleted) slot. 

Probing Strategies:
* Linear probing (the problem is clustering - groups of occupied slots) 
for $0.01 < α = n/m < 0.99$ there are clusters of size $O(logn)$
* Double hashing $h(k,i) = (h_1(k,i) + i*h_2(k,i)) \mod m$
work if $h_2(k)$ is relatively prime to $m$.

Simple Uniform Hashing - Independence of keys in terms of their mapping to slots
Uniform Hashing Assumption - Each key is equally likely to have any of the $m!$ permutations as its probe sequence. It ensures that the cost of operations is $\leq 1/(1-\alpha)$


## Lecture 11: Integer Arithmetic, Karatsuba Multiplication
Newton's method - method to approximate the root of function $f(x)$ by computing a tangent to the function at given $x_i$ and finding the intersection of that tangent line with the x-axis to get $x_{i+1}$.
We can use Newton's method to compute irrational numbers, e.g. $\sqrt{2}$, by finding a root of $f(x)=x^2-2$.

Linear convergence - at each iteration you get one digit of precision. In quadratic convergence, the number of digits that are correct, doubles at each iteration. Newton's method has quadratic convergence.

### Multiplication
In multiplication if you want to multiply two $n$-digit numbers, you divide them into $n/2$ digit numbers and perform
$T(n) = 4T(n/2) + O(n)$ operations (you divide and conquer until $n=64$, since machines can do this multiplication with one instruction). The complexity is therefore $O(n^2)$.
Karatsuba algorithm requires three multiplications, yielding the recurrence formula of $T(n) = 3T(n/2) + O(n)$, which gives the complexity of 
$O(n^{log3/log2}) = O(n^{1.585})$.

## Lecture 12: Square Roots, Newton's Method
Error analysis for the Newton's method shows that the error term at each iteration is a quadratic function of the error term in the previous iteration. This is the reason why the algorithm has a quadratic convergence. 

Toom-Cook generalised Karatsuba to dividing problem into $n/3$, $n/4$ parts etc. - we call those Toom-3 etc.
Toom-3 reduces 9 multiplications to 5, and runs in $\Theta(n^{log5/log3})$, about $\Theta(n^{1.465})$
The recurrence relationship is therefore
$T(n) = 5T(n/3) + O(n)$
For those methods though, the constants involved are quite big and for small numbers, algorithms with higher complexity suit better. 

Schönhage–Strassen algorithm is even quicker (for sufficiently large integers) and runs in $O(n \times logn \times loglogn)$ using Fast Fourier Transforms (FFT).

Fürer's (2007) algorithm has a complexity of $O(n \times logn \times 2^{O(log^{*}n)})$ where $log^{*}n$ is an iterated logarithm (number of times the log needs to be applied to get a result that is less than or equal to 1). This algorithm beats Schönhage–Strassen for integers greater than $2^{2^{64}}$.

If we use $O(n^{\alpha})$ multiplication algorithm, the division with $d$ digits of precision will be $O(lgd \times d^{\alpha})$ with careless analysis. However, the first iteration is going to be $d=1$ digits precise with the cost of $1^\alpha$, second one with the cost of $2^\alpha$ , third one with $4^\alpha$ and the last one is going to cost $d^\alpha$, but that geometric sum is bounded by $2d^\alpha$, so the division is essentially the same complexity as multiplication.

The complexity of computing square roots is the same as the complexity of computing divisions by the same argument.


## Lecture 13: Breadth-First Search (BFS)
The diameter of a graph is the maximum eccentricity of any vertex in the graph. That is, it is the greatest distance between any pair of vertices. To find the diameter of a graph, first find the shortest path between each pair of vertices. The greatest length of any of these paths is the diameter of the graph. 
Graph representation:
* Adjacency lists. Memory: $O(V + E)$
* Adjacency matrix. Memory: $O(V^2)$

If we add a parent to for each vertex and represent a parenthood as a directed edge, the edges form a tree and those edges are the shortest paths from the root element to each vertex. The running time is $O(V + E)$.
BFS is used to find the shortest paths.

## Lecture 14: Depth-First Search (DFS), Topological Sort
DFS is used for finding cycles in graphs and for topological sort.
Recursively explore graph, backtracking as necessary.
Time complexity: $O(V + E)$
DFS can be used to find tree edges: edges that led to the discovery of a new vertex. We can distinguish between different types of edges (in a directed graph):
* forward edge - edge from the node to its descendant
* backward edge - from the node to its ancestor
* cross edge - other edge (connecting different subtrees in a graph)

Tree edges and cross edges might exist in a undirected graph. 
Cycle detection: A graph has a cycle iff DFS has a backward edge.

In DFS we start a vertex and visit each vertex reachable from it. Due to recursion if we started from $V_0$ and reach  $V_k$  will will first visit all edges from $V_k$ before finishing $V_0$, which is sometimes called balancing bracket. 

Topological sort: given DAG order the vertices so that the edges point from lower order to higher order. Algorithm:
Run DFS and output the finishing times (count) of vertices, reversed.

## Lecture 15: Single-Source Shortest Paths Problem
Using BFS is not suitable when the graphs are weighted, i.e. each edge is associated with a weight. For that purpose the most popular algorithms are:
1. Dijkstra (non-negative weights): $O(VlogvV + E)$  and we know that $E=O(V^2)$ (i.e. in a complete graph there is an edge between every two vertices). 
2.  Bellman-Ford (positive and negative weights): $O(VE)$, which can be $O(V^3)$ in a strongly connected graph.

DFS/BFS might not necessarily be applicable to the shortest path problem since it does not address the fact that weights might have different values - their range is dynamic; not known up front.

General structure of shortest paths algorithms (no negative cycles):
```
Initialise u∈V d[v] = ∞ and Π[v] = None
    d[s] = 0
    Repeat until all edges have d[v] <= d[u] + w(u,v) (they cannot be relaxed):
        Select edge (u,v):
        Relax edge (u,v): if d[v] > d[u] + w(u,v):
            Update d[v] = d[u] + w(u,v)
            Set parent Π[v] = u
```
where $s$ is a source vertex, $\Pi$ is a parent of a vertex (predecessor), $w$ is a weight, $d$ is a distance to a vertex.

**Relaxation** means updating the distance to some vertex $v$ if you discover that the distance via some $u$ is smaller than your current *tentative distance* to $d$, $d[v]$, i.e. when $d[v] > d[u] + w(u,v)$

Optimal substructure:
1. Subpaths of the shortest paths are shortest paths
2. Triangle inequality holds for deltas (shortest distance, not the distance!): δ(u,v) <= δ(u,x) + δ(x,v) 


## Lecture 16: Dijkstra
Dijkstra computes shortest paths in a graph with cycles but does not deal with negative edges.

**Review:**
- $d[v]$ is the current length of shortest path from source $s$ to $v$. The act of improving $d$ is called relaxation. - - Relaxation step not only updates the distance value but also predecessor relationship.
- $\delta(s,v)$ is the shortest path from $s$ to $v$ (might not be unique).
- $\pi[v]$ is the predecessor of $v$ (parent).

Shortest path in DAGs:
1. Sort vertices in topological order $O(V+E)$.
2. Do one pass through the vertices in topologically sorted order relaxing each edge that leaves each vertex.
Total time: $O(V+E)$.

Dijkstra works for graphs with cycles but there cannot be negative edges. It is a greedy algorithm as it, at each step, takes a vertex from the priority queue which the smallest value of $d$. As a priority queue we should use Fibonacci heap that has $\Theta(1)$ for extract-min and $\Theta(1)$ amortised for decrease key (this operation is important because Dijkstra needs to relax i.e. decrease distances). This allows Dijkstra to be 
$\Theta(VlogV + E)$, cause we need to extract-min value of $d$ for each vertex ($V$ times) and update each edge $E$ times (in constant time).

## Lecture 17: Bellman-Ford
Bellman-Ford computes shortest paths in a graph and deals with negative cycles.
Its complexity is $O(VE)$, because we run the outer loop $V-1$ times and each time we're relaxing $E$ edges. After the outer loop is done we might try to relax the edges once again - if the distance is improved for one of the edges this indicates the existence of a negative cycle.

Finding a shortest simple path in a graph with negative cycles is NP-hard and is equivalent (with respect to complexity) to the longest path problem.


## Lecture 18: Speeding up Dijkstra
Dijkstra outputs the shortest path from the source $s$ to every other vertex reachable from s in the graph. The algorithm can be sped up by early termination if we are only interested in finding the shortest path from a particular source $s$ to target $t$.
There is also a method to find the shortest path between $s$ and $t$ by using Bi-directional search, at which we keep two min queues: $Q_f$ (forward queue) and $Q_b$ (backward queue) and we alternate by exploring a forward frontier from $s$ and backward frontier from $t$. Once frontiers meet, find a vertex $x$ that minimises the sum of forward and backward distance and use the predecessor relationship $\Pi_f$ and $\Pi_b$ to find the path.

Goal-directed search is the method to improve the average (empirical) time of Dijkstra, in which we change the weights of the graph according to some potential function $\lambda$ to direct the exploration towards some specific part of the graph (make some weights slightly bigger, a.k.a uphill and some of them downhill). A good choice of λ is based on choosing an appropriate landmark l (vertex we know we need to go through) and precomputing shortest distances from all the points in the graph to this landmark l . 

## Lecture 19: Dynamic Programming I: Fibonacci, Shortest Paths
DP requirements:
1. Optimal substructure (optimal solution can be constructed from optimal solutions of its subproblems)
2. Overlapping subproblems (problem can be broken down into subproblems which are reused several times)


DP ≈ "careful brute force"
DP ≈ "memoization" + "recursion"

Memoization is a technique in DP where we store the solutions to subproblems in a dictionary/array.
You can solve DP problems using top-down (memoized) and bottom-up approach. In bottom-up we are essentially doing a topological sort of the subproblem dependency DAG (i.e. in order to compute $F_{n}$ we need $F_{n-1}, F_{n-2}$). Bottom-up approach can save space, e.g. in fibonacci problem you only need to store the last two numbers to compute the next, whereas in top-down you need to store all the fibonacci numbers up to n. 

- For memoization to work the subproblem dependencies should be acyclic

#### Shortest paths
Definition: Find shortest path $\delta_k(s, v)$ where $\delta_k$ means shortest path that uses at most $k$ edges.
Solution: Guess last edge from some $u$ to $v$ (try all of them, there are $indegree(v)$ incoming edges to $v$). Thus we have: 

$\delta_k(s, v)=\min{\delta_{k-1}(s,u) + w(u,v)}|(u,v)\in E$

i.e. we introduce a new node $u$ and find a solution to a suproblem, i.e. min path from $s$ to $u$.
For a vertex, the number of head ends adjacent to a vertex is called the indegree of the vertex and the number of tail ends adjacent to a vertex is its outdegree (called "branching factor" in trees).

## Lecture 20:  Dynamic Programming II: Text Justification, Blackjack
5 easy steps to DP:
1. Define subproblems.
2. Guess (part of a solution). In Fibonacci example we did not need to guess anything. In case of shortest paths problem we guess the last edge 
3. Relate subproblem solutions.
4. Recurse & memoize (top-down) or build DP table (bottom-up). Usually bottom-up will be faster in practice (can save memory).
```
time = num subproblems * time/subproblem
```
5. Solve original problem (combine solutions to subproblems)
Usually, for the bottom-up approach (building a table), you will either use min or max.
Every DP is associated with a DAG - the subproblems recurrence is acyclic.

- Use *parent pointers* to remember the best guesses in the DP. Otherwise we find the cost of the best solution but not the best solution itself. E.g. in shortest paths we need to keep pointers to parent nodes to find the shortest path, otherwise we will find the cost of the shortest path instead.

## Lecture 21: Dynamic Programming III: Parenthesization, Edit Distance, Knapsack
DP problems for strings/sequences can roughly fall into three flavours:
- Prefix problems `x[:i]`
- Suffix problems `x[i:]` (e.g. text justification, you guess the prefix and then recur on the suffix)
- Substring problems `x[i:j]`

### Parenthesization
Problem: Say we need to compute a product of matrices: $\prod_{i} A_i$, the complexity of the operation will depend on how we put the parentheses (in what order we evaluate the multiplication). 

### Edit distance
Problem: Given two strings `x` and `y` what's the cheapest possible sequence of character edits (insertion, deletion, replacement) to turn `x` into `y`? An example of edit distance problem is **longest common subsequence**. 


### Knapsack
Problem: Maximise sum of values $v_i$ of $i$ items of size $s_i$ within the knapsack of size $S$
Solution: Form a sequence out of all the items: $i_1, ... i_n$
1. Subproblem - suffix of `i:` of items & remaining capacity 
2. Guessing - do we pack item `i` or not?
3. Recurrence - $DP(i) = \max{(DP(i+1, X), DP(i+1, X-s_i)+v_i)}$ where $X$ is our current remaining capacity

Running time: $O(n*S)$ - Not polynomial time! This is because input size is $O(nlogS)$ and $S$ is exponential in $logS$. That's why we call it pseudopolynomial time - it's polynomial in $n$ (input size) and in the numbers that are in the input (so it's kind of inbetween polynomial and exponential). It's the best you can do in Knapsack. 


## Lecture 22: Dynamic Programming IV: Guitar Fingering, Tetris, Super Mario Bros
## Lecture 23: Computational Complexity
### P, EXP, R
- $P$ - set of problems solvable in polynomial time
- $EXP$ - set of problems solvable in exponential time: $c^{poly(n)}$, where $c$ is constant
- $R$ - set of problems solvable in finite time


$P \subset EXP \subset R$
e.g. *halting problem* is not in $R$

**Decision problem** - problem with answer Yes/No

### Most decision problems are uncomputable
- program (computer program or algorithm) is just a binary string in the end (in memory) which we can then treat as a (huge) integer
- decision problem is a function $f$ that maps inputs to $\{{YES, NO}\}$. We can build a table representation of this function (i.e. for each input the answer `YES` or `NO`) and so this function is just an infinite string of bits

*Decision problem is an infinite string of bits (i.e. $\in \R$), whereas a program is a finite string of bits (i.e. $\in \N$). Since $|\R| >> |\N|$*

### NP
NP - set of decision problems solvable in polynomial time via a "lucky" algorithm (i.e. nondeterministic model) which makes "guesses" (guesses are guaranteed to a `YES` answer if possible). It's polynomial time as long as we need to make a polynomial number of guesses.

In another words, if there is any way to find a `YES` to an answer, those guesses will find it.

Alternative definition:
NP - set of decision problems with *solutions* that can be *checked* in polynomial times
i.e. when the answer is `YES` you can prove it and check the proof in polynomial time
i.e. generating (proofs of) solutions is harder than checking them


#### NP-hard 
NP-hard - at least as hard as every problem in NP. This means that any problem that is not in NP is NP-hard (but we can have a problem that is both NP and NP-hard!)

#### NP-complete
NP-complete - $NP \cap NP-hard$, i.e. intersection of NP and NP-hard

### Computational *difficulty* line

If we imagine a horizontal line of computational dificulty, i.e. line $[O, \infty]$ (where $O$ means origin) and three points $A, B, C$ such that $C \geq B \geq A$: 
- problems in $[O, A]$ are $P$
- problems in $[O, B]$ are $NP$
- problems in $[B, \infty]$ are NP-hard
- problems in $B$ (intersection) are NP-complete
- problems in $[O, C]$ are $EXP$
- problems in $[C, \infty]$ are $EXP-hard$
- problems in $C$ are EXP-complete (e.g. Chess)

We don't know whether $NP = EXP$


### Reductions
Reduction - convert problem A which you don't know how to solve into problem B that you know how to solve.

e.g. you can solve a problem to find the shortest paths in a unweighted graph using Dijkstra after assigning all weights in the graph to 1 (although you'd be faster with BFS).

If you can reduce A to B this means that B is **at least as hard as A**


## Lecture 24: Topics in Algorithms Research
SRAM - Static RAM - RAM on the chip with ultra fast access from the processor (cores)
DRAM - Dynamic RAM - RAM off the chip (order of magnitude less speed)

- problem with SRAM is simultaneous access for more cores (~10) is difficult
- it's usually 1 clock cycle to access local memory (L1 cache)
- accessing cache on a different side of the chip can take multiple of clock cycles (and it's a two-way trip)

**Execution migration** is about sending out program/code (context) from one processor to the other (instead of requesting data). The advantage of that is that it's a one-way trip which saves clock cycles. We need to decide when we want to do data vs. execution migration


**Cache-oblivious algorithm** (or cache-transcendent algorithm) is an algorithm designed to take advantage of a CPU cache without having the size of the cache (or the length of the cache lines, etc.) as an explicit parameter, i.e. algorithm can perform efficient search/sort on blocks of memory $B$ (taken from DRAM) fetched to CPU cache (block size $C$) without knowing $B$ or $C$.

Suppose you want to support *insert, delete, predecessor, successor* for $n$ integers $\in U$ (universe) in a word RAM (BSTs do this in *average* $O(logn)$). The optimal approach is to take a min of $O(lglgu)$ (Van Emde Boas Trees) or $O(\frac{lgn}{lglgU})$ (Fusion trees), which gives $O(\sqrt{\frac{lgn}{lglgn}})$

### Planar graphs
> Planar graph is a graph that can be embedded in the plane, i.e., it can be drawn on the plane in such a way that its edges intersect only at their endpoints. In other words, it can be drawn in such a way that no edges cross each other

For those graphs you can do:
- Dijkstra in $O(V)$
- Bellman-Ford in $O(n\frac{lg^2n}{lglgn})$

### Recreational algorithms
- Rubik's cube in $O(\frac{n^2}{logn})$
