[AIMA](https://people.engr.tamu.edu/guni/csce421/files/AI_Russell_Norvig.pdf)


## 2.10
Consider a modified version of the vacuum environment in Exercise 2.8, in which the agent is penalized one point for each movement.

### a. Can a simple reflex agent be perfectly rational for this environment? Explain.

Yes, given the case it's an even-sided square environment, the agent can move in a zigzag pattern, cleaning the whole environment without ever moving to a dirty square twice.

However, this assumes that time stops when the last square is cleaned.

Otherwise, no.

Example Zig-Zag Pattern for N=6
```
v < < < < <
v > > > > ^
v ^ < < < <
v > > > > ^
v ^ < < < <
> > > > > ^
```

### b. What about a reflex agent with state? Design such an agent.

Yes, the agent can keep track of the squares it has visited, and the squares it has cleaned.


If the grid is even-sided:
1. If the agent is in a dirty square, it cleans it.
2. If the agent is in a clean square, it moves to the next square in the zigzag pattern.
3. If the agent is in a square it has already visited, it halts.
Otherwise:

1. If the agent is in a dirty square, it cleans it.
2. If the agent is in a clean square, it moves to the next square in the zigzag pattern.
3. If the agent is in a square it has already visited, it makes sure no unvisited squares remain.
4. If there are unvisited squares, it visits them
5. If there are no unvisited squares, it halts.

### c. How do your answers to a and b change if the agent’s percepts give it the clean/dirty status of every square in the environment?

a: Then the problem becomes like a traveling salesman problem, but there exists an optimal solution (Even if it takes ages to compute).
1. If the agent is in a dirty square, it cleans it.
2. The agent can compute the optimal path to clean all the squares from it's current location.
3. The agent can move to the next square in the optimal path.

b: The algorithm described in a would still work, but it would be more efficient because we only need to compute the optimal path once.

## 2.11
Consider a modified version of the vacuum environment in Exercise 2.8, in which the geography of the environment—its extent, boundaries, and obstacles—is unknown, as is the initial dirt configuration. (The agent can go Up and Down as well as Left and Right.)
### a. Can a simple reflex agent be perfectly rational for this environment? Explain.
No. The agent can't know if it has cleaned the whole environment, or if it's stuck in a corner.
### b. Can a simple reflex agent with a randomized agent function outperform a simple reflex agent? Design such an agent and measure its performance on several environments.
Yes, the agent can move randomly until it finds a dirty square, then clean it, and repeat until it has cleaned the whole environment.
### c. Can you design an environment in which your randomized agent will perform poorly? Show your results.
Yes, if the environment has the following shape

```
x x x x x x x - x x x x x x x
x x x x x x x - x x x x x x x
x x x x x x x - x x x x x x x
x a x x x x x x x x x x d x x
x x x x x x x - x x x x x x x
x x x x x x x - x x x x x x x
x x x x x x x - x x x x x x x
```
legend:
- x: clean square
- -: wall
- a: agent
- d: dirty square

If we expand the two zones significantly, or even, add multiple zones, the agent will have a hard time finding the dirty square.

### d. Can a reflex agent with state outperform a simple reflex agent? Design such an agent and measure its performance on several environments. Can you design a rational agent of this type?
Yes, the agent can keep track of the squares it has visited, and the squares it has cleaned.
