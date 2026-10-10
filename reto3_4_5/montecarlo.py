import random

def single_montecarlo_simulation(edges, initial_population, initial_state, N, s):
    """
    Creates N iterations of fire spreads, each with s stages, from an initial state. The probabilities of fire transition are given by edges.

    Returns:
        active: dictionary of the form {(iteration, stage): [nodes]}
    """

    active = {} # active = {(iteration, stage): [nodes]}

    for iteration in range(N):
        population = set(initial_population.copy())
    
        for node in initial_state:
            population.discard(node)

        active[(iteration, 0)] = set(initial_state)

        for t in range(1, s):
            # Active nodes at past stage
            a = active[(iteration, t-1)]
            # Nodes that will be active at current stage
            active[(iteration, t)] = set()

            for i in a:
                for j in population: # We avoid trying to infect already infected nodes
                    # Montecarlo simulation
                    if random.random() < edges.get((i, j), 0):
                            # j becomes active in the next stage
                            active[(iteration, t)].add(j) # Since we are using sets, we don't add them more than once
        
            # Nodes that have become active are removed from population
            population = population.difference(active[(iteration, t)])

    return active

def fixed_montecarlo_simulation(edges, initial_population, initial_state, N, s, seed):
    """
    Creates N iterations of fire spreads, each with s stages, from an initial state.
    The probabilities of fire transition are given by edges.

    Returns:
        active: dictionary of the form {(iteration, stage): [nodes]}
    """
    # Pre-generate random numbers: R[node_i][node_j][stage]
    # Fixed for all iterations and independent of edges.
    random.seed(seed)
    R = {
        (i, j, t): random.random()
        for i in initial_population
        for j in initial_population
        for t in range(1, s)
    }

    active = {}

    for iteration in range(N):
        population = set(initial_population)
        for node in initial_state:
            population.discard(node)

        active[(iteration, 0)] = set(initial_state)

        for t in range(1, s):
            a = active[(iteration, t - 1)]
            active[(iteration, t)] = set()

            for i in a:
                for j in population:
                    if R[(i, j, t)] < edges.get((i, j), 0):
                        active[(iteration, t)].add(j)

            population = population.difference(active[(iteration, t)])

    return active

def montecarlo_simulation(edges, possibilities, p, N, s):
    """
    Creates N iterations of fire spreads, each with s stages, from a random initial node chosen from possibilities. The probabilities of fire transition are given by edges.

    Returns:
        active: dictionary of the form {(iteration, stage): [nodes]}
    """
    # Initial states
    initial_states = random.choices(possibilities, weights=p, k=N)
    
    active = {} # active = {(iteration, stage): [nodes]}

    for (iteration, node) in enumerate(initial_states):
        population = set(possibilities)
        population.discard(node)

        active[(iteration, 0)] = set([node])

        for t in range(1, s):
            # Active nodes at past stage
            a = active[(iteration, t-1)]
            # Nodes that will be active at current stage
            active[(iteration, t)] = set()

            for i in a:
                for j in population: # We avoid trying to infect already infected nodes
                    # Montecarlo simulation
                    if random.random() < edges.get((i, j), 0): 
                        # j becomes active in the next stage
                        active[(iteration, t)].add(j) # Since we are using sets, we don't add them more than once
        
            # Nodes that have become active are removed from population
            population = population.difference(active[(iteration, t)])

    return active


def expected_burnt_zones(active, N, s):
    """
    From a given dictionary of active zones in N simulations each with s stages, return the mean burnt zones in s stages.

    Returns:
        E = expected burnt zones in s stages
    """
    total = 0
    for iteration in range(N):
        for stage in range(s):
            total += len(active[(iteration, stage)])
    E = total / N
    return E

if __name__ == "__main__":
    import os
    from io_functions import read_graph, read_start, read_graph_as_binary

    os.makedirs("tests", exist_ok=True)

    random.seed(3)

    N = 10000 # Number of iterations per initial conditions
    s = 3 # Number of stages of simulation (s=1 is only t=0)

    # Probabilities (must sum to 1)
    p = read_start("Datos/inicio.txt")
    edges, _ = read_graph_as_binary("Datos/graph_003_probs.txt")

    population = list(range(len(p)))

    initial_states = random.choices(population, weights=p, k=N)

    active = montecarlo_simulation(edges, population, p, N, s)

    # for iteration in range(N):
    #     print("--Iteration--")
    #     for tt in range(s):
    #         print(f"Activas en etapa {tt}: ", active[(iteration, tt)])

    E = expected_burnt_zones(active, N, s)
    print("Expected burnt zones: ", E)

    # single = single_montecarlo_simulation(edges, set
    # (population), [13], N, s)

    # E2 = expected_burnt_zones(single, N, s)
    # print("Expected burnt zones 2: ", E2)

    # print(sorted(active.items()), "\n", sorted(single.items()))

    from plot_functions import draw_evolution

    if True:
        draw_evolution(edges, active, 0, s, "tests/montecarlo_test.png")
        
    if False:
        draw_evolution(edges, single, 0, s, "tests/single_montecarlo_test1.png")
        draw_evolution(edges, single, 1, s, "tests/single_montecarlo_test2.png")
        draw_evolution(edges, single, 2, s, "tests/single_montecarlo_test3.png")
