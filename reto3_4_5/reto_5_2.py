import os, glob, random
from io_functions import read_graph, read_start, remove_edges
from montecarlo import montecarlo_simulation
from plot_functions import draw_evolution

def weighted_burnt_zones(active, weights, N):
    """
    From a given dictionary of active zones in N simulations each with s stages, return the weight of burnt zones in s stages.

    Returns:
        E = expected burnt zones in s stages
    """
    total = 0
    for iteration in range(N):
        for stage in range(s):
            for node in active[(iteration, stage)]:
                total += weights[node]
    E = total / N
    return E

def greedy_edge_removals_weights(edges, p0, weights, k, s, N=1000):
    """
    Greedily remove k undirected edges, each time choosing the one that minimizes
    E[|Q_s|] estimated using a Montecarlo simulation with N iterations.
    """
    reduced = edges.copy()
    removed = []

    for _ in range(k):
        undirected = sorted({(min(i, j), max(i, j)) for (i, j) in reduced.keys()})

        best_value = float("inf")
        best_edge = None

        for (u, v) in undirected:
            trial = reduced.copy()
            trial.pop((u, v), None)
            trial.pop((v, u), None)

            active = montecarlo_simulation(trial, list(range(len(p0))), p0, N, s)

            value = weighted_burnt_zones(active, weights, N)
            if value < best_value:
                best_value = value
                best_edge = (u, v)

        reduced.pop(best_edge, None)
        reduced.pop((best_edge[1], best_edge[0]), None) # type: ignore
        removed.append(best_edge)

    return removed, best_value # type: ignore

os.makedirs("results_reto_5", exist_ok=True)

random.seed(4)

N = 1000  # Number of iterations per initial condition
s = 5     # Number of stages
k = 4     # Number of firewalls

# Read start probabilities once
p0 = read_start("Datos/inicio.txt")
population = list(range(len(p0)))

# Weights of each node
weights = {node: random.uniform(0.2, 1) for node in population}

# Find all graph files
graph_files = sorted(glob.glob("Datos/graph_*"))

# Clear results file and write results
with open("results_reto_5/resultados_5_2.txt", "w") as f:
    for graph_path in graph_files:
        graph_name = os.path.basename(graph_path)
        
        edges, _ = read_graph(graph_path)
        
        active = montecarlo_simulation(edges, list(range(len(p0))), p0, N, s)
        value = weighted_burnt_zones(active, weights, N)
        f.write(f"{graph_name}\tE[|Q_{s}|] = {value}\n")

        best_removals, best_value = greedy_edge_removals_weights(edges, p0, weights, k, s, N)

        f.write(f"Greedy minimum E[|Q_{s}|] = {best_value}")
        f.write("\nRemove:")
        for combo in best_removals:
            f.write(f" {combo}")
        f.write("\n")

best_removals, best_value = greedy_edge_removals_weights(edges, p0, weights, k, s, N)

reduced = remove_edges(edges, best_removals)

active = montecarlo_simulation(edges, list(range(len(p0))), p0, 2, s)
active_reduced = montecarlo_simulation(reduced, list(range(len(p0))), p0, 2, s)

draw_evolution(edges, active, 0, s, f"results_reto_5/{graph_name}_extra.png")
draw_evolution(reduced, active_reduced, 0, s, f"results_reto_5/{graph_name}_extra_optimo.png")

