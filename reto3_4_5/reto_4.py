import os, glob, random
from io_functions import read_graph, read_start, set_zero_edges, remove_edges
from montecarlo import single_montecarlo_simulation, fixed_montecarlo_simulation, expected_burnt_zones
from plot_functions import draw_evolution
from reto_3 import greedy_edge_removals_general

def simulation_with_alternating_greedy_v2(edges, population, initial_state, k, N, s, N_greedy, s_greedy):
    """
    Simulate N iterations of fire evolution for s stages with alternating greedy edge removals.
    
    Fire starts at initial_state, then:
    1. Use greedy_edge_removals_general to place k firewalls
    2. Simulate fire evolution for 1 steps using Montecarlo
    3. From the active fires after evolution, use greedy_edge_removals_general again
    4. Repeat until s is reached
    
    Returns mean burnt zones across all N iterations.
    """
    total_burnt = 0
    active = {}
    
    for iteration in range(N):
        # Start fresh each iteration
        current_edges = edges.copy()
        current_population = set(population.copy())

        active[(iteration, 0)] = set(initial_state)

        for t in range(1, s):
            # Step 1: Place firewalls using greedy algorithm
            removed, _ = greedy_edge_removals_general(
                current_edges, population, active[(iteration, t-1)], k, N_greedy, s_greedy
            )
            # if iteration == 0: # Para ver qué aristas vamos cortando
            #     print(removed)
            # Apply the removals
            current_edges = set_zero_edges(current_edges, removed)

            # Step 2: Simulate fire evolution for 1 step in both graphs
            active_after = single_montecarlo_simulation(
                current_edges, current_population, active[(iteration, t-1)], N=1, s=2
            )

            # Remove active fires from population
            current_population = current_population.difference(active[(iteration, t-1)])

            # Add active fires to burnt
            total_burnt += len(active[(iteration, t-1)])

            if not active_after[(0,1)]: # If there are no more active fires the fire stops
                break

            active[(iteration, t)] = active_after[(0,1)]
            
    mean_burnt = total_burnt / N
    return active, mean_burnt

# Main execution
if __name__ == "__main__":
    random.seed(1)
    
    os.makedirs("results_reto_4", exist_ok=True)
    os.makedirs("results_reto_4/simulaciones", exist_ok=True)


    # Parameters
    N = 50        # Number of Montecarlo iterations for the full simulation
    s = 5         # Number of steps for each iteration
    N_greedy = 20
    s_greedy = 3
    k_firewalls = 1  # Number of firewalls per greedy step

    # Find all graph files
    graph_files = sorted(glob.glob("Datos/graph_*"))

    # Same for all graphs
    population = list(range(20))

    # Clear results file and write results
    with open("results_reto_4/resultados.txt", "w") as f:
        for graph_path in graph_files:
            graph_name = os.path.basename(graph_path)

            # Load data
            edges, n = read_graph(graph_path)

            best_initial_node = None
            best_mean_burnt = float("-inf")

            # Try every possible initial node for s steps
            for candidate in range(n):
                initial_state = [candidate]
                active, mean_burnt = simulation_with_alternating_greedy_v2(
                    edges, population, initial_state, k_firewalls, N, s, N_greedy, s_greedy
                )
                if mean_burnt > best_mean_burnt:
                    best_mean_burnt = mean_burnt
                    best_initial_node = candidate

            # Simulate Reto 3 but with best initial node
            active, mean_burnt = simulation_with_alternating_greedy_v2(
                edges, population, [best_initial_node], k_firewalls, N, 20, N_greedy, s_greedy
            )

            draw_evolution(edges, active, 2, 20, f"results_reto_4/simulaciones/{graph_name}_iteration_{2}.png")

            f.write(f"{graph_name}\t best initial node = {best_initial_node}\t "
                    f"mean burnt = {mean_burnt}\n")

    
    # Optional: visualize final state
    # for i in range(0,10):
    #     draw_evolution(edges, active, i, 20, f"results_reto_4/simulaciones/{graph_name}_iteration_{i}.png")
