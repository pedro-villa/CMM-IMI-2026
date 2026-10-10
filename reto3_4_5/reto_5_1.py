import os, glob, random
from io_functions import read_graph, read_start, set_zero_edges, remove_edges
from montecarlo import single_montecarlo_simulation, fixed_montecarlo_simulation, expected_burnt_zones
from plot_functions import draw_evolution

def greedy_edge_removals_general(edges, population, initial_state, k, N, s):
    """
    Greedily remove k undirected edges, each time choosing the one that minimizes
    E[|Q_s|] estimated using a Montecarlo simulation with N iterations. Initial state is fixed.
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

            active = single_montecarlo_simulation(trial, list(population), initial_state, N, s)

            value = expected_burnt_zones(active, N, s)
            if value < best_value:
                best_value = value
                best_edge = (u, v)

        reduced.pop(best_edge, None)
        reduced.pop((best_edge[1], best_edge[0]), None) # type: ignore
        removed.append(best_edge)

    return removed, best_value # type: ignore    

def add_noise_to_probabilities(prob_dict, noise_range=(-0.1, 0.1)):
    """
    Add random noise to probabilities in a dictionary while keeping them in [0.0, 1.0].
    
    Args:
        prob_dict: Dictionary with (i,j) tuples as keys and probabilities as values
        noise_range: Tuple of (min_noise, max_noise) for uniform random noise
    
    Returns:
        New dictionary with noisy probabilities
    """
    noisy_dict = {}
    min_noise, max_noise = noise_range
    
    for key, prob in prob_dict.items():
        # Generate random noise
        noise = random.uniform(min_noise, max_noise)
        
        # Add noise and clamp to [0.0, 1.0]
        noisy_prob = prob + noise
        noisy_prob = max(0.0, min(1.0, noisy_prob))  # Clamp to valid range
        
        noisy_dict[key] = noisy_prob
    
    return noisy_dict

def simulation_with_alternating_greedy_noise(edges, population, initial_state, k, N, s, N_greedy, s_greedy, seed):
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
    total_burnt_defenseless = 0
    active = {}
    active_defenseless = {}
    
    for iteration in range(N):
        # Start fresh each iteration
        current_edges = edges.copy()
        current_population = set(population.copy())
        current_population_defenseless = set(population.copy())

        
        active[(iteration, 0)] = set(initial_state)
        active_defenseless[(iteration, 0)] = set(initial_state)


        for t in range(1, s):
            # Probabilities vary randomly
            current_edges = add_noise_to_probabilities(current_edges)

            # Step 1: Place firewalls using greedy algorithm
            removed, _ = greedy_edge_removals_general(
                current_edges, population, active[(iteration, t-1)], k, N_greedy, s_greedy
            )
            # if iteration == 0: # Para ver qué aristas vamos cortando
            #     print(removed)
            # Apply the removals
            current_edges = set_zero_edges(current_edges, removed)

            # Step 2: Simulate fire evolution for 1 step in both graphs
            active_after = fixed_montecarlo_simulation(
                current_edges, current_population, active[(iteration, t-1)], N=1, s=2,
                seed=iteration*seed # Makes each iteration different 
            )
            active_after_defenseless = fixed_montecarlo_simulation(
                edges, current_population_defenseless, active_defenseless[(iteration, t-1)], N=1, s=2,
                seed=iteration*seed # Makes each iteration different
            )

            # Remove active fires from population
            current_population = current_population.difference(active[(iteration, t-1)])
            current_population_defenseless = current_population_defenseless.difference(active_defenseless[(iteration, t-1)])

            # Add active fires to burnt
            total_burnt += len(active[(iteration, t-1)])
            total_burnt_defenseless += len(active_defenseless[(iteration, t-1)])


            if not active_after[(0,1)] and not active_after_defenseless[(0,1)]: # If there are no more active fires the fire stops
                break

            active[(iteration, t)] = active_after[(0,1)]
            active_defenseless[(iteration, t)] = active_after_defenseless[(0,1)]
            
    mean_saved = (total_burnt_defenseless - total_burnt) / N
    return active, active_defenseless, mean_saved

# Main execution
if __name__ == "__main__":
    os.makedirs("results_reto_5", exist_ok=True)
    os.makedirs("results_reto_5/simulaciones", exist_ok=True)

    seed = 1
    
    # Parameters
    N = 100        # Number of Montecarlo iterations for the full simulation
    s = 20           # Number of steps for each iteration, if >= n we are sure the fire will stop
    N_greedy = 50
    s_greedy = 3
    k_firewalls = 1  # Number of firewalls per greedy step

    # Find all graph files
    graph_files = sorted(glob.glob("Datos/graph_010*"))

    # Same for all graphs
    initial_state = [1]  # Fire starts at node 1
    population = list(range(20))

    # Clear results file and write results
    with open("results_reto_5/resultados_5_1.txt", "w") as f:
        for graph_path in graph_files:
            graph_name = os.path.basename(graph_path)
                        
            # Load data
            edges, n = read_graph(graph_path)

            # Run alternating greedy simulation
            # print(graph_name) # Para ver qué aristas vamos cortando
            active, active_defenseless, mean_saved = simulation_with_alternating_greedy_noise(
                edges, population, initial_state, k_firewalls, N, s, N_greedy, s_greedy, seed
            )
            f.write(f"{graph_name}\t mean saved = {mean_saved}\n")

            draw_evolution(edges, active, 1, s, f"results_reto_5/simulaciones/{graph_name}_noise_{1}.png")