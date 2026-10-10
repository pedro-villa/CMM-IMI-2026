"""
reto_3_optimizado.py
--------------------
Solución alternativa optimizada para el Reto 3: Control estocástico del fuego
mediante eliminación voraz alternante de aristas.

Optimizaciones implementadas:
  1. Poda topológica y de dominancia probabilística del cociclo frontera.
  2. Reducción de varianza mediante Números Aleatorios Comunes (CRN).
  3. Criba competitiva en dos etapas (Racing / Successive Halving).
  4. Memoización de acciones voraces por estado canónico (DP Cache).
  5. Acoplamiento estocástico sincronizado y guardas por extinción temprana.
"""

import glob
import os
import random
import shutil
import sys
import time
from typing import Dict, List, Set, Tuple

from io_functions import read_graph, set_zero_edges
from montecarlo import expected_burnt_zones, fixed_montecarlo_simulation, single_montecarlo_simulation


def read_mean_saved_results(results_path: str) -> Dict[str, float]:
    """
    Lee las medias de nodos salvados indexadas por nombre de grafo desde un
    archivo de resultados previo para permitir análisis comparativos.
    """
    previous_results: Dict[str, float] = {}
    if not os.path.exists(results_path):
        return previous_results

    with open(results_path, "r", encoding="utf-8") as results_file:
        for line in results_file:
            columns = line.rstrip("\n").split("\t")
            if len(columns) >= 3 and columns[0] != "graph":
                try:
                    previous_results[columns[0].strip()] = float(columns[2])
                    continue
                except ValueError:
                    pass
            if "mean saved =" not in line:
                continue
            graph_name, value = line.split("mean saved =", 1)
            graph_name = graph_name.split("\t", 1)[0].strip()
            previous_results[graph_name] = float(value.strip())

    return previous_results


def obtain_candidate_edges(
    reduced_edges: Dict[Tuple[int, int], float],
    initial_state: List[int],
    population: List[int],
) -> List[Tuple[int, int]]:
    """
    Filtra y poda el espacio de aristas candidatas aplicando criterios
    topológicos y dominancia probabilística:
      1. Extrae el cociclo frontera delta(I_t, S_t) con p >= 0.05.
      2. Poda por dominancia multientrada: si k=1 y un nodo recibe múltiples
         vías de contagio, un corte aislado no evita la infección inmediata.
      3. Poda de hojas: nodos susceptibles con grado saliente 0 hacia otros
         sanos no propagan el fuego a terceros.
      4. Incorpora aristas amortiguadoras a 1 paso de los objetivos directos.
    """
    frente: Set[int] = set(initial_state)
    sanos: Set[int] = set(population)

    if not frente or not sanos:
        return []

    # Identificar aristas entrantes activas hacia cada nodo susceptible
    incoming_to_target: Dict[int, Set[Tuple[Tuple[int, int], float]]] = {}
    for (u, v), p in reduced_edges.items():
        if p >= 0.05 and u in frente and v in sanos:
            canon = (min(u, v), max(u, v))
            incoming_to_target.setdefault(v, set()).add((canon, p))

    # Grado residual de cada nodo dentro del subgrafo susceptible
    susceptible_degree: Dict[int, int] = {node: 0 for node in sanos}
    for (u, v), p in reduced_edges.items():
        if p > 0.0 and u in sanos and v in sanos and u < v:
            susceptible_degree[u] += 1
            susceptible_degree[v] += 1

    candidatas: Set[Tuple[int, int]] = set()
    penalizadas: Set[Tuple[int, int]] = set()

    for target, in_edges in incoming_to_target.items():
        deg_out = susceptible_degree.get(target, 0)
        is_leaf_dead_end = (deg_out == 0)
        has_multiple_entries = (len(in_edges) > 1)

        for canon, _ in in_edges:
            if is_leaf_dead_end or has_multiple_entries:
                penalizadas.add(canon)
            else:
                candidatas.add(canon)

    # Perímetro secundario: aristas protectoras tras los objetivos directos
    for target in incoming_to_target.keys():
        for (u, v), p in reduced_edges.items():
            if p >= 0.05 and u < v:
                if (u == target and v in sanos) or (v == target and u in sanos):
                    candidatas.add((u, v))

    # Garantía de cobertura: si la poda vacía el conjunto directo, recuperar penalizadas
    direct_frontier = {canon for in_edges in incoming_to_target.values() for canon, _ in in_edges}
    if not direct_frontier.intersection(candidatas):
        candidatas.update(penalizadas)

    # Respaldo de seguridad en caso de grafos desconectados
    if not candidatas:
        return sorted({(min(u, v), max(u, v)) for (u, v), p in reduced_edges.items() if p > 0.0})

    return sorted(candidatas)


def greedy_edge_removals_general(
    edges: Dict[Tuple[int, int], float],
    population: List[int],
    initial_state: List[int],
    k: int,
    N: int,
    s: int,
    step_seed: int = 1,
    top_candidates_limit: int = 4,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Selecciona de forma voraz k cortafuegos minimizando el daño esperado
    mediante Números Aleatorios Comunes (CRN) y criba en dos fases (Racing).
    """
    reduced = edges.copy()
    removed: List[Tuple[int, int]] = []
    best_value = float("inf")

    # Tamaño de muestra para la criba piloto
    n_pilot = max(6, N // 5)

    for step in range(k):
        undirected = obtain_candidate_edges(reduced, initial_state, population)
        if not undirected:
            break

        # Semilla compartida para evaluar todas las candidatas bajo idéntico azar
        crn_seed = (step_seed * 1009 + step * 31) % (2**31 - 1)

        if len(undirected) <= top_candidates_limit or N <= n_pilot:
            shortlist = undirected
        else:
            # Fase 1: Criba rápida piloto
            pilot_results = []
            for (u, v) in undirected:
                trial = reduced.copy()
                trial.pop((u, v), None)
                trial.pop((v, u), None)

                random.seed(crn_seed)
                active = single_montecarlo_simulation(
                    trial, list(population), initial_state, n_pilot, s
                )
                pilot_score = expected_burnt_zones(active, n_pilot, s)
                pilot_results.append((pilot_score, (u, v)))

            pilot_results.sort(key=lambda item: item[0])
            shortlist = [edge for _, edge in pilot_results[:top_candidates_limit]]

        # Fase 2: Refinamiento de precisión sobre las finalistas
        best_value = float("inf")
        best_edge = None
        refine_seed = (crn_seed + 7919) % (2**31 - 1)

        for (u, v) in shortlist:
            trial = reduced.copy()
            trial.pop((u, v), None)
            trial.pop((v, u), None)

            random.seed(refine_seed)
            active = single_montecarlo_simulation(
                trial, list(population), initial_state, N, s
            )
            value = expected_burnt_zones(active, N, s)

            if value < best_value:
                best_value = value
                best_edge = (u, v)

        if best_edge is not None:
            reduced.pop(best_edge, None)
            reduced.pop((best_edge[1], best_edge[0]), None)
            removed.append(best_edge)

    return removed, best_value


def simulation_with_alternating_greedy(
    edges: Dict[Tuple[int, int], float],
    population: List[int],
    initial_state: List[int],
    k: int,
    N: int,
    s: int,
    N_greedy: int,
    s_greedy: int,
    seed: int,
    top_candidates_limit: int = 4,
) -> Tuple[Dict, Dict, float]:
    """
    Ejecuta N réplicas de la evolución temporal del fuego alternando colocación
    de cortafuegos y contagio estocástico. Emplea memoización de estados
    para calcular la política óptima una única vez por configuración.
    """
    total_burnt = 0
    total_burnt_defenseless = 0
    active: Dict[Tuple[int, int], Set[int]] = {}
    active_defenseless: Dict[Tuple[int, int], Set[int]] = {}
    greedy_cache: Dict[Tuple, List[Tuple[int, int]]] = {}

    for iteration in range(N):
        current_edges = edges.copy()
        current_population = set(population)
        current_population_defenseless = set(population)
        current_cuts: Set[Tuple[int, int]] = set()

        active[(iteration, 0)] = set(initial_state)
        active_defenseless[(iteration, 0)] = set(initial_state)

        for t in range(1, s):
            frente_defensa = active[(iteration, t - 1)]
            frente_sindef = active_defenseless[(iteration, t - 1)]

            # 1. Colocación de cortafuegos (solo si el frente con defensa sigue activo)
            if frente_defensa:
                state_key = (
                    frozenset(current_cuts),
                    frozenset(frente_defensa),
                    frozenset(current_population),
                )

                if state_key in greedy_cache:
                    removed = greedy_cache[state_key]
                else:
                    state_seed = (abs(hash(state_key)) + seed) % (2**31 - 1)
                    removed, _ = greedy_edge_removals_general(
                        current_edges,
                        list(current_population),
                        list(frente_defensa),
                        k,
                        N_greedy,
                        s_greedy,
                        step_seed=state_seed,
                        top_candidates_limit=top_candidates_limit,
                    )
                    greedy_cache[state_key] = removed

                current_edges = set_zero_edges(current_edges, removed)
                current_cuts.update(removed)

            # Semilla acoplada para el paso temporal t de propagación
            step_sim_seed = (iteration * 10007 + t * 131 + seed) % (2**31 - 1)

            # 2. Simulación acoplada de 1 paso en ambas trayectorias
            if frente_defensa:
                active_after = fixed_montecarlo_simulation(
                    current_edges,
                    current_population,
                    frente_defensa,
                    N=1,
                    s=2,
                    seed=step_sim_seed,
                )
                nuevos_defensa = active_after.get((0, 1), set())
            else:
                nuevos_defensa = set()

            if frente_sindef:
                active_after_defenseless = fixed_montecarlo_simulation(
                    edges,
                    current_population_defenseless,
                    frente_sindef,
                    N=1,
                    s=2,
                    seed=step_sim_seed,
                )
                nuevos_sindef = active_after_defenseless.get((0, 1), set())
            else:
                nuevos_sindef = set()

            # Actualizaciones in-place de poblaciones y acumulados
            current_population.difference_update(frente_defensa)
            current_population_defenseless.difference_update(frente_sindef)

            total_burnt += len(frente_defensa)
            total_burnt_defenseless += len(frente_sindef)

            # Criterio de parada: ambos incendios extinguidos
            if not nuevos_defensa and not nuevos_sindef:
                break

            active[(iteration, t)] = nuevos_defensa
            active_defenseless[(iteration, t)] = nuevos_sindef

    mean_saved = (total_burnt_defenseless - total_burnt) / N
    return active, active_defenseless, mean_saved


if __name__ == "__main__":
    seed = 1
    show_time = "-time" in sys.argv

    # Hiperparámetros calibrados para rigor y velocidad de entrega
    N = 1000             # Iteraciones externas de Monte Carlo (estabilidad de decimales)
    s = 20              # Horizonte máximo de simulación
    N_greedy = 200      # Réplicas voraces por candidata (precisión alta bajo CRN)
    s_greedy = 3        # Horizonte de anticipación táctica voraz
    k_firewalls = 1     # Cortafuegos por etapa

    os.makedirs("results_reto_3", exist_ok=True)
    graph_files = sorted(glob.glob("Datos/graph_*"))

    initial_state = [1]
    population = list(range(20))

    current_results_path = "results_reto_3/resultados.txt"
    previous_results_path = "results_reto_3/resultados2.txt"
    if os.path.exists(current_results_path):
        shutil.copyfile(current_results_path, previous_results_path)

    previous_results = read_mean_saved_results(previous_results_path)

    with open(current_results_path, "w", encoding="utf-8") as f:
        f.write(
            "graph\tmean_saved_anterior\tmean_saved_actual\t"
            "diferencia\tvariacion_porcentual\ttiempo_s\n"
        )
        for graph_path in graph_files:
            graph_start = time.perf_counter()
            graph_name = os.path.basename(graph_path)

            edges, n = read_graph(graph_path)

            active, active_defenseless, mean_saved = simulation_with_alternating_greedy(
                edges,
                population,
                initial_state,
                k_firewalls,
                N,
                s,
                N_greedy,
                s_greedy,
                seed,
                top_candidates_limit=4,
            )

            elapsed = time.perf_counter() - graph_start

            previous_mean_saved = previous_results.get(graph_name)
            if previous_mean_saved is None:
                prev_str = "N/A"
                diff_str = "N/A"
                var_str = "N/A"
            else:
                difference = mean_saved - previous_mean_saved
                variation = (
                    (difference / previous_mean_saved) * 100.0
                    if previous_mean_saved != 0.0
                    else 0.0
                )
                prev_str = f"{previous_mean_saved:.2f}"
                diff_str = f"{difference:.2f}"
                var_str = f"{variation:.2f}%"

            f.write(
                f"{graph_name}\t{prev_str}\t{mean_saved:.2f}\t"
                f"{diff_str}\t{var_str}\t{elapsed:.2f}\n"
            )
            f.flush()

            if show_time:
                print(f"{graph_name}: {elapsed:.2f} s | Salvas medias: {mean_saved:.2f}")