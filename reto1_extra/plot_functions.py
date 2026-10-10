import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def draw_evolution(edges, active, iteration, s, output_dir):
    G = nx.DiGraph()
    G.add_edges_from(edges.keys())
    pos = nx.planar_layout(G)

    plt.figure()
    nx.draw(G, pos=pos, node_color="white", edgecolors="k", with_labels=True, font_color="k", arrows=False) # type: ignore

    colors = generate_colors(s)

    for time in range(s):
        active_list = []
        if (iteration, time) in active:
            active_list = list(active[(iteration, time)])

        # Active nodes (highlighted)
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=active_list,
            node_color=colors[time]
        )

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(range(len(colors) + 1), cmap.N)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=plt.gca(),
        ticks=[i + 0.5 for i in range(len(colors))],
        label="Time Step",
    )
    cbar.ax.set_yticklabels([str(i) for i in range(len(colors))])

    plt.savefig(output_dir, dpi=300, bbox_inches="tight")       

def generate_colors(n):
    """
    Generate a list of N non-repeating colors.
    
    Args:
        n: Number of colors to generate
    
    Returns:
        List of N unique colors
    """
    base_colors = ["red", "blue", "green", "yellow", "orange", "pink", "purple",
                   "cyan", "magenta", "brown", "gray", "black", "white", 
                   "teal", "navy", "maroon", "olive", "lime", "coral", "gold"]
    
    if n > len(base_colors):
        # Generate additional unique colors if N exceeds base palette
        extra_colors = [f"color_{i}" for i in range(len(base_colors), n)]
        base_colors = base_colors + extra_colors
    
    return base_colors[:n]

if __name__ == "__main__":
    import os, glob, random
from io_functions import read_graph, read_start, set_zero_edges, remove_edges
from montecarlo import single_montecarlo_simulation, fixed_montecarlo_simulation, expected_burnt_zones
from plot_functions import draw_evolution
from reto_3 import simulation_with_alternating_greedy

# Main execution
if __name__ == "__main__":
    import glob, os
    
    os.makedirs("tests", exist_ok=True)

    graph_files = sorted(glob.glob("Datos/graph_*"))

    # Clear results file and write results
    for graph_path in graph_files:
        graph_name = os.path.basename(graph_path)

        # Load data
        edges, n = read_graph(graph_path)

        draw_evolution(edges, {}, 0, 1, f"tests/{graph_name}.png")
