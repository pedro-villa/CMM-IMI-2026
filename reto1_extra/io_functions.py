def read_graph(filename):
    """
    Reads the graph file and returns a dictionary of edge probabilities.

    Returns:
        edges: dict mapping (i, j) -> p_ij (probability that i infects j)
        n_nodes: number of nodes in the graph
    """
    edges = {}
    n_nodes = 0

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            i = int(parts[0])
            j = int(parts[1])
            p_ij = float(parts[2])
            edges[(i, j)] = p_ij
            n_nodes = max(n_nodes, i + 1, j + 1)

    return edges, n_nodes

def read_graph_as_binary(filename):
    """
    Reads the graph file and returns a dictionary of edge probabilities = 1.

    Returns:
        edges: dict mapping (i, j) -> 1
        n_nodes: number of nodes in the graph
    """
    edges = {}
    n_nodes = 0

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            i = int(parts[0])
            j = int(parts[1])
            edges[(i, j)] = 1
            n_nodes = max(n_nodes, i + 1, j + 1)

    return edges, n_nodes


def read_start(filename):
    """
    Reads the initial probabilities file and returns a vector of initial probabilities.

    Returns:
        p0: list where p0[i] is the initial probability of node i
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            node = int(parts[0])
            prob = float(parts[1])
            data.append((node, prob))

    # Sort by node index and build the vector
    data.sort(key=lambda x: x[0])
    n_nodes = data[-1][0] + 1
    p0 = [0] * n_nodes
    for node, prob in data:
        p0[node] = prob

    return p0

def remove_edges(edges, removed):
    """
    Return a copy of `edges` with the given undirected edges removed.

    Args:
        edges: dict mapping (i, j) -> probability
        removed: iterable of (u, v) pairs (undirected). Both (u, v) and (v, u)
                 will be removed from the returned dict.

    Returns:
        A new dict with the removed edges (both directions) taken out.
    """
    reduced = edges.copy()
    for (u, v) in removed:
        reduced.pop((u, v), None)
        reduced.pop((v, u), None)
    return reduced

def set_zero_edges(edges, removed):
    """
    Return a copy of `edges` with the given undirected edges set to 0.

    Args:
        edges: dict mapping (i, j) -> probability

    Returns:
        A new dict with the removed edges (both directions) with value 0.
    """
    reduced = edges.copy()
    for (u, v) in removed:
        reduced[(u, v)] = 0.0
        reduced[(v, u)] = 0.0
    return reduced