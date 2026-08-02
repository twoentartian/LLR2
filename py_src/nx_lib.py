"""NetworkX helpers exposed to simulator configuration files."""

from __future__ import annotations

import os
from typing import Iterable, Optional

import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection


def split_to_equal_size_communities(topology: nx.Graph, num_of_communities: int) -> list[list]:
    """Partition a graph into approximately equal-size communities."""
    if not 1 <= num_of_communities <= topology.number_of_nodes():
        raise ValueError("num_of_communities must be between 1 and the graph's node count")
    if num_of_communities == 1:
        return [list(topology.nodes)]

    communities = [list(community) for community in kernighan_lin_bisection(topology)]
    while len(communities) < num_of_communities:
        new_communities = []
        for community in communities:
            subgraph = topology.subgraph(community)
            if len(subgraph) > 1:
                new_communities.extend(list(map(list, kernighan_lin_bisection(subgraph))))
            else:
                new_communities.append(community)
        communities = new_communities

    while len(communities) > num_of_communities:
        communities = sorted(communities, key=len)
        communities = [communities[0] + communities[1], *communities[2:]]

    while True:
        largest = max(communities, key=len)
        smallest = min(communities, key=len)
        if len(largest) - len(smallest) <= 1:
            break
        smallest.append(largest.pop())

    return communities


def get_inter_community_edges(topology: nx.Graph, communities: Iterable[Iterable]) -> list[tuple]:
    """Return graph edges whose endpoints belong to different communities."""
    node_to_community = {
        node: community_index
        for community_index, community in enumerate(communities)
        for node in community
    }
    return [
        (source, destination)
        for source, destination in topology.edges()
        if node_to_community[source] != node_to_community[destination]
    ]


def load_topology_from_edge_list_file(file_path: str) -> nx.Graph:
    """Load an undirected integer-node graph from a whitespace edge list."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    graph = nx.Graph()
    with open(file_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            node_1, node_2 = map(int, line.split())
            graph.add_edge(node_1, node_2)
    return graph


def display_graph(
    G: nx.Graph,
    layout: Optional[str] = None,
    node_color="lightblue",
    node_size=500,
    with_labels=True,
    title="Network Graph",
    figsize=(10, 8),
    edge_color="gray",
    font_size=10,
) -> None:
    """Display a graph using matplotlib, importing plotting support lazily."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    layouts = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "random": nx.random_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "shell": nx.shell_layout,
    }
    if layout is None:
        position = nx.nx_agraph.graphviz_layout(G)
    elif layout in layouts:
        position = layouts[layout](G, seed=42) if layout == "spring" else layouts[layout](G)
    else:
        raise ValueError("Invalid layout")

    nx.draw(
        G,
        position,
        with_labels=with_labels,
        node_color=node_color,
        node_size=node_size,
        edge_color=edge_color,
        font_size=font_size,
        font_weight="bold",
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
