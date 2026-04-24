from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import colormaps
from matplotlib.lines import Line2D


def _node_positions(network):
    positions = dict(nx.get_node_attributes(network.G, "pos"))
    for node in network.nodes:
        positions.setdefault(node.id, (node.x, node.y))
    return positions


def _request_lookup(env):
    starts = {}
    targets = {}
    current_nodes = {}
    for request in getattr(env, "requests", []):
        starts.setdefault(request.start, []).append(request.id)
        targets.setdefault(request.target, []).append(request.id)
        if request.edge == -1:
            current_nodes.setdefault(request.now, []).append(request.id)
    return starts, targets, current_nodes


def _edge_fidelity(edge):
    if len(edge.links) == 0:
        return np.nan
    return float(np.mean([link.fidelity for link in edge.links]))


def visualize_environment(
    env,
    filename: Optional[str | Path] = None,
    show_plot: bool = False,
    title: Optional[str] = None,
    label_nodes: Optional[bool] = None,
):
    """
    Draw the generated quantum network environment.

    The visualization encodes:
    * topology edges and router positions
    * edge color by average active-link fidelity
    * edge width by active quantum-link count
    * node color by swap probability
    * request starts, targets, and current packet locations after reset
    """
    network = env.network
    graph = network.G
    positions = _node_positions(network)
    starts, targets, current_nodes = _request_lookup(env)

    if label_nodes is None:
        label_nodes = graph.number_of_nodes() <= 60

    node_swap_probs = [network.nodes[node].swap_prob for node in graph.nodes]
    node_sizes = [
        180 if node not in starts and node not in targets else 260
        for node in graph.nodes
    ]

    edge_list = list(graph.edges())
    edge_by_nodes = {
        (min(edge.start, edge.end), max(edge.start, edge.end)): edge
        for edge in network.edges
    }
    active_counts = []
    fidelities = []
    dead_edges = []
    for start, end in edge_list:
        edge = edge_by_nodes.get((min(start, end), max(start, end)))
        if edge is None or edge.dead:
            active_counts.append(0)
            fidelities.append(np.nan)
            dead_edges.append((start, end))
            continue
        active_counts.append(len(edge.links))
        fidelities.append(_edge_fidelity(edge))

    max_links = max(network.n_quantum_links, max(active_counts, default=0), 1)
    edge_widths = [0.8 + 3.2 * count / max_links for count in active_counts]
    edge_colors = [0.0 if np.isnan(fidelity) else fidelity for fidelity in fidelities]

    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    ax.set_title(title or "Quantum Network Environment", fontsize=14, pad=12)

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=edge_list,
        ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        edge_cmap=colormaps["viridis"],
        edge_vmin=0.5,
        edge_vmax=1.0,
        alpha=0.78,
    )
    if dead_edges:
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=dead_edges,
            ax=ax,
            width=1.2,
            edge_color="#9ca3af",
            style="dashed",
            alpha=0.9,
        )

    nodes = nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=node_swap_probs,
        cmap=colormaps["summer_r"],
        vmin=0,
        vmax=1,
        node_size=node_sizes,
        linewidths=0.8,
        edgecolors="#1f2937",
    )

    if label_nodes:
        nx.draw_networkx_labels(
            graph,
            positions,
            ax=ax,
            font_size=8,
            font_color="#111827",
        )

    if starts:
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=list(starts),
            ax=ax,
            node_shape="s",
            node_color="#ef4444",
            node_size=110,
            edgecolors="#7f1d1d",
            linewidths=0.8,
            label="request start",
        )
    if targets:
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=list(targets),
            ax=ax,
            node_shape="^",
            node_color="#22c55e",
            node_size=140,
            edgecolors="#14532d",
            linewidths=0.8,
            label="request target",
        )
    if current_nodes:
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=list(current_nodes),
            ax=ax,
            node_shape="o",
            node_color="none",
            node_size=360,
            edgecolors="#f97316",
            linewidths=2.0,
            label="packet location",
        )

    request_labels = {}
    for node, request_ids in current_nodes.items():
        request_labels[node] = ",".join(str(request_id) for request_id in request_ids[:3])
        if len(request_ids) > 3:
            request_labels[node] += "+"
    if request_labels:
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=request_labels,
            ax=ax,
            font_size=7,
            font_color="#9a3412",
            verticalalignment="bottom",
        )

    node_cbar = fig.colorbar(nodes, ax=ax, fraction=0.026, pad=0.02)
    node_cbar.set_label("Node swap probability")

    fidelity_norm = plt.Normalize(vmin=0.5, vmax=1.0)
    fidelity_sm = plt.cm.ScalarMappable(cmap=colormaps["viridis"], norm=fidelity_norm)
    fidelity_sm.set_array([])
    edge_cbar = fig.colorbar(fidelity_sm, ax=ax, fraction=0.026, pad=0.08)
    edge_cbar.set_label("Average link fidelity")

    legend_items = [
        Line2D([0], [0], marker="s", color="w", label="Request start", markerfacecolor="#ef4444", markeredgecolor="#7f1d1d", markersize=8),
        Line2D([0], [0], marker="^", color="w", label="Request target", markerfacecolor="#22c55e", markeredgecolor="#14532d", markersize=9),
        Line2D([0], [0], marker="o", color="#f97316", label="Packet location", markerfacecolor="none", markersize=10, linewidth=0),
        Line2D([0], [0], color="#6b7280", label="More active links = wider edge", linewidth=3),
    ]
    if dead_edges:
        legend_items.append(Line2D([0], [0], color="#9ca3af", label="Dead edge", linestyle="dashed"))
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
    )

    ax.set_axis_off()
    ax.margins(0.08)

    if filename is not None:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)
