"""Experimentos, visualizações e métricas do projeto de Rede Bayesiana."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import networkx as nx
from bayesian_network import build_industrial_network

OUT = Path(__file__).parent / "results"
QUERIES = ("mechanical_failure", "electrical_failure", "wear", "overload")
SCENARIOS = {
    "normal": {},
    "thermal_vibration": {"high_temperature": True, "high_vibration": True},
    "electrical_clues": {"high_temperature": True, "bad_pressure": False, "high_vibration": False},
    "multiple": {"high_temperature": True, "high_vibration": True, "bad_pressure": True},
}


def run() -> dict:
    OUT.mkdir(exist_ok=True)
    network = build_industrial_network()
    results = {name: network.posterior_table(QUERIES, evidence) for name, evidence in SCENARIOS.items()}
    with (OUT / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    graph = nx.DiGraph()
    for node, parents in network.parents.items():
        graph.add_node(node)
        graph.add_edges_from((parent, node) for parent in parents)
    plt.figure(figsize=(10, 5))
    pos = nx.spring_layout(graph, seed=7)
    nx.draw_networkx(graph, pos, node_color="#8ecae6", node_size=2400, font_size=8,
                     arrows=True, edge_color="#555555")
    plt.title("Rede Bayesiana do sistema industrial")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "bayesian_network.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(QUERIES))
    width = 0.2
    for i, (scenario, values) in enumerate(results.items()):
        ax.bar([v + (i - 1.5) * width for v in x], [values[q] for q in QUERIES],
               width, label=scenario.replace("_", " "))
    ax.set_xticks(list(x), [q.replace("_", "\n") for q in QUERIES])
    ax.set_ylabel("Probabilidade posterior")
    ax.set_ylim(0, 1)
    ax.set_title("Diagnóstico posterior por cenário")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "posterior_probabilities.png", dpi=300)
    plt.close(fig)

    sensitivity = []
    for vibration in [False, True]:
        evidence = {"high_temperature": True, "high_vibration": vibration}
        sensitivity.append(network.posterior("mechanical_failure", evidence))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(["temperatura alta\nvibração normal", "temperatura alta\nvibração alta"],
            sensitivity, marker="o", color="#d1495b")
    ax.set_ylabel("P(falha mecânica | evidências)")
    ax.set_ylim(0, 1)
    ax.set_title("Análise de sensibilidade à vibração")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity_analysis.png", dpi=300)
    plt.close(fig)
    results["sensitivity_delta"] = sensitivity[1] - sensitivity[0]
    with (OUT / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
