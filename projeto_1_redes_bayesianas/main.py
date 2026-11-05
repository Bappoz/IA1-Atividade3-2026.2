"""Projeto 1 completo: Rede Bayesiana e experimento de diagnóstico industrial."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Mapping
import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


State = Dict[str, bool]
OUT = Path(__file__).parent / "results"
QUERIES = ("mechanical_failure", "electrical_failure", "wear", "overload")
SCENARIOS = {
    "normal": {},
    "thermal_vibration": {"high_temperature": True, "high_vibration": True},
    "electrical_clues": {"high_temperature": True, "bad_pressure": False, "high_vibration": False},
    "multiple": {"high_temperature": True, "high_vibration": True, "bad_pressure": True},
}


class BayesianNetwork:
    """Rede Bayesiana binária com inferência exata por enumeração."""

    def __init__(self, parents: Mapping[str, Iterable[str]], cpts: Mapping[str, Mapping[tuple, float]]):
        self.parents = {node: tuple(parent_list) for node, parent_list in parents.items()}
        self.cpts = {node: dict(table) for node, table in cpts.items()}
        self.nodes = tuple(self.parents)
        if set(self.nodes) != set(self.cpts):
            raise ValueError("Cada nó precisa possuir uma CPT.")
        self._validate_dag()
        for node in self.nodes:
            expected = {tuple(values) for values in product([False, True], repeat=len(self.parents[node]))}
            if set(self.cpts[node]) != expected:
                raise ValueError(f"CPT incompleta ou com chaves extras para '{node}'.")
            if not all(isinstance(value, (int, float)) and 0 <= value <= 1
                       for value in self.cpts[node].values()):
                raise ValueError("Probabilidades devem estar em [0, 1].")

    def _validate_dag(self) -> None:
        visiting, visited = set(), set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("A estrutura contém ciclo.")
            if node in visited:
                return
            visiting.add(node)
            for parent in self.parents[node]:
                if parent not in self.parents:
                    raise ValueError(f"Pai desconhecido: {parent}")
                visit(parent)
            visiting.remove(node)
            visited.add(node)

        for node in self.nodes:
            visit(node)

    def probability(self, assignment: State) -> float:
        """Calcula a probabilidade conjunta de uma atribuição completa."""
        if set(assignment) != set(self.nodes):
            raise ValueError("A atribuição deve informar todos os nós.")
        if not all(isinstance(value, (bool, np.bool_)) for value in assignment.values()):
            raise ValueError("Atribuições devem ser booleanas.")
        result = 1.0
        for node in self.nodes:
            key = tuple(assignment[parent] for parent in self.parents[node])
            p_true = self.cpts[node][key]
            result *= p_true if assignment[node] else 1 - p_true
        return result

    def posterior(self, query: str, evidence: Mapping[str, bool] | None = None) -> float:
        """Calcula P(query=True | evidence) por enumeração exata."""
        evidence = dict(evidence or {})
        if query not in self.nodes or not set(evidence).issubset(self.nodes):
            raise ValueError("Consulta ou evidência desconhecida.")
        numerator = denominator = 0.0
        hidden = [node for node in self.nodes if node not in evidence]
        for values in product([False, True], repeat=len(hidden)):
            assignment = dict(evidence)
            assignment.update(dict(zip(hidden, values)))
            joint = self.probability(assignment)
            denominator += joint
            if assignment[query]:
                numerator += joint
        if denominator == 0:
            raise ValueError("Evidência impossível segundo a rede.")
        return numerator / denominator

    def posterior_table(self, queries: Iterable[str], evidence: Mapping[str, bool] | None = None) -> Dict[str, float]:
        return {query: self.posterior(query, evidence) for query in queries}


def build_industrial_network() -> BayesianNetwork:
    """Cria a rede causal do cenário industrial."""
    parents = {
        "wear": (), "poor_lubrication": ("wear",), "overload": (),
        "high_temperature": ("overload", "poor_lubrication"),
        "high_vibration": ("wear", "overload"),
        "bad_pressure": ("poor_lubrication", "overload"),
        "mechanical_failure": ("wear", "high_vibration", "bad_pressure"),
        "electrical_failure": ("overload", "high_temperature"),
    }
    cpts = {
        "wear": {(): 0.18},
        "poor_lubrication": {(False,): 0.08, (True,): 0.62},
        "overload": {(): 0.15},
        "high_temperature": {(False, False): 0.03, (False, True): 0.28,
                            (True, False): 0.42, (True, True): 0.88},
        "high_vibration": {(False, False): 0.04, (False, True): 0.45,
                           (True, False): 0.55, (True, True): 0.91},
        "bad_pressure": {(False, False): 0.04, (False, True): 0.35,
                         (True, False): 0.40, (True, True): 0.82},
        "mechanical_failure": {
            (False, False, False): 0.005, (False, False, True): 0.12,
            (False, True, False): 0.18, (False, True, True): 0.58,
            (True, False, False): 0.20, (True, False, True): 0.66,
            (True, True, False): 0.72, (True, True, True): 0.97},
        "electrical_failure": {(False, False): 0.01, (False, True): 0.18,
                              (True, False): 0.10, (True, True): 0.55},
    }
    return BayesianNetwork(parents, cpts)


def run() -> dict:
    """Executa inferência, gera figuras e salva métricas."""
    OUT.mkdir(exist_ok=True)
    network = build_industrial_network()
    results = {name: network.posterior_table(QUERIES, evidence) for name, evidence in SCENARIOS.items()}

    graph = nx.DiGraph()
    for node, parents in network.parents.items():
        graph.add_node(node)
        graph.add_edges_from((parent, node) for parent in parents)
    plt.figure(figsize=(10, 5))
    nx.draw_networkx(graph, nx.spring_layout(graph, seed=7), node_color="#8ecae6",
                     node_size=2400, font_size=8, arrows=True, edge_color="#555555")
    plt.title("Rede Bayesiana do sistema industrial")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "bayesian_network.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    x, width = range(len(QUERIES)), 0.2
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

    sensitivity = [network.posterior("mechanical_failure",
                    {"high_temperature": True, "high_vibration": vibration})
                   for vibration in [False, True]]
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
