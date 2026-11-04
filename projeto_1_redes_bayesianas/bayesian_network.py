"""Rede Bayesiana binária e inferência exata por enumeração."""
from __future__ import annotations
from itertools import product
from typing import Dict, Iterable, Mapping

State = Dict[str, bool]


class BayesianNetwork:
    """Representa um DAG binário com CPTs P(nó=True | pais)."""

    def __init__(self, parents: Mapping[str, Iterable[str]], cpts: Mapping[str, Mapping[tuple, float]]):
        self.parents = {node: tuple(parent_list) for node, parent_list in parents.items()}
        self.cpts = {node: dict(table) for node, table in cpts.items()}
        self.nodes = tuple(self.parents)
        if set(self.nodes) != set(self.cpts):
            raise ValueError("Cada nó precisa possuir uma CPT.")
        self._validate_dag()
        for node in self.nodes:
            for value in self.cpts[node].values():
                if not 0 <= value <= 1:
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
        "wear": (),
        "poor_lubrication": ("wear",),
        "overload": (),
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
