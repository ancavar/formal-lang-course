from project.task2 import (
    NondeterministicFiniteAutomaton,
    regex_to_dfa,
    graph_to_nfa,
    State,
)

from collections import defaultdict
from typing import Optional, Dict, Set, Iterable, Type
import numpy as np
import scipy.sparse as sp
from networkx import MultiDiGraph


class AdjacencyMatrixFA:
    def __init__(
        self,
        automaton: Optional[NondeterministicFiniteAutomaton] = None,
        matrix_class: Type[sp.spmatrix] = sp.csr_matrix,
    ):
        self.state_index: Dict[State, int] = {}
        self.start_state_indices: Set[int] = set()
        self.final_state_indices: Set[int] = set()
        self.matrix_class = matrix_class
        self.total_states = 0
        self.state_index = {}
        self.index_state = {}
        self.transition_matrices: Dict[str, sp.csr_matrix] = {}

        if automaton is None:
            return

        graph = automaton.to_networkx()
        self.total_states = graph.number_of_nodes()
        self.state_index = {state: idx for idx, state in enumerate(graph.nodes)}
        self.index_state = {idx: state for state, idx in self.state_index.items()}

        for node, attributes in graph.nodes(data=True):
            if attributes.get("is_start", False):
                self.start_state_indices.add(self.state_index[node])
            if attributes.get("is_final", False):
                self.final_state_indices.add(self.state_index[node])

        transitions = defaultdict(
            lambda: np.zeros((self.total_states, self.total_states), dtype=bool)
        )

        for source, target, symbol in graph.edges(data="label"):
            if symbol:
                transitions[symbol][
                    self.state_index[source], self.state_index[target]
                ] = True

        self.transition_matrices = {
            sym: self.matrix_class(matrix) for sym, matrix in transitions.items()
        }

    def accepts(self, word: Iterable[str]) -> bool:
        current_states = set(self.start_state_indices)

        for symbol in word:
            next_states = set()
            for state in current_states:
                for dst in self.transition_matrices[symbol].nonzero()[1]:
                    next_states.add(dst)
            current_states = next_states
            if not current_states:
                return False

        return bool(set(self.final_state_indices).intersection(current_states))

    def is_empty(self) -> bool:
        reachability_matrix = self.transitive_closure()

        for start_state in self.start_state_indices:
            for final_state in self.final_state_indices:
                if reachability_matrix[start_state, final_state]:
                    return False

        return True

    def transitive_closure(self):
        reach = self.matrix_class((self.total_states, self.total_states), dtype=bool)
        reach.setdiag(True)

        if not self.transition_matrices:
            return reach

        adjacency_matrix = sp.csr_matrix(
            (self.total_states, self.total_states), dtype=bool
        )
        for matrix in self.transition_matrices.values():
            adjacency_matrix += matrix
        dist_matrix = sp.csgraph.floyd_warshall(
            adjacency_matrix, directed=True, unweighted=True
        )
        reach_matrix = dist_matrix < np.inf

        return self.matrix_class(reach_matrix)


def intersect_automata(
    automaton1: AdjacencyMatrixFA,
    automaton2: AdjacencyMatrixFA,
    matrix_class: Type[sp.spmatrix] = sp.csr_matrix,
) -> AdjacencyMatrixFA:
    intersected_automaton = AdjacencyMatrixFA(matrix_class=matrix_class)

    intersected_automaton.total_states = (
        automaton1.total_states * automaton2.total_states
    )

    intersected_automaton.state_index = {
        (s1, s2): (
            automaton1.state_index[s1] * automaton2.total_states
            + automaton2.state_index[s2]
        )
        for s1 in automaton1.state_index
        for s2 in automaton2.state_index
    }

    intersected_automaton.index_state = {
        idx: state for state, idx in intersected_automaton.state_index.items()
    }

    intersected_automaton.start_state_indices = {
        s1 * automaton2.total_states + s2
        for s1 in automaton1.start_state_indices
        for s2 in automaton2.start_state_indices
    }

    intersected_automaton.final_state_indices = {
        f1 * automaton2.total_states + f2
        for f1 in automaton1.final_state_indices
        for f2 in automaton2.final_state_indices
    }

    intersected_automaton.transition_matrices = {}

    common_symbols = set(automaton1.transition_matrices.keys()).intersection(
        automaton2.transition_matrices.keys()
    )

    for symbol in common_symbols:
        matrix1 = automaton1.transition_matrices[symbol]
        matrix2 = automaton2.transition_matrices[symbol]

        intersected_automaton.transition_matrices[symbol] = sp.kron(
            matrix1, matrix2, format=matrix_class([[]]).getformat()
        )

    return intersected_automaton


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
    matrix_class: Type[sp.spmatrix] = sp.csr_matrix,
) -> set[tuple[int, int]]:
    regex_adj = AdjacencyMatrixFA(regex_to_dfa(regex), matrix_class)
    graph_adj = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes), matrix_class
    )
    intersect = intersect_automata(regex_adj, graph_adj, matrix_class)
    result_set = set()
    transitive_closure = intersect.transitive_closure()

    regex_start_states = [
        key
        for key in regex_adj.state_index
        if regex_adj.state_index[key] in regex_adj.start_state_indices
    ]
    regex_final_states = [
        key
        for key in regex_adj.state_index
        if regex_adj.state_index[key] in regex_adj.final_state_indices
    ]

    for st in start_nodes:
        for fn in final_nodes:
            for st_reg in regex_start_states:
                for fn_reg in regex_final_states:
                    if transitive_closure[
                        intersect.state_index[(st_reg, st)],
                        intersect.state_index[(fn_reg, fn)],
                    ]:
                        result_set.add((st, fn))

    return result_set
