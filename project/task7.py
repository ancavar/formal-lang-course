from pyformlang.cfg import CFG, Production
import networkx as nx
from typing import Set
from scipy.sparse import csr_matrix
from project.task6 import cfg_to_weak_normal_form


def _initialize_matrix(weak_cnf_cfg, graph, vertex_to_id):
    n = graph.number_of_nodes()
    adjacency_matrix = {
        var: csr_matrix((n, n), dtype=bool) for var in weak_cnf_cfg.variables
    }

    # A -> terminal
    for v1, v2, symbol in graph.edges(data="label"):
        for production in weak_cnf_cfg.productions:
            if len(production.body) == 1 and production.body[0].value == symbol:
                adjacency_matrix[production.head][
                    vertex_to_id[v1], vertex_to_id[v2]
                ] = True

    # A -> epsilon
    for variable in weak_cnf_cfg.variables:
        if Production(variable, []) in weak_cnf_cfg.productions:
            for i in range(n):
                adjacency_matrix[variable][i, i] = True

    return adjacency_matrix


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    weak_cnf_cfg = cfg_to_weak_normal_form(cfg)

    vertex_to_id = {v: i for i, v in enumerate(graph.nodes)}
    id_to_vertex = {i: v for v, i in vertex_to_id.items()}

    adjacency_matrix = _initialize_matrix(weak_cnf_cfg, graph, vertex_to_id)
    new = set(cfg.variables)

    while new:
        new.pop()
        for production in weak_cnf_cfg.productions:
            if len(production.body) == 2:
                M, N = production.body
                prod = adjacency_matrix[M] @ adjacency_matrix[N]
                nonterm = production.head
                prev = adjacency_matrix[nonterm]
                adjacency_matrix[nonterm] += prod
                if (prev != adjacency_matrix[nonterm]).count_nonzero() != 0:
                    new.add(nonterm)

    return {
        (id_to_vertex[row], id_to_vertex[col])
        for nonterm, matrix in adjacency_matrix.items()
        if nonterm == cfg.start_symbol
        for row, col in zip(*matrix.nonzero())
        if id_to_vertex[row] in start_nodes and id_to_vertex[col] in final_nodes
    }
