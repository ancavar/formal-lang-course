from pyformlang.cfg import CFG, Production
import networkx as nx


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    if len(cfg.productions) == 0:
        return cfg

    new_cfg = (
        cfg.remove_useless_symbols()
        .eliminate_unit_productions()
        .remove_useless_symbols()
    )

    new_productions = new_cfg._get_productions_with_only_single_terminals()
    new_productions = new_cfg._decompose_productions(new_productions)
    return CFG(start_symbol=cfg.start_symbol, productions=set(new_productions))


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    weak_cnf_cfg = cfg_to_weak_normal_form(cfg)

    r = list()
    new = list()

    for v1, v2, symbol in graph.edges(data="label"):
        for production in weak_cnf_cfg.productions:
            # A -> terminal
            if len(production.body) == 1 and production.body[0].value == symbol:
                r.append((production.head, v1, v2))
                new.append((production.head, v1, v2))

    # A -> epsilon
    for variable in weak_cnf_cfg.variables:
        if Production(variable, []) in weak_cnf_cfg.productions:
            for vertex in graph.nodes:
                r.append((variable, vertex, vertex))
                new.append((variable, vertex, vertex))

    while new:
        (N, n, m) = new.pop()

        # (N', n', m) where N' -> MN
        for M, n_prime, m_prime in r:
            if m_prime == n:
                for production in weak_cnf_cfg.productions:
                    if (
                        len(production.body) == 2
                        and production.body[0] == M
                        and production.body[1] == N
                    ):
                        N_prime = production.head
                        new_relation = (production.head, n_prime, m)
                        if new_relation not in r:
                            r.append(new_relation)
                            new.append(new_relation)

        # (N', n, m') where N' -> NM
        for M, n_prime, m_prime in r:
            if m == n_prime:
                for production in weak_cnf_cfg.productions:
                    if (
                        len(production.body) == 2
                        and production.body[0] == N
                        and production.body[1] == M
                    ):
                        N_prime = production.head
                        new_relation = (N_prime, n, m_prime)
                        if new_relation not in r:
                            r.append(new_relation)
                            new.append(new_relation)

    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    return {
        (start, final)
        for variable, start, final in r
        if start in start_nodes
        and final in final_nodes
        and variable == cfg.start_symbol
    }
