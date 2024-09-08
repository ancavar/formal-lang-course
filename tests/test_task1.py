import networkx as nx
import pydot
import cfpq_data
import pytest
from project.task1 import get_graph_data, save_labeled_two_cycles_graph, GraphData


def test_get_graph_data():
    expected_node_count = 332
    expected_edge_count = 269
    expected_labels = {"a", "d"}
    graph = get_graph_data("wc")

    assert isinstance(graph, GraphData)
    assert graph.node_count == expected_node_count
    assert graph.edge_count == expected_edge_count
    assert set(graph.labels) == expected_labels


@pytest.mark.parametrize(
    "n, m, labels",
    [
        (3, 4, ["a", "b"]),
        (5, 5, ["x", "y"]),
        (2, 3, ["label1", "label2"]),
    ],
)
def test_create_labeled_two_cycles_graph(tmp_path, n, m, labels):
    test_path = tmp_path / ("test.dot")

    graph = save_labeled_two_cycles_graph(n, m, labels, test_path)

    assert test_path.exists()

    pydot_graph = pydot.graph_from_dot_file(str(test_path))[0]
    graph = nx.nx_pydot.from_pydot(pydot_graph)

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == n + m + 1
    assert set(cfpq_data.get_sorted_labels(graph)) == set(labels)
