import itertools
from collections import defaultdict

from typing import TYPE_CHECKING

from torch.utils.data.datapipes.iter.combining import _ChildDataPipe, IterDataPipe
from torch.utils.data.graph import traverse

if TYPE_CHECKING:
    import graphviz


class Node:
    def __init__(self, dp, *, name=None):
        self.dp = dp
        self.name = name or type(dp).__name__.replace("IterDataPipe", "")
        self.childs = set()
        self.parents = set()

    def add_child(self, child):
        self.childs.add(child)
        child.parents.add(self)

    def remove_child(self, child):
        self.childs.remove(child)
        child.parents.remove(self)

    def add_parent(self, parent):
        self.parents.add(parent)
        parent.childs.add(self)

    def remove_parent(self, parent):
        self.parents.remove(parent)
        parent.childs.remove(self)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented

        return hash(self) == hash(other)

    def __hash__(self):
        return hash(self.dp)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self}-{hash(self)}"


def to_nodes(dp):
    def recurse(dp_graph, child=None):
        for dp_node, dp_parents in dp_graph.items():
            node = Node(dp_node)
            if child is not None:
                node.add_child(child)
            yield node
            yield from recurse(dp_parents, child=node)

    def aggregate(nodes):
        groups = defaultdict(list)
        for node in nodes:
            groups[node].append(node)

        nodes = set()
        for node, group in groups.items():
            if len(group) == 1:
                nodes.add(node)
                continue

            aggregated_node = Node(node.dp)

            for duplicate_node in group:
                for child in duplicate_node.childs.copy():
                    duplicate_node.remove_child(child)
                    aggregated_node.add_child(child)

                for parent in duplicate_node.parents.copy():
                    duplicate_node.remove_parent(parent)
                    aggregated_node.add_parent(parent)

            nodes.add(aggregated_node)

        child_dp_nodes = set(
            itertools.chain.from_iterable(node.parents for node in nodes if isinstance(node.dp, _ChildDataPipe))
        )

        if not child_dp_nodes:
            return nodes

        for node in child_dp_nodes:
            fixed_parent_node = Node(
                type(str(node).lstrip("_"), (IterDataPipe,), dict(dp=node.dp, childs=node.childs))()
            )
            nodes.remove(node)
            nodes.add(fixed_parent_node)

            for parent in node.parents.copy():
                node.remove_parent(parent)
                fixed_parent_node.add_parent(parent)

            for child in node.childs:
                nodes.remove(child)
                for actual_child in child.childs.copy():
                    actual_child.remove_parent(child)
                    actual_child.add_parent(fixed_parent_node)

        return nodes

    return aggregate(recurse(traverse(dp)))


def to_graph(dp) -> "graphviz.Digraph":
    """Turns a datapipe into a :class:`graphviz.Digraph` representing the graph of the datapipe.

    .. note::

        The package :mod:`graphviz` is required to use this function.

    .. note::

        The most common interfaces for the returned graph object are:

        - :meth:`~graphviz.Digraph.render`: Save the graph to a file.
        - :meth:`~graphviz.Digraph.view`: Open the graph in a viewer.
    """
    try:
        import graphviz
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The package `graphviz` is required to be installed to use this function. "
            "Please `pip install graphviz` or `conda install -c conda-forge graphviz`."
        ) from None

    # The graph style as well as the color scheme below was copied from https://github.com/szagoruyko/pytorchviz/
    # https://github.com/szagoruyko/pytorchviz/blob/0adcd83af8aa7ab36d6afd139cabbd9df598edb7/torchviz/dot.py#L78-L85
    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="10",
        ranksep="0.1",
        height="0.2",
        fontname="monospace",
    )
    graph = graphviz.Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    for node in to_nodes(dp):
        if not node.parents:
            fillcolor = "lightblue"
        elif not node.childs:
            fillcolor = "darkolivegreen1"
        else:
            fillcolor = None

        graph.node(name=repr(node), label=str(node), fillcolor=fillcolor)

        for child in node.childs:
            graph.edge(repr(node), repr(child))

    return graph
