"""Journal class for storing ML-Master solution tree"""

import copy
import logging
from pathlib import Path
from typing import Optional

from .metric import MetricValue
from .node import Node

logger = logging.getLogger(__name__)


class Journal:
    """A collection of nodes representing the solution tree.

    The journal is the core data structure that contains:
    - The generated code samples
    - Information about how code samples relate (tree structure)
    - Code execution results
    - Evaluation information such as metrics
    """

    def __init__(self):
        self.nodes: list[Node] = []
        self.draft_nodes: list[Node] = []

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal.

        Args:
            node: The node to append
        """
        node.step = len(self.nodes)
        self.nodes.append(node)

        # Track draft nodes separately
        if node.parent is None or (hasattr(node, 'stage') and node.stage == "draft"):
            self.draft_nodes.append(node)

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not buggy."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[MetricValue]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good: bool = True) -> Optional[Node]:
        """Return the best solution found so far.

        Args:
            only_good: If True, only consider non-buggy nodes

        Returns:
            The node with the best metric value, or None if no valid nodes
        """
        if only_good:
            nodes = self.good_nodes
        else:
            nodes = self.nodes

        if not nodes:
            return None

        # Filter out nodes with None metric values
        valid_nodes = [n for n in nodes if n.metric.value is not None]
        if not valid_nodes:
            return None

        return max(valid_nodes, key=lambda n: n.metric)

    def generate_summary(self, include_code: bool = False) -> str:
        """Generate a summary of the journal for the agent.

        Args:
            include_code: If True, include code in the summary

        Returns:
            A formatted summary string
        """
        summary = []
        for n in self.good_nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            if n.analysis:
                summary_part += f"Results: {n.analysis}\n"
            if n.metric.value is not None:
                summary_part += f"Validation Metric: {n.metric.value}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)

    def generate_summary_from_node(self, target_node: Node, include_code: bool = False) -> str:
        """Generate a summary based on the parent/target node.

        Args:
            target_node: The node to generate summary from
            include_code: If True, include code in the summary

        Returns:
            A formatted summary string
        """
        summary = []
        history_nodes = []
        related_nodes = []

        # Get all ancestors
        current_node = target_node
        while current_node.parent:
            current_node = current_node.parent
            history_nodes.append(current_node)

        # Get siblings
        if target_node.parent:
            related_nodes = [n for n in target_node.parent.children if n != target_node][:5]

        history_nodes = history_nodes + related_nodes

        for n in history_nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            if n.analysis:
                summary_part += f"Results: {n.analysis}\n"
            if n.metric.value is not None:
                summary_part += f"Validation Metric: {n.metric.value}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID.

        Args:
            node_id: The node ID to find

        Returns:
            The node with the given ID, or None if not found
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None


def get_path_to_node(journal: Journal, node_id: str) -> list[str]:
    """Get the path from root to the given node.

    Args:
        journal: The journal containing the node
        node_id: The ID of the target node

    Returns:
        A list of node IDs from root to target (inclusive)
    """
    path = [node_id]
    target_node = journal.get_node_by_id(node_id)

    if not target_node:
        return path

    node2parent = {n.id: n.parent.id for n in journal.nodes if n.parent is not None}

    while node_id in node2parent:
        parent_id = node2parent[node_id]
        path.append(parent_id)
        node_id = parent_id

    return path[::-1]


def get_longest_path(journal: Journal) -> list[str]:
    """Get the longest path from root to any leaf.

    Args:
        journal: The journal to search

    Returns:
        A list of node IDs representing the longest path
    """
    longest_path = []
    for node in journal.nodes:
        path = get_path_to_node(journal, node.id)
        if len(path) > len(longest_path):
            longest_path = path
    return longest_path


def filter_on_path(journal: Journal, path: list[str]) -> Journal:
    """Filter journal to only include nodes on the given path.

    Args:
        journal: The journal to filter
        path: List of node IDs to keep

    Returns:
        A new journal containing only nodes on the path
    """
    journal_copy = Journal()
    journal_copy.nodes = [n for n in journal.nodes if n.id in path]

    # Redact sensitive info
    for n in journal_copy.nodes:
        if hasattr(n, '_term_out'):
            n._term_out = "<OMITTED>"
        if hasattr(n, 'exc_stack'):
            n.exc_stack = "<OMITTED>"

    return journal_copy


def filter_for_best_path(journal: Journal, best_node_id: str) -> Journal:
    """Filter journal to only include nodes on the path to the best node.

    Args:
        journal: The journal to filter
        best_node_id: The ID of the best node

    Returns:
        A new journal containing only nodes on the path to best
    """
    path_to_best = get_path_to_node(journal, best_node_id)
    return filter_on_path(journal, path_to_best)


def filter_journal(journal: Journal) -> Journal:
    """Filter journal to the best path or longest path if no valid best node.

    Args:
        journal: The journal to filter

    Returns:
        A filtered journal
    """
    best_node = journal.get_best_node(only_good=True)

    if best_node is not None:
        return filter_for_best_path(journal, best_node.id)
    else:
        longest_path = get_longest_path(journal)
        return filter_on_path(journal, longest_path)


def journal_to_string_tree(journal: Journal) -> str:
    """Convert journal to a string representation of the tree.

    Args:
        journal: The journal to convert

    Returns:
        A string representation of the solution tree
    """
    best_node = journal.get_best_node()
    tree_str = "Solution tree\n"

    def append_rec(node: Node, level: int) -> None:
        nonlocal tree_str
        indent = "  " * level
        if node.is_buggy:
            s = f"{indent}◍ bug (ID: {node.id})\n"
        else:
            markers = []
            if best_node and node.id == best_node.id:
                markers.append("best")
            marker_str = " & ".join(markers) if markers else ""
            if marker_str and node.metric.value is not None:
                s = f"{indent}● {node.metric.value:.4f} ({marker_str}) (ID: {node.id})\n"
            elif node.metric.value is not None:
                s = f"{indent}● {node.metric.value:.4f} (ID: {node.id})\n"
            else:
                s = f"{indent}● None (ID: {node.id})\n"
        tree_str += s
        for child in sorted(node.children, key=lambda c: c.id):
            append_rec(child, level + 1)

    for n in journal.draft_nodes:
        append_rec(n, 0)

    return tree_str
