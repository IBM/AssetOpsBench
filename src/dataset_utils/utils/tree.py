import copy
from typing import Optional


class OPS:
    OR = "OR"
    AND = "AND"


class Node:
    def __init__(self, ops: str, statement: Optional[str], manual=False) -> None:
        self.children = []
        self.val = None
        self.ops = ops
        self.statement = statement

        if not manual:
            self.build_graph()

    def build_graph(self):
        if self.statement:
            split = self.statement.split(self.ops)
            if len(split) >= 2:
                self.children = [
                    Node(self.ops, split[0]),
                    Node(self.ops, self.ops.join(split[1:])),
                ]
                self.val = self.ops
            else:
                self.val = self.statement

    def __copy__(self):
        _temp = Node(ops=self.ops, statement=self.statement, manual=True)

        return _temp

    @staticmethod
    def generate_graph(statement: str):
        root = None

        for oi, o in enumerate([OPS.OR]):
            if oi == 0:
                root = Node(o, statement)

                return root


def _print_tree(root: Node, level=0, prefix="Root: ") -> str:
    if root is not None and root.val:
        _s = "    " * level + prefix + root.val + "\n"
        for x in root.children:
            _s += _print_tree(x, level + 1, "L---- ")

        return _s
    return ""


def print_tree(root):
    concat_string = _print_tree(root)
    print(concat_string)


def get_tree_repr(root: Node):
    _temp = copy.deepcopy(root)
    _temp.val = ""
    concat_string = _print_tree(_temp, prefix="")
    return concat_string


def traverse_tree(root: Node) -> Node:
    _temp = copy.deepcopy(root)
    recursive_traverse_tree(_temp)
    concat_tree(None, _temp)
    return _temp


def concat_tree(parent: Optional[Node], child: Node):
    if child.children:
        for c in child.children:
            concat_tree(child, c)

        if len(child.children) == 1 and parent:
            parent.children.remove(child)
            parent.children.append(child.children[0])


def recursive_traverse_tree(root: Node) -> None:
    if root.children:
        for c in root.children:
            recursive_traverse_tree(c)

        if root.val == OPS.OR:
            sel_ind = 0
            root.children = [root.children[sel_ind]]
        else:
            root.children = root.children


# Function to find the root parent of a node with path compression
def findParent(parent, x):
    if parent[x] == x:
        return x

    # Path compression
    parent[x] = findParent(parent, parent[x])
    return parent[x]


# Function to unite two subsets
def unionSets(parent, x, y):
    px = findParent(parent, x)
    py = findParent(parent, y)
    if px != py:
        # Union operation
        parent[px] = py


def getComponents(V, edges):

    # Initialize each node as its own parent
    parent = [i for i in range(V)]

    # Union sets using the edge list
    for edge in edges:
        unionSets(parent, edge[0], edge[1])

    # Apply path compression for all nodes
    for i in range(V):
        parent[i] = findParent(parent, i)

    # Group nodes by their root parent
    resMap = {}
    for i in range(V):
        root = parent[i]
        if root not in resMap:
            resMap[root] = []
        resMap[root].append(i)

    # Collect all components into a result list
    res = list(resMap.values())

    return res
