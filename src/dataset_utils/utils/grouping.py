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
