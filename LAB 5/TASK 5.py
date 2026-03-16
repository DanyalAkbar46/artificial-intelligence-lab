class Node:
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)


def dfs_with_stack(start_node):
    visited = set()
    stack = [start_node]

    while stack:
        current = stack.pop()
        if current not in visited:
            print(current.value, end=" ")  
            visited.add(current)

           
            for neighbor in reversed(current.neighbors):
                if neighbor not in visited:
                    stack.append(neighbor)



if __name__ == "__main__":
   
    A = Node("A")
    B = Node("B")
    C = Node("C")
    D = Node("D")
    E = Node("E")

    
    A.add_neighbor(B)
    A.add_neighbor(C)
    B.add_neighbor(D)
    B.add_neighbor(E)
    C.add_neighbor(E)

   
    print("DFS Traversal:")
    dfs_with_stack(A)