def bfs(graph, start):
    visited = []
    to_visit = [start]  

    while to_visit:
        current = to_visit[0]  
        to_visit = to_visit[1:]

        if current not in visited:
            print(current, end=" ") 
            visited.append(current)

           
            for neighbor in graph[current]:
                if neighbor not in visited and neighbor not in to_visit:
                    to_visit.append(neighbor)



graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": ["F"],
    "F": []
}

print("BFS Traversal:")
bfs(graph, "A")