import heapq

def heuristic(a, b):
   
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, neighbors_func):
   
    frontier = []
    heapq.heappush(frontier, (0, start))
    
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        _, current = heapq.heappop(frontier)
        
        if current == goal:
            break
        
        for neighbor in neighbors_func(current):
            new_cost = cost_so_far[current] + 1 
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current
    
   
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    return path


def grid_neighbors(node):
    x, y = node
    moves = [(1,0), (-1,0), (0,1), (0,-1)]
    result = []
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 5 and 0 <= ny < 5:
            result.append((nx, ny))
    return result

start = (0, 0)
goal = (4, 4)
path = astar(start, goal, grid_neighbors)
print("Shortest path:", path)