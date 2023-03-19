import heapq
def find_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            new_paths = find_paths(graph, node, end, path)
            for p in new_paths:
                paths.append(p)
    return paths
###################################################################
def build_graph(adj_matrix):
	adj_matrix=list(adj_matrix)
    num_nodes = len(adj_matrix)
    graph = {}
    for i in range(num_nodes):
        node = str(i)
        neighbors = []
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                neighbors.append(str(j))
        graph[node] = neighbors
    return graph
##################################################################
def dijkstra(graph, start, end):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous_vertices = {node: None for node in graph}

    queue = [(0, start)]

    while queue:
        current_distance, current_vertex = heapq.heappop(queue)
		
        if current_vertex == end:
            path = []
            while previous_vertices[current_vertex] is not None:
                path.append(current_vertex)
                current_vertex = previous_vertices[current_vertex]
            path.append(start)
            path.reverse()
            return path
		
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_vertices[neighbor] = current_vertex
                heapq.heappush(queue, (distance, neighbor))
    return []
##################################################################################
def find_subsets(s):
    subsets = []
    for i in range(2**len(s)):
        subset = []
        for j in range(len(s)):
            if i & (1<<j):
                subset.append(s[j])
        subsets.append(subset)
    return subsets
#############################################
def find_maximum_clique(graph):
    vertices = list(graph.keys())
    subsets = find_subsets(vertices)
    max_clique = []
    for subset in subsets:
        is_clique = True
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                if subset[j] not in graph[subset[i]]:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique and len(subset) > len(max_clique):
            max_clique = subset
    return max_clique
###################################################################
