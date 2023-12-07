###### Hybrid Genetic Algorithm to find the best possible route based on time travel , ease of driving and route distance. #######
import numpy as np 
import networkx as nx

class HybridGA():
    def __init__(self, start, finish,graph, population_size, generations, mutation_rate):
        self._start = start 
        self._finish= finish
        self._graph = graph 
        self._population = population_size
        self._generations = generations
        self._mutation_rate = mutation_rate 

        
    def calculate_fitness(self,path):
        travel_time = 0 
        ease_of_driving = 0
        
        for i in range(len(path)-1):
            source_node = path[i]
            target_node = path[i+1]

            if self._graph.has_edge(source_node,target_node):
                edge_attributes = self._graph[source_node][target_node]
                travel_time += edge_attributes['time']
                ease_of_driving += edge_attributes['ease_of_driving']
            else: 
                #Handle the case for where the edge does not exist for example assign a penalty 
                travel_time +=1000
                ease_of_driving +=0 
        return -travel_time - ease_of_driving
    
    def initialize_population(self,nodes):
        return [list(np.random.permutation(nodes)) for _ in range(self._population)]
    
    def crossover(self,p1, p2):
        crossover_point = np.random.randint(1,len(p1)-1)
        child = p1[:crossover_point] + [gene for gene in p2 if gene not in p1[:crossover_point]]
       
        #Ensure the child only contains the valid arcs 
        child = [node for node in child if node in self._graph.nodes]
        return child
    
    def mutate(self, individual):
        if np.random.rand() < self._mutation_rate:
            mutation_point1, mutation_point2 = np.random.choice(len(individual), 2, replace=False)
            individual[mutation_point1], individual[mutation_point2] = individual[mutation_point2] , individual[mutation_point1]
        
        individual = [node for node in individual if node in self._graph.nodes]
        return individual
    
    def update_pareto_front(self,pareto, offspring):
        #Combine current pareto front and offspring
        combined_front = pareto + offspring

        #Identify non-dominated solutions using Pareto dominance 
        non_dominated_front = [] 
        for sol in combined_front:
            if all(self.calculate_fitness(sol) >= self.calculate_fitness(sol2) for sol2 in combined_front):
                non_dominated_front.append(sol)
        return non_dominated_front


    def is_valid_path(self,path):
        for i in range(len(path)-1):
            sn = path[i]
            tn = path[i+1]
            if not self._graph.has_edge(sn,tn):
                return False
        
        return True
    
    def steady_state_GA(self, replacement_rate):
        nodes = list(self._graph.nodes()) 
        population = self.initialize_population(nodes)

        # Create the initial Pareto front with the first solution
        pareto_front = [population[0]]

        for generation in range(self._generations):
           #Evaluate fitness for each individual 
            fitness_value = [self.calculate_fitness(individual + [self._finish])for individual in population]
            selected_indices = np.argsort(fitness_value)[-int(self._population/2):]
            parents = [population[i] for i in selected_indices]
  
            # Create offspring using crossover and mutation
            offspring = []
            while len(offspring) < int(replacement_rate * self._population):
                index1, index2 = np.random.choice(len(parents), 2, replace=False)
                parent1, parent2 = parents[index1], parents[index2]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)

                if self.is_valid_path(child):
                    offspring.append(child)
        
        # Update Pareto front
        pareto_front = self.update_pareto_front(pareto_front, offspring)
        
        best_route = max(pareto_front,key=lambda x:self.calculate_fitness(x + [self._finish]))
        if best_route[-1] == self._finish:
            return best_route
        
        
        replacement_indices = selected_indices[:int(replacement_rate * self._population)]
        for i, index in enumerate(replacement_indices):
            population[index] = offspring[i]
        
        # Select the best route from the final Pareto front
        best_route = max(pareto_front, key=lambda x: self.calculate_fitness(x+ [self._finish] ))
        return best_route


    def GA(self):
        nodes = list(self._graph.nodes())
        population = self.initialize_population(nodes)

        for generation in range(self._generations):

            fitness_value = [self.calculate_fitness(individual + [self._finish])for individual in population]
            selected_indices = np.argsort(fitness_value)[-int(self._population/2):]
            parents = [population[i] for i in selected_indices]
            
            offspring = []
            while len(offspring) < self._population - len(parents):
                ind1,ind2 = np.random.choice(len(parents),2,replace=False)
                p1,p2 = parents[ind1],parents[ind2]
                child = self.crossover(p1,p2)
                child = self.mutate(child)
                offspring.append(child)

            population = parents+offspring
        
        best_route = max(population, key=lambda x: self.calculate_fitness(x+[self._finish]))
        return best_route + [self._finish]
    
G = nx.DiGraph()
G.add_edges_from([(1, 2, {'time': 10, 'ease_of_driving': 8}),
                  (2, 3, {'time': 15, 'ease_of_driving': 7}),
                  (3, 4, {'time': 20, 'ease_of_driving': 6}),
                  (4, 1, {'time': 5, 'ease_of_driving': 9}),
                  (2, 4, {'time': 25, 'ease_of_driving': 5})])


start = 3
finish = 1 

for node in ((G.nodes)):
    G.nodes[node]['order'] = 0
    if node == start:
        G.nodes[node]['order'] = -1000
   
ga = HybridGA(start, finish, G, population_size=50, generations=100, mutation_rate=0.2)
best_route = ga.steady_state_GA(replacement_rate=0.2)
sorted_best_route = sorted(best_route, key=lambda x:G.nodes[x]['order'])
if sorted_best_route[0]!= start and sorted_best_route[-1]!= finish:
    print("Error: Start or finish node is not in the best route")
elif sorted_best_route[0] == start :
    sorted_best_route = [(sorted_best_route[i],sorted_best_route[i+1]) for i in range(len(sorted_best_route)-1) if sorted_best_route[i] != finish]
    print(sorted_best_route)
    print([(sorted_best_route[i],sorted_best_route[i+1]) for i in range(len(sorted_best_route)-1)])
    total_time = 0 
    ease_of = 0
    for u in sorted_best_route:
        total_time += G[u[0]][u[1]]['time']
        ease_of += G[u[0]][u[1]]['ease_of_driving']
    print("Best Route:", sorted_best_route)
    print("Total Time:", total_time)
    print("Total Ease of Driving:",ease_of)
