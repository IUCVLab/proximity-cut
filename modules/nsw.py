import random
import sortedcontainers
import numpy as np
from scipy.spatial import distance
from numpy.linalg import norm as vector_norm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class Node:
    ''' Graph node class. Major properties are `value` to access embedding and `neighbourhood` for adjacent nodes '''
    def __init__(self, value, idx, _cls):
        self.value = value
        self._class = _cls
        self.idx = idx
        self.neighbourhood = set()
        
class NSWGraph:
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def __init__(self, values=None, dist=None):
        self.dist = dist if dist else self.eucl_dist
        self.nodes = [Node(node[0], i, node[1]) for i, node in enumerate(values)] if values else []

    def neg_dot_prod(self, v1, v2):
        return -np.dot(v1, v2)

    def eucl_dist(self, v1, v2):
        return distance.euclidean(v1, v2)
        
    def get_edges(self):
        for i, node in enumerate(self.nodes):
            for n in node.neighbourhood:
                yield (i, n)
        
    def search_nsw_basic(self, query, top=5, guard_hops=100, callback=None):
        ''' basic algorithm, takes vector query and returns a pair (nearest_neighbours, hops)'''
        candidates = sortedcontainers.SortedList()
        result = sortedcontainers.SortedList()
        visitedSet = set()

        # taking random node as an entry point
        current = random.randint(0, len(self.nodes) - 1)
        candidates.add((self.dist(query, self.nodes[current].value), current))
        result.add((self.dist(query, self.nodes[current].value), current))

        hops = 0
        closest_candidate_ever = None
        while hops < guard_hops:
            hops += 1
            if len(candidates) == 0: break
            closest_sim, сlosest_id = candidates[0]
            if closest_candidate_ever == сlosest_id: break
            closest_candidate_ever = сlosest_id
            # k-th best
            if len(result) >= top:
                if result[top-1][0] < closest_sim: break

            for friend in self.nodes[сlosest_id].neighbourhood:
                if friend not in visitedSet:
                    visitedSet.add(friend)
                    sim = self.dist(query, self.nodes[friend].value)
                    candidates.add((sim, friend))
                    result.add((sim, friend))
                    
            if callback is not None:
                callback(self.nodes[friend].value, candidates)

        return [v for k, v in result[:top]], hops
    
    def multi_search(self, query, attempts=1, top=5):   
        '''Implementation of `K-NNSearch`, but without keeping the visitedSet'''
        result = set()
        for i in range(attempts):
            closest, hops = self.search_nsw_basic(query, top=top)
            result.update(closest)    
        index = list((i, self.dist(query, self.nodes[i].value)) for i in result)    
        sorted_index = sorted(index, key=lambda pair: pair[1])[:top]
        return [x[0] for x in sorted_index]
    
    def build_navigable_graph(self, values, K=5, attempts=3):
        '''Accepts container with values. Returns list with graph nodes'''
        # create graph with one node
        self.nodes.append(Node(values[0][0], len(self.nodes), values[0][1]))
        # insert the remaining nodes one at a time
        for i in range(1, len(values)):
            val = values[i][0]
            # search K nearest neighbors of the current value existing in the graph
            top_k = min(len(self.nodes), K) # for the first K insertions            
            closest = self.multi_search(val, attempts, top_k)
            # create a new node
            self.nodes.append(Node(val, len(self.nodes) + 1, values[i][1]))
            # connect the closest nodes to the current node
            self.nodes[len(self.nodes) - 1].neighbourhood.update(closest)
            # connect the current node to the closest values
            for c in closest:
                self.nodes[c].neighbourhood.add(len(self.nodes) - 1)
    
    def plot(self):
        plt.figure(figsize=(10, 10))     
        middles = []
        for e in self.get_edges():
            n0, n1 = self.nodes[e[0]], self.nodes[e[1]]
            sx, sy = n0.value
            fx, fy = n1.value
            # plot only relatively short edges
            if self.dist(n0.value, n1.value) > 0.2:
                continue

            if n0._class != n1._class:
                plt.plot([sx, fx], [sy, fy], linewidth=1, c='#FFAA00')
                middles.append((n0.value + n1.value) / 2.)
            else:
                plt.plot([sx, fx], [sy, fy], linewidth=1, c='#DDDDDD')

        markers = ['o', '>']
        for i, m in enumerate(markers):
            x = [node.value[0] for node in self.nodes if node._class == i]
            y = [node.value[1] for node in self.nodes if node._class == i]
            plt.scatter(x, y, marker=m, s=20)
            
        plt.scatter([v[0] for v in middles], [v[1] for v in middles], marker='*', s=100)
        
        plt.show()

    def multi_search_and_plot(self, query, attempts=1, top_k=5):
        ids = self.multi_search(query, attempts, top_k)
        n = np.array([query] + [self.nodes[id].value for id in ids])
        c = [2] + [self.nodes[id]._class for id in ids]
        
        data = {}
        data['x'] = n[:,0]
        data['y'] = n[:,1]
        data['class'] = c
        plt.figure(figsize=(8,5))
        sns.scatterplot(
            x="x", y="y",
            style="class", hue="class",
            data=data,
            legend="full",
            alpha=0.7
        )
        plt.show()
        
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


if __name__ == "__main__":
    print("Module NSW launched as program.")