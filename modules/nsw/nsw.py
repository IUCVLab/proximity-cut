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
        
    def __repr__(self):
        return f"`#{self.idx}: '{self.value}' ~ [{self.neighbourhood}]`"
        
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
        
    def search_nsw_basic(self, query, visitedSet, candidates, result, top=5, guard_hops=100, callback=None):
        ''' basic algorithm, takes vector query and returns a pair (nearest_neighbours, hops)'''

        # taking random node as an entry point
        tmpResult = sortedcontainers.SortedList()
        entry = random.randint(0, len(self.nodes) - 1)
        if entry not in visitedSet:
            candidates.add((self.dist(query, self.nodes[entry].value), entry))
        tmpResult.add((self.dist(query, self.nodes[entry].value), entry))
        
        hops = 0
        while hops < guard_hops:
            hops += 1
            if len(candidates) == 0: break
            
            # 6 get element c closest from candidates (see paper 4.2.)
            # 7 remove c from candidates
            closest_dist, сlosest_id = candidates.pop(0)
            
            # k-th best of global result
            # new stop condition from paper
            # if c is further than k-th element from result
            # than break repeat
            #! NB this statemrnt from paper will not allow to converge in first run.
            #! thus we use tmpResult if result is empty
            if len(result or tmpResult) >= top:
                if (result or tmpResult)[top-1][0] < closest_dist: break

            #  for every element e from friends of c do:
            for e in self.nodes[сlosest_id].neighbourhood:
                # 13 if e is not in visitedSet than
                if e not in visitedSet:                   
                    d = self.dist(query, self.nodes[e].value)
                    # 14 add e to visitedSet, candidates, tempRes
                    visitedSet.add(e)
                    candidates.add((d, e))
                    tmpResult.add((d, e))
                    
            if callback is not None:
                callback(self.nodes[сlosest_id].value, tmpResult)

        return tmpResult, hops
    
    def search_nsw_basic_wrapped(self, query, top=5, guard_hops=100, callback=None):
        visitedSet, candidates, result = set(), sortedcontainers.SortedList(), sortedcontainers.SortedList()
        tmpResult, hops = self.search_nsw_basic(query, visitedSet, candidates, result, top, guard_hops, callback)
        return [v for k, v in tmpResult[:top]], hops
    
    def multi_search(self, query, attempts=1, top=5):   
        '''Implementation of `K-NNSearch`, but without keeping the visitedSet'''

        # share visitedSet among searched. Paper, 4.2.p2
        visitedSet, candidates, result = set(), sortedcontainers.SortedList(), sortedcontainers.SortedList()
        
        for i in range(attempts):
            closest, hops = self.search_nsw_basic(query, visitedSet, candidates, result, top=top)
            result.update(closest)
            result = sortedcontainers.SortedList(set(result))
            
        return [v for k, v in result[:top]]
    
    def build_navigable_graph(self, values, attempts=3, verbose=False):
        '''Accepts container with values. Returns list with graph nodes'''
        # create graph with one node
        self.nodes.append(Node(values[0][0], len(self.nodes), values[0][1]))
        
        # The tests indicate [36] that at least for Euclid data with
        # d = 1...20, the optimal value for number of neighbors to
        # connect (f) is about 3d
        d = len(values[0][0])
        f = 3 * d
        if verbose:
            print(f"Data dimensionality detected is {d}. regularity = {f}")
        
        # insert the remaining nodes one at a time
        for i in range(1, len(values)):
            val = values[i][0]
            # search f nearest neighbors of the current value existing in the graph         
            closest = self.multi_search(val, attempts, f)
            # create a new node
            node = Node(val, len(self.nodes), values[i][1])
            self.nodes.append(node)
            # connect the closest nodes to the current node
            node.neighbourhood.update(closest)
            # connect the current node to the closest values
            for c in closest:
                self.nodes[c].neighbourhood.add(len(self.nodes) - 1)
            if verbose:
                if i * 10 % len(values) == 0:
                    print(f"\t{100 * i / len(values):.2f}% of graph construction")
    
    def plot(self, edgeborder=0.2):
        plt.figure(figsize=(10, 10))     
        middles = []
        for e in self.get_edges():
            n0, n1 = self.nodes[e[0]], self.nodes[e[1]]
            sx, sy = n0.value
            fx, fy = n1.value
            # plot only relatively short edges
            if self.dist(n0.value, n1.value) > edgeborder:
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

        
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


if __name__ == "__main__":
    print("Module NSW launched as program.")