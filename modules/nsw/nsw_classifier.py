import sys
import random
import time
import sortedcontainers
from collections import Counter
from nsw.nsw import Node, NSWGraph


class NSWClassifier(NSWGraph):
    
    def __init__(self):
        super().__init__()
        self.cut = set()

    def build_navigable_graph(self, values, attempts=3, verbose=False):
        self.nodes.append(Node(values[0][0], len(self.nodes), values[0][1]))
        d = len(values[0][0])
        f = 3 * d
        if verbose:
            print(f"Classifier is building a graph. Data dimensionality detected is {d}. regularity = {f}")
        
        start = time.time()
        # insert the remaining nodes one at a time
        for i in range(1, len(values)):
            val = values[i][0]
            closest = self.multi_search(val, attempts, f)
            node = Node(val, len(self.nodes), values[i][1])
            self.nodes.append(node)
            node.neighbourhood.update(closest)
            for c in closest:
                self.nodes[c].neighbourhood.add(len(self.nodes) - 1)
                if node._class != self.nodes[c]._class:
                    self.cut.add((node.idx, c))
            if verbose:
                if i * 10 % len(values) == 0:
                    print(f"\t{100 * i / len(values):.2f}% of graph construction")
            
        end = time.time()
        print(f"Classifier graph is build in {end - start:.3f}s")

    def classify_by_path_basic(self, query, guard_hops=100, callback=None):
        visitedSet, candidates, tmpResult = set(), sortedcontainers.SortedList(), sortedcontainers.SortedList()
        entry = random.randint(0, len(self.nodes) - 1)
        candidates.add((self.dist(query, self.nodes[entry].value), entry))
        tmpResult.add((self.dist(query, self.nodes[entry].value), entry))
        
        hops = 0    
        #distance from first
        closest_dist_ever = candidates[0][0]
        class_ = self.nodes[entry]._class
        while hops < guard_hops:
            hops += 1
            if len(candidates) == 0: break
            closest_dist, сlosest_id = candidates.pop(0)
                
            if closest_dist_ever < closest_dist: break
            closest_dist_ever = closest_dist
            class_ = self.nodes[сlosest_id]._class
            
            for e in self.nodes[сlosest_id].neighbourhood:
                if e not in visitedSet:                   
                    d = self.dist(query, self.nodes[e].value)
                    visitedSet.add(e)
                    candidates.add((d, e))
                    tmpResult.add((d, e))
                    
            if callback is not None:
                callback(self.nodes[сlosest_id].value, tmpResult)

        return class_

    def classify_by_path_basic_no_heap(self, query, guard_hops=100):
        entry = random.randint(0, len(self.nodes) - 1)
        closest, closest_dist = entry, self.dist(query, self.nodes[entry].value)

        hops = 0    
        while hops < guard_hops:
            hops += 1
            cl = closest
            for e in self.nodes[cl].neighbourhood:
                d = self.dist(query, self.nodes[e].value)
                if d < closest_dist:
                    closest, closest_dist = e, d
            if cl == closest:
                break
        return self.nodes[closest]._class
    
    
    def classify_by_path(self, query, attempts=5, top=5):
        result = Counter()
        for i in range(attempts):
            # c = self.classify_by_path_basic(query)
            c = self.classify_by_path_basic_no_heap(query)
            result[c] += 1
        most_common = result.most_common(1)[0]
        
        # not confident
        if most_common[1] * 2 < attempts:
            return None
        
        return most_common[0]
    
    def classify_fuzzy_by_path(self, query, attempts=5, top=5):
        result = Counter()
        for i in range(attempts):
            # c = self.classify_by_path_basic(query)
            c = self.classify_by_path_basic_no_heap(query)
            result[c] += 1
        return {(k, v / attempts) for k, v in result.items()}
                        
    def classify_knn(self, query, attempts=5, k=11):
        top = self.multi_search(query, attempts, k)
        classes = Counter([self.nodes[i]._class for i in top])
        most_common = classes.most_common(1)[0]
        
        # not confident
        if most_common[1] * 2 < k:
            return None
        
        return most_common[0]
    
    def classify_fuzzy_knn(self, query, attempts=5, k=11):
        top = self.multi_search(query, attempts=attempts, top=k)
        print(top)
        classes = Counter([self.nodes[i]._class for i in top])
        
        return {(key, val / k) for key, val in classes.items()}