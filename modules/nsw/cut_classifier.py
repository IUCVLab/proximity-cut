import sys
import random
import time
import sortedcontainers
import numpy as np
from collections import Counter
from nsw.nsw import Node, NSWGraph
from nsw.nsw_classifier import NSWClassifier
from nsw import rbf

class CutClassifier:
    
    def __init__(self, G, cut, verbose=True, wilson=True):
        print(f"Graph initialized with cut ({len(cut)}).")
        self.graph = G
        self.cut = cut
        if wilson:
            self.clean_cut = self.wilson()
        else:
            self.clean_cut = self.cut
        print(f"Clean cut ({len(self.clean_cut)}).")
        self.eps = self.get_mid_shortest_dist()
        print(f"Shortest dist estimated ({self.eps:.4f}).")
        self.support = self.get_support(N=len(self.graph.nodes) // 10)
        print(f"Support with {len(self.support)} nodes is created.")
        self.support_nsw = NSWGraph()
        self.support_nsw.build_navigable_graph(self.support, attempts=5, verbose=verbose)
        print("Support graph is built.")
        self.classifier = self.get_classifier_function()
        print("Classifier function is ready.")
    
    def wilson(self):
        print("Wilson: Data shape", self.graph.nodes[0].value.shape[0])
        cutset = set([r[0][0] for r in self.cut] + [r[0][1] for r in self.cut])
        local_cuts = dict((key, []) for key in cutset)
        for e in self.cut:
            local_cuts[e[0][0]].append(e)
            local_cuts[e[0][1]].append(e)

        clean_cut = set(self.cut)
        for k, lst in local_cuts.items():
            # maybe we should vary a share of nodes here. 
            if len(lst) <= self.graph.nodes[0].value.shape[0] * 4 / 2: 
                continue
            for e in lst:
                if e in clean_cut: 
                    clean_cut.remove(e)
        return list(clean_cut)

    def get_mid_shortest_dist(self, sample=50):
        dists = []
        for i in range(sample):
            n0 = random.choice(self.clean_cut)
            p0 = (self.graph.nodes[n0[0][0]].value + self.graph.nodes[n0[0][1]].value) / 2
            sd = []
            for n1 in self.clean_cut:
                if n0 == n1:
                    continue
                p1 = (self.graph.nodes[n1[0][0]].value + self.graph.nodes[n1[0][1]].value) / 2
                d = self.graph.dist(p1, p0)
                sd.append(d)
            dists.append(min(sd))
        dists.sort()
        return dists[len(dists) // 2]
    
    def get_support(self, N=50):
        support = []
        for i in range(N):
            n = random.choice(self.graph.nodes)
            support.append((n.value, n._class))
        return support
    
    def get_classifier_function(self, classes=None):
        '''Returns REAL 0..1 value of belonging to a class. Based on [Gradient theorem](https://en.wikipedia.org/wiki/Gradient_theorem)'''
    
        grad = rbf.get_grad_field_function(self.graph, self.clean_cut)
        
        
        def f(x, R=2, small=0.001, closest=5, M=10, callback=None):
            # step 1. Closest support items
            vs, cs, rs = set(), sortedcontainers.SortedList(), sortedcontainers.SortedList()
            top, hops = self.support_nsw.search_nsw_basic(x, vs, cs, rs,  top=closest)
            top = top[:closest]
            
            # TBD implement multiclass voting
            votes = {0: 0, 1: 0}
            monitor = dict()
            
            for d, n in top[:closest]:
                val, class_ = self.support_nsw.nodes[n].value, self.support_nsw.nodes[n]._class
                r = x - val
                integral = 0.
                # step 2. Get integral along (d0 .. x) vector
                for i in range(M):
                    p = val + (r * i / M)
                    integral += np.dot(grad(p, self.eps * R), r)
                # print(f"<i={integral}>~<{class_}>\t", end="", )
                if abs(integral) < small:
                    votes[class_] += 1
                    monitor[n] = 0
                elif integral > small:
                    votes[1] += 1
                    monitor[n] = 1
                else:
                    votes[0] += 1
                    monitor[n] = -1
            # print(votes, end="")
            if callback is not None:
                callback(self, x, monitor)
            return votes[1] / (votes[0] + votes[1])
        return f