import time
import matplotlib.pyplot as plt

def analyze_classifier(classifier_method, test, label, iterations=7):
    clf_times = []
    hits = 0
    times = iterations

    for i in range(times):
        print(f"iteration {i}")
        s = time.perf_counter()
        for t in test:
            hits += classifier_method(t[0]) == t[1]
        f = time.perf_counter()
        clf_times.append(1000 * (f - s) / len(test))
        print(f"{label}, {hits}/{len(test) * (i+1)}, time {f-s:.3f}")
        
    print(f"{label} classifier accuracy = {100 * hits / len(test) / times:.2f}%")

    plt.xlabel("iteration #", fontsize=14)
    plt.ylabel("avg time, ms", fontsize=14)
    plt.plot(range(len(clf_times)), clf_times, 'k--', label=label)
    plt.xticks(range(0, len(clf_times)+1, 5))
    plt.legend()
    plt.show()
    
    # print("\t".join([str(x) for x in clf_times]))