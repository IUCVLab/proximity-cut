from collections import Counter

def get_hvdm(dataset, q=2):
    size = len(dataset)
    dim = len(dataset[0][0])
    classes = set(v[1] for v in dataset)
    print(f"size={size}, dim={dim}, classes={classes}")
    
    result = [None for _ in range(dim)]
    for i, x in enumerate(dataset[0][0]):
        if not (type(x) == float or type(x) == int):
            result[i] = dict()

    
    for i in range(dim):
        # this is category
        if result[i] is not None:
            # take all feature categories
            cats = sorted(list(set(v[0][i] for v in dataset)))
            # construct frequency stats for dataset
            N_x = Counter()
            N_xc = Counter()
            for row in dataset:
                c = row[1]
                x = row[0][i]
                N_x[x] += 1
                N_xc[(x, c)] += 1
            # implement all VDM pairs
            for x in cats:
                for y in cats:
                    d = 0
                    if x != y:
                        for c in classes:
                            d += abs(N_xc[(x, c)] / N_x[x] - N_xc[(y, c)] / N_x[y]) ** q    
                    result[i][(x, y)] = d
                    result[i][(y, x)] = d
    
    def f_hvdm(x, y):
        d = 0
        for i in range(len(x)):
            if result[i] is not None:
                d += result[i][(x[i], y[i])]
            else:
                d += abs(x[i] - y[i]) ** q
        return d
    
    return f_hvdm