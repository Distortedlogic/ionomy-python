import random
from inspect import isclass

def mutNodeReplacement(individual, pset):
    if len(individual) < 2:
        return individual,

    found = False
    prim_indices = [idx for idx, node in enumerate(individual) if node.arity != 0]
    while not found and prim_indices:
        random.shuffle(prim_indices)
        index = prim_indices.pop()
        node = individual[index]

        prims = [p for p in pset.primitives[node.ret] if p.args == node.args and type(p) != type(node)]
        if len(prims) > 1:
            individual[index] = random.choice(prims)
            found = True

    return individual,
