import sys, inspect, random
from deap import gp

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def graph(ind):
    plt.rcParams["figure.figsize"] = (50, 40)

    nodes, edges, labels = gp.graph(ind)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g)

    nx.draw_networkx_nodes(g, pos, node_size=20000, node_color='grey')
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels, font_size=50)
    plt.savefig('graph.png')
    plt.show()

def generate_safe(pset, min_, max_, terminal_types, type_=None):
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if type_ in terminal_types:
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                graph(gp.PrimitiveTree(expr))
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a terminal of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            if inspect.isclass(term):
                term = term()
            expr.append(term)
        else:
            primitives_without_terminal_args = [
                p for p in pset.primitives[type_] if
                all([arg not in terminal_types for arg in p.args])
            ]
            primitives_with_only_terminal_args = [
                    p for p in pset.primitives[type_] if
                    all([arg in terminal_types for arg in p.args])
                ]
            primitives_with_mixed_args = [
                p for p in pset.primitives[type_]
                if p not in primitives_without_terminal_args
                and p not in primitives_with_only_terminal_args
            ]
            try:
                # Might not be respected if there is a type without terminal args
                if height <= depth or (depth >= min_ and random.random() < pset.terminalRatio):
                    if len(primitives_with_only_terminal_args + primitives_with_mixed_args) == 0:
                        prim = random.choice(primitives_without_terminal_args)
                    else:
                        prim = random.choice(primitives_with_only_terminal_args + primitives_with_mixed_args)
                else:
                    if len(primitives_without_terminal_args + primitives_with_mixed_args) == 0:
                        prim = random.choice(primitives_with_only_terminal_args)
                    else:
                        prim = random.choice(primitives_without_terminal_args + primitives_with_mixed_args)
            except IndexError:
                graph(gp.PrimitiveTree(expr))
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a primitive of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))

    # graph(gp.PrimitiveTree(expr))
    return expr