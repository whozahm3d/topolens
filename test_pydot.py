import networkx as nx
import traceback

G = nx.complete_graph(5)
try:
    import pydot
    print("Trying nx.drawing.nx_pydot.graphviz_layout...")
    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='sfdp')
    print("Success! Pos keys:", list(pos.keys()))
except Exception as e:
    print("pydot layout failed with:")
    traceback.print_exc()
