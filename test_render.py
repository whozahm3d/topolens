import os
import sys
import filecmp
import pandas as pd
import networkx as nx
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from render.render_graphs import render_graph
import config

def test():
    df = pd.read_csv('data/splits/test.csv')
    sample_ids = df['graph_id'].head(3).tolist()
    
    os.makedirs('temp_render', exist_ok=True)
    
    for graph_id in sample_ids:
        row = df[df['graph_id'] == graph_id].iloc[0]
        G = nx.read_graphml(row['graph_path'])
        
        print(f"Rendering {graph_id}...")
        res = render_graph(G, graph_id, 'temp_render')
        print(f"Layout used: {res['layout_algorithm']}")
        
        orig_img = f"data/images/{graph_id}.png"
        new_img = f"temp_render/{graph_id}.png"
        
        if not filecmp.cmp(orig_img, new_img, shallow=False):
            print(f"FAILED: {graph_id} images differ")
            return
        else:
            print(f"PASSED: {graph_id}")
            
    print("All passed!")

if __name__ == '__main__':
    test()
