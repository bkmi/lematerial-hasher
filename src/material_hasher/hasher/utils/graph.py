from structuregraph_helpers.hash import generate_hash
from networkx import Graph

def get_weisfeiler_lehman_hash(graph: Graph,):
    return generate_hash(graph, True, False, 100)