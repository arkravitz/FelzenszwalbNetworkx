from scipy import ndimage
import numpy as np
import networkx as nx
from scipy import misc
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.filters import gaussian_filter
from skimage import data
from skimage import io

im = rescale(io.imread('data/blackhole.jpg',as_grey=True), .2)
im_noblur = rescale(io.imread('data/blackhole.jpg',as_grey=True), .2)
im = gaussian_filter(im, 4)
#im = rgb2gray(im)
print im.shape

k = 80.0 # Constant for the tau function

G = nx.grid_graph(dim=list(im.shape[:2]))

segment_graph = nx.Graph()
segment_graph.add_nodes_from(G.nodes())

for u,v,d in G.edges(data=True):
    d['weight'] = np.abs(im[u] - im[v])*255

sorted_edges = sorted(G.edges(data=True), key=lambda (u,v,d): d['weight'])

'''
Must pass in the connected nodes of the vertex
'''
def max_diff(nodes):
    global segment_graph
    # Degenerate case
    if len(nodes) == 1:
        return 0.0

    edge = max(segment_graph.edges(nodes,data=True), key=lambda (u,v,d): d['weight'])
    return edge[2]['weight']

def min_internal_diff(nodes1, nodes2):
    global k
    tauC1 = k/len(nodes1)
    tauC2 = k/len(nodes2)

    return min(max_diff(nodes1) + tauC1, max_diff(nodes2) + tauC2)

print "starting algo"
for u,v,d in sorted_edges:
    connected1 = nx.node_connected_component(segment_graph, u)
    connected2 = nx.node_connected_component(segment_graph, v)
    if d['weight'] <= min_internal_diff(connected1, connected2):
        segment_graph.add_edge(u,v,weight=d['weight'])

group = 1
group_img = np.zeros(im.shape)
print "Drawing image"
for subgraph in nx.connected_component_subgraphs(segment_graph):
    for node in subgraph.nodes():
        group_img[node] = group
    group += 1
print "matplotlibbin"

plt.imshow(im_noblur, cmap=plt.cm.gray)
plt.matshow(group_img)
plt.show()
