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

IMAGE_PATH = 'data/puppy1.jpeg'
k = 70.0 # Constant for the tau function
RESCALE_FACTOR = 0.1
GAUSSIAN_BLUR = 4


def get_graph_from_image(image): 
  grid = nx.grid_graph(dim=list(image.shape[:2])) 

  for u,v,d in grid.edges(data=True):
    d['weight'] = np.abs(image[u] - image[v])*255
 
  return grid 

'''
Must pass in the connected nodes of the vertex
'''
def max_diff(nodes, segment_graph):
    # Degenerate case
    if len(nodes) == 1:
        return 0.0

    edge = max(segment_graph.edges(nodes,data=True), key=lambda (u,v,d): d['weight'])
    return edge[2]['weight']


def min_internal_diff(nodes1, nodes2, graph):
    global k
    tauC1 = k/len(nodes1)
    tauC2 = k/len(nodes2)

    return min(max_diff(nodes1, graph) + tauC1, max_diff(nodes2, graph) + tauC2)


def segment(segment_graph, sorted_edges):
  for u,v,d in sorted_edges:
    connected1 = nx.node_connected_component(segment_graph, u)
    connected2 = nx.node_connected_component(segment_graph, v)
    if d['weight'] <= min_internal_diff(connected1, connected2, segment_graph):
      segment_graph.add_edge(u,v,weight=d['weight'])
  return segment_graph


def visualize(image_no_blur, segment_graph):
  group = 1
  group_img = np.zeros(image_no_blur.shape)

  for subgraph in nx.connected_component_subgraphs(segment_graph):
    for node in subgraph.nodes():
      group_img[node] = group
    group += 1

  plt.imshow(image_no_blur, cmap=plt.cm.gray)
  plt.matshow(group_img)
  plt.show()


def main():

  print 'loading image and preprocessing'
  # image preprocessing 
  image = rescale(io.imread(IMAGE_PATH, as_grey=True), RESCALE_FACTOR)
  image_no_blur = image
  image = gaussian_filter(image, GAUSSIAN_BLUR)
  print 'image of size ' + str(image.shape) + ' after resizing'

  # k sanity check 
  print 'k = %.2f - should be approx %.2f' % (k, np.mean(image.shape))

  # make graphs from image
  print 'constructing graph'
  grid_graph = get_graph_from_image(image)
  segment_graph = nx.Graph()
  segment_graph.add_nodes_from(grid_graph.nodes())
  sorted_edges = sorted(grid_graph.edges(data=True), key=lambda (u,v,d): d['weight'])

  # run algo 
  print 'segmenting image graph'
  segment_graph = segment(segment_graph, sorted_edges)
  
  # visualize
  print 'visualizing'
  visualize(image_no_blur, segment_graph)


if __name__ == '__main__':
  main()
