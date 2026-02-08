import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import gudhi as gd
from scipy.spatial.distance import hamming
from io import BytesIO

import plotly.graph_objs as go
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio

def visualize_simplicial_complex(simplex_tree, filtration_value, vertex_names=None, save_filename=None, plot_size=1, dpi=600, pos=None):
    G = nx.Graph()
    triangles = []  # List to store triangles (3-nodes simplices)

    for simplex, filt in simplex_tree.get_filtration():
        if filt <= filtration_value:
            if len(simplex) == 2:
                G.add_edge(simplex[0], simplex[1])
            elif len(simplex) == 1:
                G.add_node(simplex[0])
            elif len(simplex) == 3:
                triangles.append(simplex)

    # Calculate node positions if not provided
    if pos is None:
        pos = nx.spring_layout(G)

    # Node trace
    x_values, y_values = zip(*[pos[node] for node in G.nodes()])
    node_labels = [vertex_names[node] if vertex_names else str(node) for node in G.nodes()]
    node_trace = go.Scatter(x=x_values, y=y_values, mode='markers+text', hoverinfo='text', marker=dict(size=14), text=node_labels, textposition='top center', textfont=dict(size=14))

    # Edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace = go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(width=3, color='rgba(0,0,0,0.5)'))
        edge_traces.append(edge_trace)

    # Triangle traces
    triangle_traces = []
    for triangle in triangles:
        x0, y0 = pos[triangle[0]]
        x1, y1 = pos[triangle[1]]
        x2, y2 = pos[triangle[2]]
        triangle_trace = go.Scatter(x=[x0, x1, x2, x0, None], y=[y0, y1, y2, y0, None], fill='toself', mode='lines+markers', line=dict(width=2), fillcolor='rgba(255,0,0,0.2)')
        triangle_traces.append(triangle_trace)

    # Configure the layout of the plot
    layout = go.Layout(showlegend=False, hovermode='closest', xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=16, family='Arial, sans-serif')), yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=16, family='Arial, sans-serif')))

    fig = go.Figure(data=edge_traces + triangle_traces + [node_trace], layout=layout)

    # Set the figure size
    fig.update_layout(width=plot_size * dpi, height=plot_size * dpi)

    # Save the figure if a filename is provided
    if save_filename:
        pio.write_image(fig, save_filename, width=plot_size * dpi, height=plot_size * dpi, scale=1)

    # Show the figure
    fig.show()

    return G

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_pi(PI,
                 x_min = 0.0,
                 x_max = 4.0,
                 y_min = 0.0,
                 y_max = 4.0
                 ):

    ncols =  PI.shape[0]
    bins = PI.shape[1]

    dx = (x_max - x_min) / bins
    dy = (y_max - y_min) / bins

    fig = make_subplots(
        rows=1, cols=ncols,
        subplot_titles=[f"PI (H{i})" for i in range(ncols)],
        horizontal_spacing=0.06
    )

    for i in range(ncols):
        fig.add_trace(
            go.Heatmap(
                z=PI[i],
                x0=x_min, dx=dx,
                y0=y_min, dy=dy,
                colorscale="Viridis",
                colorbar=dict(title="value", len=0.85)  # each subplot gets its own bar
            ),
            row=1, col=i+1
        )

    fig.update_layout(
        width=350*ncols,
        height=380,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    fig.show()