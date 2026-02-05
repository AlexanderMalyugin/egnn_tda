import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_pi(PI,
                 x_min = 0.0,
                 x_max = 4.0,
                 y_min = 0.0,
                 y_max = 4.0,
                 name : str = "fig.png"
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
                #colorbar=dict(title="value", len=0.85)  # each subplot gets its own bar
            ),
            row=1, col=i+1
        )

    fig.update_layout(
        width=350*ncols,
        height=380,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    fig.show()
    fig.write_image(name)