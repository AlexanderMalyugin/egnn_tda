import plotly.graph_objects as go
import plotly.io as pio

import numpy as np

pio.templates["template_TNR"] = go.layout.Template(
    # layout_annotations=[
    #     dict(
    #         name="draft watermark",
    #         text="DRAFT",
    #         textangle=-30,
    #         opacity=0.1,
    #         font=dict(color="black", size=100),
    #         xref="paper",
    #         yref="paper",
    #         x=0.5,
    #         y=0.5,
    #         showarrow=False,
    #     )
    # ],
    layout={
        #'plot_bgcolor': 'white',
        'plot_bgcolor': 'rgba(0,0,0,0)',  
        'paper_bgcolor' : 'rgba(0,0,0,0)',
        'legend': dict(
            x=0.95, y=0.6, yanchor='top', xanchor="right", orientation='v'
        ),
        'font': dict(
            size=20, family='Times New Roman', color='black'
        ),
        'yaxis': dict(
            linewidth = 2,
            title_font=dict(size=20),
            tickfont=dict(size=20),
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='black',
            #type='log',
            showticklabels=True,
            mirror=True,
            ticks='inside',
            ticklen = 10,
            tickwidth = 2,
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
            #nticks=6,
            exponentformat='E'
        ),
        'xaxis': dict(
            linewidth = 2,
            title_font=dict(size=20),
            tickfont=dict(size=20),
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='lightgrey',
            #type='log',
            showticklabels=True,
            mirror=True,
            ticks='inside',
            ticklen = 10,
            tickwidth = 2,
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
            #nticks=4,
            exponentformat='E'
        ),
    }
)
