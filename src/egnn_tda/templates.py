import plotly.graph_objects as go
import plotly.io as pio

pio.templates["template_TNR"] = go.layout.Template(

    layout={
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
            showticklabels=True,
            mirror=True,
            ticks='inside',
            ticklen = 10,
            tickwidth = 2,
            showline=True,
            linecolor='black',
            gridcolor='lightgrey',
            exponentformat='E'
        ),
    }
)
