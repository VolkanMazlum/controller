import plotly.graph_objects as go
import plotly.express as ex
import numpy as np




def plot_activity_pos_neg(freq_pos, freq_neg, mean_pos, mean_neg, time_vect,label, show=False, to_html = False, to_png = True, path=''):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_vect, y=freq_pos, name="positive",mode="lines", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=time_vect, y=freq_neg, name="negative",mode="lines", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=time_vect, y=mean_pos, name="mean pos",line=dict(color="red", dash="dash")))
    fig.add_trace(go.Scatter(x=time_vect, y=mean_neg, name="mean neg",line=dict(color="blue",dash="dash")))
    fig.update_layout(width=1200,
                    height=800,
                title={
                    'text': label,
                    'font' : {"size":36},
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#fcfcfc',
                font=dict(
                    size=24,
                    color="Black"
            )
        )

    if to_png:
        fig.update_xaxes(title = "Time [ms]",
                    linecolor="black",
                    gridcolor = "#bfbfbf",
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    title_standoff = 0
                    )
        fig.update_yaxes(title = "Frequency [Hz]",
                        linecolor="black", 
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        gridcolor="#bfbfbf"
                        )

        fig.write_image(path+label+".png")

    else:
        fig.update_xaxes(title = "Time [ms]",
                        linecolor="black",
                        gridcolor = "#bfbfbf",
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        title_standoff = 0,
                        rangeslider_visible=True)
        fig.update_yaxes(title = "Frequency [Hz]",
                        linecolor="black", 
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        gridcolor="#bfbfbf"
                        )
    
    
    if to_html:
        fig.write_html(path+label+".html", include_plotlyjs='cdn')
    if show:
        fig.show()

def plot_activity(freq, mean, time_vect, label, show=False, to_html = False, to_png = True, path=''):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_vect, y=freq, name = "frequency",mode="lines", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=time_vect, y=mean, name = "mean",line=dict(dash="dash",color="red")))
    fig.update_layout(width=1200,
                    height=800,
                title={
                    'text': label,
                    'font' : {"size":36},
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#fcfcfc',
                font=dict(
                    size=24,
                    color="Black"
            )
        )

    if to_png:
        fig.update_xaxes(title = "Time [ms]",
                    linecolor="black",
                    gridcolor = "#bfbfbf",
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    title_standoff = 0
                    )
        fig.update_yaxes(title = "Frequency [Hz]",
                        linecolor="black", 
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        gridcolor="#bfbfbf"
                        )

        fig.write_image(path+label+".png")

    if to_html or show:
        print("in show")
        fig.update_xaxes(title = "Time [ms]",
                        linecolor="black",
                        gridcolor = "#bfbfbf",
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        title_standoff = 0,
                        rangeslider_visible=True)
        fig.update_yaxes(title = "Frequency [Hz]",
                        linecolor="black", 
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        gridcolor="#bfbfbf"
                        )
    
    if to_html:
        fig.write_html(path+label+".html", include_plotlyjs='cdn')
    if show:
        fig.show()
    return fig

