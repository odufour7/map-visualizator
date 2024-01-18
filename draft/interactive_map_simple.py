import json
import numpy as np
import plotly.graph_objects as go
import pandas as pd

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import webbrowser

df = pd.DataFrame(np.array([["Place Saint-Jean",2], ["Place des Terreaux",1]]),columns=['place', 'value'])
data = json.loads('{"type": "FeatureCollection","features": [{"type": "Feature", "properties": {"NAME": "Place Saint-Jean"},'+ 
'"geometry": {"type": "Polygon", "coordinates": [[[4.826871 , 45.760673],[4.827051 , 45.761122],[4.825956 , 45.761350],[4.825772 , 45.760983]]]}},'+
'{"type": "Feature", "properties": {"NAME": "Place des Terreaux"},'+
'"geometry": {"type": "Polygon", "coordinates": [[[4.834173, 45.767407], [4.834065, 45.767756],  [4.83280, 45.767658], [4.832846, 45.767146]]]}}]}')
### to modify the coordinates : https://epsg.io/map#srs=4326&x=4.825772&y=45.760983&z=19&layer=streets , pay attention the coordinates system used is EPSG:4326 
### the polygon starts on the bottom right and you have to go counter-clockwise
### first coordinate is x-axis (longitude), second is y-axis (lattitude)
### to modify the plot : https://plotly.com/python-api-reference/generated/plotly.graph_objects.Choroplethmapbox.html 


fig = go.Figure(go.Choroplethmapbox(geojson=data, locations=df.place, z=df.value,
                                    colorscale='bluered',featureidkey='properties.NAME',
                                    marker_opacity=0.5, showscale=False,
                                    hoverlabel_namelength=-1,customdata=['<a href="https://google.com"> Place Saint-Jean  </a>',
                                                                         '<a href="https://google.com"> Place des Terreaux</a>']))
fig.update_layout(mapbox=dict(style="carto-positron", zoom=14, center = {"lon": 4.8337,"lat": 45.763420}),
                  margin={"l":300,"r":300,"t":0,"b":130})
fig.update_traces(hovertemplate=['<b> <a href="https://google.com"> Place Saint-Jean  </a></b> <extra></extra>',
                                 '<b> <a href="https://google.com"> Place des Terreaux</a></b> <extra></extra>'], 
                  hoverlabel=dict(font_size=20,bgcolor="white"))

fig.write_html("/Users/oscar/Desktop/interactive_map.html")
fig.show()
