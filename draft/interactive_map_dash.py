import dash
from dash import html
from dash.dependencies import Input, Output
import folium
import webbrowser

# Create a Dash app
app = dash.Dash(__name__)

# Create a Folium map centered on Lyon
m = folium.Map(location=[45.763420,  4.8337], zoom_start=16)

# Add polygons for Place Saint Jean and Place des Terreaux
folium.Polygon(locations=[
                [45.760673, 4.826871],
                [45.761122, 4.827051],
                [45.761350, 4.825956],
                [45.760983, 4.825772],
                [45.760673, 4.826871]], color='blue', fill=True, fill_color='blue', fill_opacity=0.2, popup='Place Saint Jean').add_to(m)
folium.Polygon(locations=[
                [45.767407, 4.834173],
                [45.767756, 4.834065],
                [45.767658, 4.83280],
                [45.767146, 4.832846],
                [45.767407, 4.834173]], color='red', fill=True, fill_color='red', fill_opacity=0.2, popup='Place des Terreaux').add_to(m)


# Convert the Folium map to HTML
m.save('map.html')

# Define the layout of the Dash app
app.layout = html.Div([
    html.Iframe(id='map', srcDoc=open('map.html', 'r').read(), width='100%', height='600'),
    html.Div(id='video-popup')
])

# Define a callback to handle the click event
@app.callback(
    Output('video-popup', 'children'),
    [Input('map', 'n_clicks')],
    prevent_initial_call=True
)
def open_video(n_clicks):
    if n_clicks:
        # Open the YouTube video in a new tab
        webbrowser.open_new_tab('https://www.youtube.com/watch?v=Ch7VxxTBe1c')
        return None

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)