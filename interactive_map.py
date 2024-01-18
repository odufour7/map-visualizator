""" Map visualisator for Madras project.
    Run : streamlit run ./interactive_map_oscar_mohcine.py """

import json
from dataclasses import dataclass
import folium
import streamlit as st
from streamlit_folium import st_folium
from folium.features import CustomIcon
from typing import Dict, Tuple


@dataclass
class Camera:
    """Data class representing a camera with its location and video URL."""
    location: tuple
    url: str
    name: str
    field : list
    logo : str

tile_layers = {
    "Open Street Map": "openstreetmap",
    "CartoDB Positron": "CartoDB positron",
    "CartoDB Dark_Matter": "CartoDB dark_matter",
}

def load_cameras_from_json(file_path: str):
    """
    Load camera data from a JSON file and return a dict. of Camera objects.

    Args:
    file_path (str): The path to the JSON file.

    Returns:
    Dict[str, Camera]: A dictionary mapping camera names to Camera objects.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from file: {file_path}")
        return {}

    cameras = {}
    for key, info in data.items():
        try:
            # Ensure the data structure is as expected
            location = tuple(info["location"])
            url = info["url"]
            name = info["name"]
            field = info["field"]
            logo = info["logo"]
            cameras[key] = Camera(location=location, url=url, name=name, field=field, logo=logo)
        except KeyError as e:
            # Handle missing keys in the data
            st.error(f"Missing key in camera data: {e}")
            continue  # Skip this camera and continue with the next
        except Exception as e:
            # Catch any other unexpected errors
            st.error(f"Error processing camera data: {e}")
            continue

    return cameras



def create_map(center: Tuple[float, float], tile_layer, zoom: int = 16) -> folium.Map:
    """Create a folium map with camera markers and polygon layers.

    Args:
    center (Tuple[float, float]): The center of the map (latitude, longitude).
    zoom_start (int): The initial zoom level of the map.

    Returns:
    folium.Map: A folium map object.
    """
    m = folium.Map(location=center, zoom_start=zoom, tiles=tile_layer,  max_zoom = 21)
    camera_layers = [folium.FeatureGroup(name=key, show=True).add_to(m) for key in cameras.keys()]  # Create a layer/group for each camera
    # polygon_layers = [folium.FeatureGroup(name="field "+key, show=True, overlay=True).add_to(m) for key in cameras.keys() ] # Create a layer/group for each polygon
    
    # polygons = [
    #     folium.PolyLine(
    #         locations=camera.field,
    #         tooltip="field " +key,
    #         fill_color="blue",
    #         color=None,
    #         fill_opacity=0.1,
    #         fill=True,
    #     )
    #     for key, camera in cameras.items()]
            
    # for polygon, layer in zip(polygons, polygon_layers): # Add polygons to the map
    #     polygon.add_to(layer)

    for (key, camera), layer in zip(cameras.items(), camera_layers): # Add markers to the map
        icon = CustomIcon(camera.logo, icon_size=(110, 110) )

        coords = camera.location
        tooltip = f"{key}: {camera.name}"
        folium.Marker(location=coords, tooltip=tooltip, icon=icon).add_to(layer)

    folium.LayerControl().add_to(m) # Add layer control to the map (permet de choisir les layers à afficher)
    return m


def setup() -> None:
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Madras Project",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("Interactive Map with Multiple Layers")
    st.markdown(
        """
    **Layer Selection:**
    Use the layer control button in the top right corner of the map to toggle different layers. 
    You can select video overlays, camera markers, and other features from this control panel.
    """,
        unsafe_allow_html=True,
    )


def main(cameras: Dict[str, Camera], selected_layer) -> None:
    """Implement the main logic of the app.

    Args:
    cameras (Dict[str, Camera]): A dictionary of Camera objects.
    """
    center = [45.76322690683106, 4.83001470565796]  # Coordinates for Lyon, France
    m = create_map(center, tile_layer=tile_layers[selected_layer])
    map_data = st_folium(m, width=950, height=800) # width and height must be less than 990 to be able to see the layer control  
    placeholder = st.sidebar.empty()
    video_name = map_data.get("last_object_clicked_tooltip")
    if video_name:
        placeholder.info(f"Selected Camera: {video_name}")
        camera = cameras.get(video_name.split(":")[0])
        if camera:
            st.sidebar.video(camera.url)
        else:
            st.sidebar.error(f"No video linked to {video_name}.")
    else:
        st.sidebar.error("No camera selected.")


if __name__ == "__main__":
    setup()
    cameras = load_cameras_from_json("cameras.json")
    selected_layer = st.selectbox("Choose a Map Style:", list(tile_layers.keys()))
    main(cameras, selected_layer)