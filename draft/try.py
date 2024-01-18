import json
from dataclasses import dataclass
import folium
import streamlit as st
from streamlit_folium import st_folium


@dataclass
class Camera:
    location: tuple
    url: str
    field : list


def load_cameras_from_json(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
        cameras = {}
        for name, info in data.items():
            cameras[name] = Camera(location=tuple(info["location"]), url=info["url"], field=info["field"])
        return cameras


def create_map(center, zoom_start=16):
    m = folium.Map(location=center, zoom_start=zoom_start,  max_zoom = 21)
    camera_layers = [folium.FeatureGroup(name=name, show=True).add_to(m) for name in cameras.keys()]  # Create a layer/group for each camera

    polygon_layers = [folium.FeatureGroup(name="field "+name, show=True, overlay=True).add_to(m) for name in cameras.keys() ] # Create a layer/group for each polygon

    polygons = [
        folium.PolyLine(
            locations=camera.field,
            tooltip="field " +name,
            fill_color="blue",
            color="red",
            fill_opacity=0.2,
            fill=True,
        )
        for name, camera in cameras.items()]
            

    for polygon, layer in zip(polygons, polygon_layers): # Add polygons to the map
        polygon.add_to(layer)

    for (name, camera), layer in zip(cameras.items(), camera_layers): # Add markers to the map
        coords = camera.location
        tooltip = name
        folium.Marker(location=coords, tooltip=tooltip).add_to(layer)

    folium.LayerControl().add_to(m) # Add layer control to the map (permet de choisir les layers à afficher)
    return m


def setup():
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


def main(cameras):
    center = [45.76322690683106, 4.83001470565796]  # Coordinates for Lyon, France
    m = create_map(center)
    map_data = st_folium(m, width=800, height=800)
    placeholder = st.sidebar.empty()
    video_name = map_data.get("last_object_clicked_tooltip")
    if video_name:
        placeholder.info(f"Selected Camera: {video_name}")
        camera = cameras.get(video_name)
        if camera:
            st.sidebar.video(camera.url)
        else:
            st.sidebar.error(f"No video linked to {video_name}.")
    else:
        st.sidebar.error("No camera selected.")


if __name__ == "__main__":
    setup()
    cameras = load_cameras_from_json("cameras.json")
    main(cameras)