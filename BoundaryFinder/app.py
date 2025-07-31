# Import required libraries
import streamlit as st
import osmnx as ox
import folium
import spacy
import geopandas as gpd
from streamlit_folium import st_folium
import json
import pandas as pd
from shapely.geometry import Point
import overpy

st.set_page_config(page_title="Boundary Finder", page_icon="🌍", layout="centered")

# Load NLP model
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

# Initialize session state
if "location" not in st.session_state:
    st.session_state["location"] = None
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "geojson_data" not in st.session_state:
    st.session_state["geojson_data"] = None
if "markers" not in st.session_state:
    st.session_state["markers"] = []  
if "add_marker" not in st.session_state:
    st.session_state["add_marker"] = False  
if "add_buffer" not in st.session_state:
    st.session_state["add_buffer"] = False  
if "buffers" not in st.session_state:
    st.session_state["buffers"] = []  

# Extract city/country name
def extract_location(prompt):
    doc = nlp(prompt)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
    return None

# Process location
def process_location():
    location = extract_location(st.session_state["input_text"])
    if location:
        st.session_state["location"] = location
        try:
            gdf = ox.geocode_to_gdf(location)
            st.session_state["geojson_data"] = json.loads(gdf.to_json())
        except Exception as e:
            st.error(f"Error fetching boundary: {e}")
            st.session_state["geojson_data"] = None
    else:
        location = st.session_state["input_text"].strip()
        if location:
            try:
                gdf = ox.geocode_to_gdf(location)
                if gdf.empty:
                    st.warning(f"Location '{location}' not found.")
                    st.session_state["geojson_data"] = None
                else:
                    st.session_state["location"] = location
                    st.session_state["geojson_data"] = json.loads(gdf.to_json())
            except Exception as e:
                st.warning(f"Location '{location}' error: {e}")
                st.session_state["geojson_data"] = None
        else:
            st.session_state["location"] = None
            st.session_state["geojson_data"] = None

# Function to fetch POIs within the buffer using Overpass API
def fetch_pois_within_buffer(buffers):
    pois = []
    api = overpy.Overpass()  # Instantiate Overpass without the timeout argument
    api.timeout = 180  # Set the timeout attribute to 180 seconds

    for lat, lon, radius in buffers:
        # Overpass query to fetch POIs within the radius of a given point
        overpass_query = f"""
        [out:json];
        (
          node["amenity"](around:{radius},{lat},{lon});
          way["amenity"](around:{radius},{lat},{lon});
          relation["amenity"](around:{radius},{lat},{lon});
        );
        out body;
        """
        
        try:
            # Execute the query
            result = api.query(overpass_query)

            if not result.nodes and not result.ways and not result.relations:
                st.warning(f"No POIs found within {radius} meters of ({lat}, {lon}).")
            
            for node in result.nodes:
                pois.append({
                    "name": node.tags.get("name", "Unnamed POI"),
                    "type": node.tags.get("amenity", "Unknown"),
                    "latitude": node.lat,
                    "longitude": node.lon
                })
            for way in result.ways:
                pois.append({
                    "name": way.tags.get("name", "Unnamed POI"),
                    "type": way.tags.get("amenity", "Unknown"),
                    "latitude": way.nodes[0].lat,
                    "longitude": way.nodes[0].lon
                })
            for relation in result.relations:
                pois.append({
                    "name": relation.tags.get("name", "Unnamed POI"),
                    "type": relation.tags.get("amenity", "Unknown"),
                    "latitude": relation.nodes[0].lat,
                    "longitude": relation.nodes[0].lon
                })
        except Exception as e:
            st.error(f"Error fetching POIs: {e}")

    return pois

# Generate map
def get_boundary_map(location_name=None, geojson_data=None, markers=None, buffers=None):
    try:
        m = folium.Map(location=[50, 10], zoom_start=4)

        # Draw boundary
        if location_name and geojson_data:
            folium.GeoJson(
                data=geojson_data, 
                name=f"{location_name} Boundary",
                style_function=lambda x: {
                    'fillColor': '#3186cc',
                    'color': '#000',
                    'weight': 2,
                    'fillOpacity': 0.25
                }
            ).add_to(m)

        # Add markers
        if markers:
            for lat, lon in markers:
                folium.Marker(location=[lat, lon], popup=f"Marker: {lat}, {lon}").add_to(m)

        # Add buffers
        if buffers:
            for lat, lon, radius in buffers:
                folium.Circle(
                    location=[lat, lon],
                    radius=radius,  
                    color="red",
                    fill=True,
                    fill_opacity=0.3
                ).add_to(m)

        return m
    except Exception as e:
        st.error(f"Error creating map: {e}")
        return None



st.title("🌍 Boundary Finder")
st.write("Type a location and see its boundary. Add markers and buffers!")

# User input
st.text_input("Enter a location:", key="input_text", on_change=process_location)

# Buttons
col1, col2, col3, col4 = st.columns(4)

# Show on Map button
with col1:
    if st.button("Show on Map", use_container_width=True):
        process_location()

# Download GeoJSON button (always visible but only provides file if data exists)
with col2:
    geojson_str = json.dumps(st.session_state["geojson_data"], indent=2) if st.session_state["geojson_data"] else None
    st.download_button(
        label="Download GeoJSON",
        data=geojson_str if geojson_str else "",
        file_name=f"{st.session_state['location']}_boundary.geojson" if st.session_state["location"] else "boundary.geojson",
        mime="application/json",
        disabled=geojson_str is None,
        use_container_width=True
    )

# Add Marker button
with col3:
    if st.button("Add Marker", use_container_width=True):
        st.session_state["add_marker"] = not st.session_state["add_marker"]

# Add Buffer button
with col4:
    if st.button("Add Buffer", use_container_width=True):
        st.session_state["add_buffer"] = not st.session_state["add_buffer"]

# Display the map
map_obj = get_boundary_map(
    st.session_state.get("location"),
    st.session_state.get("geojson_data"),
    st.session_state.get("markers"),
    st.session_state.get("buffers")
)

map_data = st_folium(map_obj, width=700, height=400)

# Handle clicks for adding markers
if st.session_state["add_marker"] and map_data["last_clicked"]:
    lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    if (lat, lon) not in st.session_state["markers"]:
        st.session_state["markers"].append((lat, lon))

# Handle buffer addition
if st.session_state["add_buffer"]:
    for lat, lon in st.session_state["markers"]:
        if (lat, lon, 500) not in st.session_state["buffers"]:
            st.session_state["buffers"].append((lat, lon, 500))  

# Fetch POIs within buffer and display in a table
if st.session_state["buffers"]:
    pois = fetch_pois_within_buffer(st.session_state["buffers"])
    if pois:
        pois_df = pd.DataFrame(pois)
        st.write("Points of Interest within the buffer area:")
        st.dataframe(pois_df)
    else:
        st.write("No points of interest found within the buffer areas.")
