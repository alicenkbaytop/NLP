import streamlit as st
import ollama
import requests
import folium
import re
import pandas as pd
from shapely.geometry import Point
from streamlit_folium import st_folium
import json
import osmnx as ox
import pprint
from geopy.geocoders import Nominatim

# OpenStreetMap Overpass API URL
OVERPASS_URL = "http://overpass-api.de/api/interpreter"


# Function to query OpenStreetMap
def query_osm(lat, lon, radius, category):
    """Query OpenStreetMap Overpass API for POIs based on category"""

    category_tags = {
        "restaurant": '["amenity"="restaurant"]',
        "cafe": '["amenity"="cafe"]',
        "hospital": '["amenity"="hospital"]',
        "pharmacy": '["amenity"="pharmacy"]',
        "bank": '["amenity"="bank"]',
        "supermarket": '["shop"="supermarket"]',
        "school": '["amenity"="school"]',
        "park": '["leisure"="park"]',
        "bus_station": '["public_transport"="station"]',
    }

    tag = category_tags.get(category, '["amenity"]')

    query = f"""
    [out:json];
    (
      node{tag}(around:{radius},{lat},{lon});
      way{tag}(around:{radius},{lat},{lon});
      relation{tag}(around:{radius},{lat},{lon});
    );
    out center;
    """

    response = requests.get(OVERPASS_URL, params={"data": query})
    return response.json().get("elements", []) if response.status_code == 200 else []


def llm_process(user_query):
    """Uses Ollama LLM to extract location, buffer, and time-based info from user input, and adds geocoding via OSMnx"""

    system_prompt = (
    "Extract geospatial details from this query. "
    "Return a valid JSON object with keys: 'latitude' (float, optional), 'longitude' (float, optional), "
    "'location' (str, optional), 'buffer_radius' (int, optional), 'category' (str, optional), 'open_hours' (str, optional). "
    "Ensure the 'location' field is suitable for geocoding with OpenStreetMap. "
    "If the query includes a local building name (e.g., 'İBB Kasımpaşa Ek Hizmet Binası'), expand it to a full address or include the city (e.g., 'İstanbul') to improve geocoding accuracy. "
    "Avoid returning overly specific or informal building names if a more general area or neighborhood is available. "
    "For example, instead of 'İBB Kasımpaşa Ek Hizmet Binası', return 'Kasımpaşa, İstanbul'. "
    "Ensure 'category' matches one of the following OpenStreetMap categories: "
    "'restaurant', 'cafe', 'hospital', 'pharmacy', 'bank', 'supermarket', 'school', 'park', 'bus_station'. "
    "Respond only with a valid JSON object."
    )

    response = ollama.chat(
        model="qwen3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    )

    raw_text = response["message"]["content"].strip()

    # Try to extract valid JSON block from LLM output using regex
    match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            extracted_data = json.loads(json_str)
        except json.JSONDecodeError:
            print("\n⚠️ JSON parsing failed\n")
            extracted_data = {}
    else:
        print("\n⚠️ No JSON block found in LLM output\n")
        extracted_data = {}

    # If coordinates are missing but location is present, use OSMnx to geocode
    

    # Inside llm_process(), replace the geocoding block:
    if "latitude" not in extracted_data or "longitude" not in extracted_data:
        location_name = extracted_data.get("location")
        if location_name:
            try:
                # Try with OSMnx
                gdf = ox.geocode_to_gdf(location_name)
                lat, lon = gdf.geometry.y.values[0], gdf.geometry.x.values[0]
                extracted_data["latitude"] = lat
                extracted_data["longitude"] = lon
            except Exception as e:
                print(f"\n⚠️ OSMnx Geocoding Failed: {e}\n")
                # Fallback: Try geopy
                try:
                    geolocator = Nominatim(user_agent="chatgis")
                    location = geolocator.geocode(location_name)
                    if location:
                        extracted_data["latitude"] = location.latitude
                        extracted_data["longitude"] = location.longitude
                        print(f"\n✅ Fallback geopy result: {location.latitude}, {location.longitude}\n")
                    else:
                        print(f"\n❌ Fallback geopy could not find: {location_name}\n")
                except Exception as geo_e:
                    print(f"\n❌ Geopy Fallback Error: {geo_e}\n")

    # Fallback: If category is missing or null, manually extract it from user_query
    if extracted_data.get("category") is None:
        category_mapping = {
            "restaurant": "restaurant",
            "cafe": "cafe",
            "hospital": "hospital",
            "pharmacy": "pharmacy",
            "bank": "bank",
            "supermarket": "supermarket",
            "school": "school",
            "park": "park",
            "bus station": "bus_station",
        }

        for key, value in category_mapping.items():
            if key in user_query.lower():
                extracted_data["category"] = value
                break

    print("\n✅ Final Extracted Data Object:")
    pprint.pprint(extracted_data)
    print("-" * 50 + "\n")

    return extracted_data


# Streamlit Page Layout
st.set_page_config(page_title="ChatGIS", page_icon="🌍", layout="wide")

# === Left Sidebar ===
with st.sidebar:
    st.title("🌍 ChatGIS")
    st.markdown("**Geospatial AI Chatbot & Mapping Tool**")
    interaction_type = st.radio("Choose Mode:", ["Chat Assistant", "Map Click Search"])
    st.markdown("---")
    st.subheader("ℹ️ About ChatGIS")
    st.write(
        "ChatGIS allows users to find Points of Interest (POIs) on a map using natural language "
        "or by manually selecting a location. Powered by OpenStreetMap and an AI model."
    )

# === POI Categories ===
poi_categories = {
    "All": "",
    "Restaurants": "restaurant",
    "Cafes": "cafe",
    "Hospitals": "hospital",
    "Parks": "park",
    "Pharmacies": "pharmacy",
    "Banks": "bank",
    "Supermarkets": "supermarket",
    "Schools": "school",
    "Bus Stops": "bus_station",
}

# === Right Side: Main Content ===
col1, col2 = st.columns([2, 1])

# Default location (Istanbul)
default_location = [41.0082, 28.9784]

# === Map Section (Top) ===
with col1:
    st.subheader("🗺️ Interactive Map")

    if interaction_type == "Map Click Search":
        map_center = folium.Map(location=default_location, zoom_start=12)

        if "selected_location" not in st.session_state:
            st.session_state.selected_location = None

        map_data = st_folium(
            map_center, height=400, width=800, returned_objects=["last_clicked"]
        )
    else:
        st.info("🗺️ Map is only shown in **Manual Selection** mode.")

# === Controls & Output Section (Bottom) ===
with col2:
    if interaction_type == "Map Click Search":
        st.subheader("📍 Select a Location")
        buffer_radius = st.slider("Buffer Radius (meters)", 100, 5000, 1000)
        selected_category = st.selectbox(
            "Select POI Category", list(poi_categories.keys())
        )

        if 'map_data' in locals() and map_data and map_data.get("last_clicked"):
            lat, lon = map_data["last_clicked"].get("lat"), map_data["last_clicked"].get("lng")
            st.session_state.selected_location = (lat, lon)

        if st.session_state.selected_location:
            lat, lon = st.session_state.selected_location
            st.write(f"📍 Selected Location: {lat}, {lon}")
            pois = query_osm(lat, lon, buffer_radius, poi_categories[selected_category])

            poi_list = [
                {
                    "Name": poi.get("tags", {}).get("name", "Unknown"),
                    "Latitude": poi.get("lat"),
                    "Longitude": poi.get("lon"),
                }
                for poi in pois
                if poi.get("lat") and poi.get("lon")
            ]

            if poi_list:
                st.subheader(f"📌 Found {len(poi_list)} POIs in '{selected_category}'")
                st.dataframe(pd.DataFrame(poi_list))
            else:
                st.write(
                    f"❌ No POIs found within {buffer_radius}m of ({lat}, {lon}). Try increasing the buffer distance."
                )

    elif interaction_type == "Chat Assistant":
        st.subheader("💬 Chat with AI")

        user_query = st.text_area(
            "Ask a geospatial question (e.g., 'Find restaurants within 5000 meters of İBB Kasımpaşa Ek Hizmet Binası')"
        )

        if "last_processed_query" not in st.session_state:
            st.session_state.last_processed_query = ""
        if "extracted_data" not in st.session_state:
            st.session_state.extracted_data = None

        query_submitted = st.button("Send Query")

        if query_submitted and user_query.strip():
            if user_query != st.session_state.last_processed_query:
                st.session_state.last_processed_query = user_query
                print("\n📝 User Query Received:")
                print(f"Query Text: {user_query}")
                print(f"Query Submitted: {query_submitted}\n")

                extracted_data = llm_process(user_query)
                st.session_state.extracted_data = extracted_data
            else:
                extracted_data = st.session_state.extracted_data
        else:
            extracted_data = st.session_state.extracted_data

        if extracted_data:
            category = extracted_data.get("category")
            lat = extracted_data.get("latitude")
            lon = extracted_data.get("longitude")
            buffer_radius = extracted_data.get("buffer_radius", 1000)
            open_hours = extracted_data.get("open_hours", "Unknown")

            if not category:
                category = ""

            search_buffers = [buffer_radius, 2000]
            found_pois = False

            for radius in search_buffers:
                pois = query_osm(lat, lon, radius, category)
                poi_list = [
                    {
                        "Name": poi.get("tags", {}).get("name", "Unknown"),
                        "Type": category if category else poi.get("tags", {}).get("amenity", "Unknown"),
                        "Latitude": poi.get("lat"),
                        "Longitude": poi.get("lon"),
                        "Open Hours": poi.get("tags", {}).get("opening_hours", open_hours),
                    }
                    for poi in pois
                    if poi.get("lat") and poi.get("lon")
                ]

                if poi_list:
                    found_pois = True
                    st.write(f"📍 **{category.capitalize()} within {radius}m of ({lat}, {lon}):**")
                    for poi in poi_list:
                        st.write(
                            f"🏥 **{poi['Name']}** - 📍 ({poi['Latitude']}, {poi['Longitude']}) - ⏰ Open Hours: {poi['Open Hours']}"
                        )
                    break

            if not found_pois:
                st.write(
                    f"❌ No {category if category else 'POI'} found within {search_buffers[-1]}m of ({lat}, {lon}). Try searching a larger area."
                )
