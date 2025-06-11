import streamlit as st
from itinerary.generator import generate_itinerary, extract_places_from_text

from utils.formatter import format_prompt
from export.exporter import export_itinerary_text
import folium
from geopy.geocoders import Nominatim
from streamlit_folium import folium_static

# Streamlit page configuration
st.set_page_config(page_title="Istanbul Travel Planner", layout="wide", page_icon="🗺")
st.title("🧳 Istanbul Travel Planner")

# Sidebar for form inputs
with st.sidebar:
    st.subheader("📝 Trip Details")

    with st.form("trip_form"):
        days = st.slider("How many days are you staying?", 1, 14, 5)
        interests = st.multiselect("Select your interests", 
            ["Historical", "Cultural", "Nature", "Shopping", "Food", "Nightlife"])
        diet = st.selectbox("Dietary preference", 
            ["None", "Vegan", "Vegetarian", "Halal", "Gluten-free"])
        pace = st.selectbox("Travel pace", ["Relaxed", "Moderate", "Packed"])
        season = st.selectbox("Preferred season", ["Spring", "Summer", "Autumn", "Winter"])

        submitted = st.form_submit_button("Generate Itinerary")

# Geocoding function to get coordinates from place names
def get_coordinates(location_name):
    geolocator = Nominatim(user_agent="travel_planner")
    try:
        location = geolocator.geocode(f"{location_name}, Istanbul, Turkey", timeout=10)
        if location:
            return location.latitude, location.longitude
    except:
        return None
    return None

# Main content area for itinerary display and map
if submitted:
    with st.spinner("Generating your itinerary with AI..."):
        data = {
            "days": days, "interests": interests, "diet": diet,
            "pace": pace, "season": season
        }
        prompt = format_prompt(data)
        plan = generate_itinerary(prompt)

    # Display the itinerary nicely
    st.subheader("📋 Your Personalized Itinerary")
    plan_with_bold = plan.replace("**", "").replace("**", "")
    st.markdown(f"<div style='background-color:#f7f7f7;padding:10px;border-radius:8px;'><pre>{plan_with_bold}</pre></div>", unsafe_allow_html=True)

    # Ask LLM to extract place names from plan text
    with st.spinner("🔍 Extracting locations with AI..."):
        extracted_places = extract_places_from_text(plan)

    st.subheader("🗺️ Places Found")
    if extracted_places:
        st.markdown(", ".join(extracted_places))
    else:
        st.warning("No locations found.")

    # Create a map
    if extracted_places:
        map_ = folium.Map(location=[41.0082, 28.9784], zoom_start=13)
        for place in extracted_places:
            coords = get_coordinates(place)
            if coords:
                folium.Marker(location=coords, popup=place).add_to(map_)

        st.subheader("📍 Your Travel Map")
        folium_static(map_)

    # Export itinerary as TXT
    if st.download_button("📥 Download Itinerary (TXT)", plan, file_name="istanbul_itinerary.txt"):
        st.success("Download complete!")
