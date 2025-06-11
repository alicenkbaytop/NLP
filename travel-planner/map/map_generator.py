import folium
from folium.plugins import MarkerCluster

def generate_map(itinerary_locations):
    m = folium.Map(location=[41.0082, 28.9784], zoom_start=12)  # Istanbul center
    marker_cluster = MarkerCluster().add_to(m)
    
    for day, places in itinerary_locations.items():
        for idx, place in enumerate(places):
            folium.Marker(
                location=place['coords'],
                popup=f"{day}: {place['name']}",
                tooltip=place['name'],
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_cluster)
    return m
