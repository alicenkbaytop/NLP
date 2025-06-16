# 🌍 ChatGIS

**ChatGIS** is an interactive geospatial chatbot and mapping tool that combines AI-powered natural language processing with OpenStreetMap data. It allows users to discover nearby Points of Interest (POIs) such as restaurants, hospitals, parks, and more, either by entering natural language queries or manually clicking on a map.

---

## 🚀 Features

- 🔍 **Natural Language Search** using an AI LLM (via [Ollama](https://ollama.com)) to interpret queries like:
  - _"Find cafes within 2km of Kasımpaşa, İstanbul"_
- 🗺️ **Map-based Search** by clicking on a map and selecting POI categories and radius
- 📍 **OpenStreetMap Integration** for POI data and geocoding
- 🧠 **LLM Location Extraction** that extracts and geocodes meaningful locations from text
- 📊 POI result listing with name, location, and open hours
- 📦 Modular and extensible backend with support for additional POI types

---

## 🛠️ Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Geocoding & Maps**:
  - [OSMnx](https://github.com/gboeing/osmnx)
  - [Folium](https://python-visualization.github.io/folium/)
  - [OpenStreetMap Overpass API](https://overpass-api.de/)
  - [Geopy](https://geopy.readthedocs.io/en/stable/)
- **AI/NLP**: [Ollama](https://ollama.com) (uses models like `qwen3`)
- **Data Handling**: `pandas`, `json`, `re`, `shapely`

---

## 📦 Installation

1. **Clone the repository**
    ```git clone ```
    ``` cd chatgis ```
2. **Install requirements.**
    ```pip install -r requirements.txt```

3. **Run the app**
    ```streamlit run app.py```


## 💬 Example Queries
* "Find hospitals near Taksim, İstanbul"

* "Show me banks within 1000 meters of Kadıköy"

* "Where are the parks around Beşiktaş?"
