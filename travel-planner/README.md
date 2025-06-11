# 🧳 Istanbul Travel Planner

Welcome to the **Istanbul Travel Planner**, an AI-powered web app that helps you generate a personalized travel itinerary for your trip to Istanbul, Turkey. With just a few preferences selected, this app will create a day-by-day plan, highlight places to visit, and visualize them on an interactive map.

## 🚀 Features

- 🎯 Custom itinerary generation based on your preferences
- 🧠 AI-powered text generation and location extraction
- 🗺️ Interactive map with geocoded destinations
- 📥 Downloadable itinerary in `.txt` format
- 🖥 Built with [Streamlit](https://streamlit.io/)

## 🧰 Technologies Used

- **Streamlit** – for building the web interface
- **OpenAI/LLM APIs** – for generating the itinerary and extracting place names (via `generate_itinerary` and `extract_places_from_text`)
- **Folium** – for rendering interactive maps
- **Geopy** – for geocoding place names
- **Custom Utilities**:
  - `utils/formatter.py` – formats the prompt for itinerary generation
  - `export/exporter.py` – supports exporting itinerary as plain text

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/istanbul-travel-planner.git
   cd istanbul-travel-planner
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
🎛 Usage
Open the app in your browser (typically at http://localhost:8501)

Fill in your trip details in the sidebar:

Number of days

Travel interests

Dietary preferences

Desired pace and season

Click Generate Itinerary

View your customized plan, explore the map, and optionally download it.

📂 Project Structure
bash
Copy
Edit
├── app.py                    # Main Streamlit app
├── itinerary/
│   ├── generator.py          # AI functions to generate and extract info from itinerary text
├── utils/
│   ├── formatter.py          # Utility to format prompt for itinerary generation
├── export/
│   ├── exporter.py           # Itinerary export functions
├── requirements.txt          # Python dependencies
└── README.md                 # Project readme file
🧠 Notes
This app relies on AI APIs and internet access for geocoding; ensure network availability.

Add your own API keys or credentials if required by the AI or geocoding services.

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

🤝 Contributions
Feel free to fork the repository, suggest features, or submit pull requests. Contributions are welcome!

vbnet
Copy
Edit

Let me know if you’d like the `requirements.txt` file or help packaging this into a deployable app (e.
