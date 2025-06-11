# Istanbul Travel Planner 🧳

A Streamlit web application that generates personalized travel itineraries for Istanbul using AI, complete with interactive maps and location visualization.

## Features

- **AI-Powered Itinerary Generation**: Creates customized travel plans based on your preferences
- **Interactive Maps**: Visualizes recommended locations on an interactive Folium map
- **Flexible Customization**: Adjust trip duration, interests, dietary needs, pace, and season
- **Location Extraction**: Automatically identifies and maps places mentioned in your itinerary
- **Export Functionality**: Download your itinerary as a text file for offline use

## Requirements

```
streamlit
folium
geopy
streamlit-folium
```

## Installation

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install streamlit folium geopy streamlit-folium
   ```
3. Ensure you have the following modules in your project structure:
   - `itinerary/generator.py` - Contains `generate_itinerary()` and `extract_places_from_text()` functions
   - `utils/formatter.py` - Contains `format_prompt()` function
   - `export/exporter.py` - Contains `export_itinerary_text()` function

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Configure your trip in the sidebar:
   - **Days**: Select trip duration (1-14 days)
   - **Interests**: Choose from Historical, Cultural, Nature, Shopping, Food, Nightlife
   - **Diet**: Specify dietary preferences (None, Vegan, Vegetarian, Halal, Gluten-free)
   - **Pace**: Select travel intensity (Relaxed, Moderate, Packed)
   - **Season**: Choose preferred season (Spring, Summer, Autumn, Winter)

3. Click "Generate Itinerary" to create your personalized plan

4. View your itinerary, explore locations on the interactive map, and download as needed

## Application Structure

### Main Components

- **Sidebar Form**: Collects user preferences for itinerary customization
- **Itinerary Display**: Shows the AI-generated travel plan with formatted text
- **Location Extraction**: Uses AI to identify place names from the generated itinerary
- **Interactive Map**: Displays all extracted locations as markers on a Folium map centered on Istanbul
- **Export Feature**: Allows users to download their itinerary as a text file

### Key Functions

- `generate_itinerary(prompt)`: Generates travel itinerary using AI based on formatted prompt
- `extract_places_from_text(plan)`: Extracts location names from itinerary text using AI
- `format_prompt(data)`: Formats user preferences into a prompt for AI processing
- `get_coordinates(location_name)`: Geocodes location names to latitude/longitude coordinates

## Features in Detail

### Geocoding
Uses Nominatim geocoder to convert place names to coordinates for map visualization. Includes error handling for failed geocoding attempts.

### Map Visualization
- Centers on Istanbul (coordinates: 41.0082, 28.9784)
- Zoom level 13 for optimal city-level view
- Interactive markers for each identified location
- Popup labels showing place names

### User Interface
- Wide layout for better map and content display
- Responsive sidebar for input controls
- Loading spinners for AI processing steps
- Success/warning messages for user feedback
- Styled itinerary display with background formatting

## Dependencies

This application requires several custom modules that should be implemented separately:

1. **itinerary.generator**: AI-powered itinerary generation and place extraction
2. **utils.formatter**: Prompt formatting utilities
3. **export.exporter**: Export functionality for itineraries

## Configuration

The application is configured for Istanbul-specific travel planning with:
- Default map center on Istanbul
- Geocoding searches scoped to "Istanbul, Turkey"
- User agent "travel_planner" for Nominatim requests
- 10-second timeout for geocoding requests

## Error Handling

- Graceful handling of geocoding failures
- Timeout protection for API calls
- Fallback behavior when no locations are extracted
- User-friendly error messages and warnings
