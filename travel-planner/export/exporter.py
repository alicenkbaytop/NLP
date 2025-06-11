import os

def export_itinerary_text(itinerary_text: str, filename: str = "itinerary.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(itinerary_text)
    return filename
