from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
import json


load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="gemma2-9b-it",
    temperature=0.3,
)

def generate_itinerary(prompt: str) -> str:
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def extract_places_from_text(itinerary_text: str) -> list:
    instruction = (
        "Extract all named locations (e.g., landmarks, museums, squares, districts) "
        "from this travel itinerary. Return ONLY a JSON array of strings. "
        "Example: [\"Hagia Sophia\", \"Topkapi Palace\"]"
    )

    prompt = f"{instruction}\n\nText:\n{itinerary_text}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        places = json.loads(response.content.strip())
        if isinstance(places, list):
            return list(set(places))  # remove duplicates
        else:
            return []
    except Exception as e:
        print("Error parsing LLM response:", e)
        print("Raw response:", response.content)
        return []