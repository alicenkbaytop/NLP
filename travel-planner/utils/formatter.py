def format_prompt(data):
    interests = ', '.join(data['interests']) if data['interests'] else 'None'
    diet = data['diet'] if data['diet'] != "None" else 'No dietary preference'
    pace = data['pace']
    season = data['season']
    
    # Construct the prompt without using "budget"
    prompt = f"""
    Generate a personalized 5-day itinerary for Istanbul based on the following preferences:
    - Number of days: {data['days']}
    - Interests: {interests}
    - Dietary preference: {diet}
    - Travel pace: {pace}
    - Preferred season: {season}
    
    The itinerary should include daily activities, suggested meals, transportation options, and any recommendations based on the chosen preferences.
    """
    
    return prompt
