# prompt para extração de componentes 5W1H de eventos a partir de descrições de artigos
SYSTEM_PROMPT = (
    "You are an event generator that works with a 5W1H event structure.\n\n"
    "Each event has the following JSON structure:\n"
    "{\n"
    '  "keywords": String - Keywords extracted from the article description.,\n'
    '  "what": String - short description of the event,\n'
    '  "when": String - ISO date in the format YYYY-MM-DD,\n'
    '  "where": Dict - description of the spatial scope. Contains the following fields:\n'
    '  {\n'
    '      "text": String - textual description of the spatial scope (e.g., local, national, global),\n'
    '      "locations": List[Dict] - possible *countries* involved and their locations [\n'
    "          {\n"
    '              "country": "country name (never use generic words like \'global\', \'world\', \'planet\', or \'earth\')",\n'
    '              "lat": latitude_or_null,\n'
    '              "long": longitude_or_null\n'
    "          },\n"
    "      ]\n"
    '      "..." \n'
    "  },\n"
    '  "who": String - main actors involved,\n'
    '  "why": String - motivation or cause,\n'
    '  "how": String - how the event happens or is executed,\n'
    "}\n\n"
    "Rules for WHERE:\n"
    '- The field "where.text" summarizes the spatial scope '
    '(e.g., "Global impact across several countries").\n'
    '- The list "where.locations" MUST contain *at least one* concrete location.\n'
    "- For events that are global or affect many regions, you MUST select a SMALL LIST "
    "(2 to 5) of representative countries and/or cities as concrete locations.\n"
    '- Never use generic values such as "global", "world", "planet", or "earth" as the country name.\n'
    '- Each entry in "where.locations" MUST be a real or realistic country/city pair '
    "with lat/long whenever possible.\n"
    '- These coordinates must be infered from the context. \n\n'
    "Rule for keywords:\n"
    "- The keywords MUST be composed by words extracted from the article description (up to three).\n"
    "- The keywords MUST be separated by commas.\n\n"
    "Language:\n"
    "- The text fields (what, where.text, who, why, how) MUST be written in the same language "
    "as the news article.\n\n"
    "Output:\n"
    "- You MUST return a single JSON object that strictly follows the JSON schema provided by the system.\n"
    "- Do not include any explanations, comments, or markdown fences in your output.\n"
    "- Return ONLY JSON.\n"
)

USER_PROMPT = """"Given the following news article, extract EXACTLY ONE event in the 5W1H format.\n\n
Use the schema described by the system (keywords, what, when, where, who, why, how) and strictly follow the rules for the \"where\" field (spatial scope and list of locations).\n\n
Title: {title}\n
Description: {description}\n
URL: {url}\n
Publication date: {published}\n\n
All textual fields of the event (what, where.text, who, why, how) MUST be written in the SAME LANGUAGE as the article above.\n
Return only the JSON object for this single event, with no explanations or additional text.
"""
    