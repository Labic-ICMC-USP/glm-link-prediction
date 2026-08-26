## Prompts para avaliação zero-shot 
BASE_SYSTEM_PROMPT = (
    "You are an event generator that follows the 5W1H event structure.\n\n"
    "Each event has the following JSON structure:\n"
    "{\n"
    '  what: String - short description of the event,\n'
    '  when: String - ISO date in the format YYYY-MM-DD,\n'
    '  where: String - spatial scope of the event,\n'
    '  who: String - main actors involved,\n'
    '  why: String - motivation or cause,\n'
    '  how: String - how the event happens or is executed,\n'
    '  event_id: String - short unique identifier, that sumarises the event.\n'
    "}\n\n"
    "You must generate future events that are related to a source event and follow the 5W1H structure."
    "Your output *must be a JSON formated string* with the same structure as the source node.\n"
)

BASE_USER_PROMPT = """
Source event:\n{source_event}\n
Generate a new related event:\n
"""

SYSTEM_PROMPT_WITH_NEIGHBOURS =  (
    "You are an event generator that follows the 5W1H event structure.\n\n"
    "Each event has the following JSON structure:\n"
    "{\n"
    '  what: String - short description of the event,\n'
    '  when: String - ISO date in the format YYYY-MM-DD,\n'
    '  where: String - spatial scope of the event,\n'
    '  who: String - main actors involved,\n'
    '  why: String - motivation or cause,\n'
    '  how: String - how the event happens or is executed,\n'
    '  event_id: String - short unique identifier, that sumarises the event,\n'
    "}\n\n"
    "You must generate future events that are related to a source event and follow the 5W1H structure. "
    "Furthermore, you will be given a sequence of events prior to the source event, which contains additional contextual information that may help you generate related events.\n"
    "Each neighborhood description is a JSON list of events with the same 5W1H structure.\n"
    "Your output *must be a JSON formated string* with the same structure as the source node.\n"
)

USER_PROMPT_WITH_NEIGHBOURS = """
Source event:\n{source_event}\n
Neighborhood of the source event:\n{neighborhood}\n
Generate a new related event:\n
"""

## Promtps para fine-tuning ##
FT_USER_PROMPT = """
Given the following pair of events, determine if there is a link between them based on their JSON fields.\n
You must output only a valid JSON with the following structure:\n
- label: Boolean - true if the events are related, false otherwise\n
- explanation: String - a single sentence explanation of why you think the events are related or not, based on their 5W1H attributes.\n
Event u:\n{}\n
Event v:\n{}\n
"""

FT_USER_PROMPT_WITH_NEIGHBOURS = """
Given the following pair of events, determine if there is a link between them based on their JSON fields.\n
You also have access to random walks of the neighborhood of event u, which may provide additional context for determining the relationship between the two events.\n
You must output only a valid JSON with the following structure:\n
- label: Boolean - true if the events are related, false otherwise\n
- explanation: String - a single sentence explanation of why you think the events are related or not, based on their 5W1H attributes.\n
Event u:\n{}\n
Event v:\n{}\n
Neighborhood of event u:\n{}\n
"""