## Prompts para avaliação zero-shot 
BASE_SYSTEM_PROMPT = (
"You are an event graph link prediction model.\n"
"Your task is to analyze samples from an event graph and decide whether a candidate event should be connected to a given anchor event.\n"
"Each event is described using structured 5W1H metadata: what, who, when, where, why, and how.\n"
"At the end of the prompt, you will receive:\n"
"1. an anchor event A;\n"
"2. a candidate event B.\n"
"You must decide whether there should be a link from A to B in the event graph.\n"
"A link means that event B is plausibly related to event A according to semantic, temporal, geographic, or causal continuity.\n"
"Return only a valid JSON object with the following fields:\n"
"{\n"
"""   "explanation": String - brief explanation of the decision,\n"""
"""   "link": Boolean - true if there should be a link from A to B, false otherwise.\n"""
"}\n"
"Do not return markdown.\n"
"Do not return any text outside the JSON object."
)

BASE_USER_PROMPT = """Given the following pair of events, determine if there is a link between the anchor event A and the candidate event B.\n
Anchor event A:\n{}\n
Candidate event B:\n{}\n
Based on the above, return only a JSON object with "explanation" (your reasoning) and "link" (true or false).\n 
"""

### V3 #####
SYSTEM_PROMPT_WITH_NEIGHBOURS =  (
"You are an event graph link prediction model.\n"
"Given an anchor event A and a candidate event B, each described with structured 5W1H metadata (what, who, when, where, why, how), decide whether a directed link A - B exists.\n"
"Only judge whether B plausibly continues from A, never the reverse.\n"
"Default to false. Predict true only if at least one of the criteria below holds strongly:\n"
"1. Causal — B is a direct consequence or effect of A.\n"
"2. Temporal — B is a direct follow-up to A in a chain of events.\n"
"3. Semantic — A and B describe different stages, aspects, or consequences of the same incident.\n"
"4. Geographic — A and B occur in related locations as part of the same incident.\n"
"If none of these hold strongly, predict false.\n"
"Do not predict true based on shared topic, actors, or location alone, temporal proximity alone, or indirect links inferable only from general world knowledge.\n"
"You will receive a random walk of events preceding A, oldest to newest.\n"
"The random walk provides local context only. Use it to assess whether B is consistent with the neighborhood of A, but never use it as evidence of a direct link.\n"
"Events in the random walk are listed from oldest to newest.\n."
"Return only a valid JSON object with the following fields:\n"
"{\n"
"""   "explanation": String - brief explanation of the decision,\n"""
"""   "link": Boolean - true or false.\n"""
"}\n"
"Do not return markdown.\n"
"Do not return any text outside the JSON object."
)

USER_PROMPT_WITH_NEIGHBOURS = """
Given the following pair of events, determine if there is a link between the anchor event A and the candidate event B.\n
Anchor event A:\n{}\n
Candidate event B:\n{}\n
Random walk context (oldest to newest):\n{}\n
Based on the above, return only a JSON object with "explanation" (your reasoning) and "link" (true or false).\n 
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