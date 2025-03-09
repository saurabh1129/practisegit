import autogen
import requests
from flask_cors import CORS
from typing import Annotated
from configparse import ConfigParser
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model routes from config
config = ConfigParser

ROUTES = {
    "OFD": config.read_config("model_routes", "OFD"),
    "TKG": config.read_config("model_routes", "TKG"),
    "CAUSAL": config.read_config("model_routes", "CAUSAL")
}

config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST.json",
    filter_dict={"model": ["llama-3.3-70b-versatile"]},
)

llm_config = {
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0.2,
    "api_type": "groq",
}

# Define the planner agent
planner_agent = autogen.ConversableAgent(
    name="planner_agent",
    system_message="""
    I am the planner agent, I am an expert in planning and orchestrating the workflow of the system. I do the following tasks:
    1. I take the user input/user query and then send it to the decomposer agent.
    2. The decomposer agent then sends back the decomposed queries.
    3. I then send the decomposed queries to the classifier agent.
    """,
    llm_config=llm_config
)

# Define the decomposer agent
decomposer_agent = autogen.ConversableAgent(
    name="decomposer_agent",
    system_message="""
    I am the decomposer agent, I am an expert in decomposing the user query into smaller sub-queries, I don't answer the user query, I just decompose it. I do the following tasks:
    1. I take the user input/user query from the planner agent and then decompose it into smaller sub-queries.
    2. I create a list where the very first item is the original user input (main question), followed by the decomposed sub-queries.
    3. I send this list of queries back to the planner agent, formatted as a single string with each query on a new line, prefixed with 'query:' (e.g., 'query:Original user question\nquery:Sub-query 1\nquery:Sub-query 2').
    """,
    llm_config=llm_config
)

# Define the classifier agent
classifier_agent = autogen.ConversableAgent(
    name="classifier_agent",
    system_message=""" 
    I am the classifier agent, I am an expert in classifying tasks into different categories, 
    I don't answer the sub-queries, I just classify them. I do the following tasks:

    1. I take the decomposed sub-queries from the decomposer agent or planner agent and then classify them into different categories.
    I will strictly classify the sub-queries into these categories:

        1.1. **OFD (Ontological Functional Dependency)**:
            Questions focus on identifying entities, attributes, and their functional dependencies within a dataset, 
            as well as how entities relate and influence each other.

            **Key Traits**:
            - Entity/Attribute Identification
            - Dependency Discovery (how entities depend on each other)
            - Relationship Analysis
            - Dataset-Based Reasoning (identifying patterns and dependencies)

            **Examples**:
            - What are the entities and attributes in this dataset?
            - How do entities depend on each other?
            - What functional dependencies exist between attributes?

            **Classification**: If the query focuses on relationships, dependencies, or dataset-based reasoning, classify it as OFD.

        1.2. **TKG (Temporal Knowledge Graph)**:
            Questions focus on time-based reasoning, event progression, and relationships evolving over time.

            **Key Traits**:
            - Time-based Reasoning (how entities/events evolve)
            - Event Progression (sequences of events or changes over time)
            - Time-constrained Facts (valid info at specific times)
            - Temporal Causal Sequences (cause and effect with timestamps)

            **Examples**:
            - What was the market trend from 2020-2023?
            - How did mergers affect stock prices over the past decade?
            - Who were the US presidents in the 1990s?

            **Classification**: If the query involves time or evolving relationships, classify it as TKG.

        1.3. **CAUSAL (Causal Question)**:
            Questions focus on understanding cause-and-effect relationships between variables.

            **Key Traits**:
            - Cause-and-Effect Relationships
            - Counterfactual Reasoning (what would happen if X didn't occur?)
            - Intervention Scenarios (how changes in X impact Y)
            - Causal Mechanisms (how X leads to Y)
            - Temporal Precedence (did X occur before Y, and does it influence Y?)

            **Examples**:
            - How does temperature affect CPU performance?
            - What is the impact of sanctions on GDP?
            - If a player is substituted, how does it affect the team's win probability?

            **Classification**: If the query involves cause-and-effect or intervention, classify it as CAUSAL.

    2. I send the classified sub-queries to the executor agent.
    
    I classify sub-queries into: OFD, TKG, or CAUSAL. 
    I return each sub-query in the format 'category:<category>, query:<sub-query_text>'.
    """,
    llm_config=llm_config
)


# Define the executor agent with function calling capability
executor_agent = autogen.ConversableAgent(
    name="executor_agent",
    system_message="""
    I am the executor agent. I specialize in routing classified sub-queries to their appropriate endpoints by calling the execute_task function for each sub-query. 
    I never answer the classified sub-queries myself. My only job is to route them correctly and wait for responses. Once I have received all the responses, I will pass them to the aggregator_agent by saying '@aggregator_agent, here are the responses: [list of responses]'. I do not provide any final answer or summary; that is the role of the aggregator_agent.

    I route classified sub-queries to their endpoints using execute_task and collect responses. After receiving all responses, I pass them to the aggregator_agent with '@aggregator_agent, here are the responses: [list]' and say 'Task complete. aggregator_agent, please proceed to perform your tasks' to ensure the conversation continues.

    Here are the correct routes:
     - **OFD (Ontological Functional Dependency):** {ROUTES['OFD']}
     - **TKG (Temporal Knowledge Graph):** {ROUTES['TKG']}
     - **CAUSAL (Cause and Effect):** {ROUTES['CAUSAL']}

    Responsibilities:
    1. I take the classified sub-queries from the classifier agent.
    2. For each sub-query, I call execute_task with the classified sub-query string and wait for the response.
    3. After all responses are received, I compile them and pass them to the aggregator_agent using the format '@aggregator_agent, here are the responses: [response1], [response2], ...'
    4. I collect all the responses and pass them to the aggregator_agent for further processing.
    5. I do not provide any final answer or summary; that is the role of the aggregator_agent.
    """,
    llm_config=llm_config
)

# Define the aggregator agent
aggregator_agent = autogen.ConversableAgent(
    name="aggregator_agent",
    system_message="""
    I am the aggregator agent, I am expert in aggregating the different category responses. I never answer the sub-queries myself; I just aggregate the responses. I do the following tasks:
    1. I wait until the executor agent has processed all sub-queries and their responses are available in the chat history.
    2. I collect all the responses from the executor agent.
    3. I prepare a final response that starts with 'The final answer is:' followed by a summary of all the answers received from the executor agent.
    4. My final response must always begin with 'The final answer is:' to clearly indicate the conclusion.

    I wait for the executor_agent to send '@aggregator_agent, here are the responses:' followed by a list. I then summarize the responses and provide a final answer starting with 'The final answer is:' , marking the end of the conversation.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Define the execute_task function
def execute_task(classified_task: Annotated[str, "The classified sub-query in the format 'category:<category>, query:<sub-query_text>'"]) -> dict:
    """Executes a task based on classification, sending sub-queries to model endpoints."""
    try:
        # Debug: Print the classified task to verify input
        print(f"Classified task received: {classified_task}")

        # Extract category and query from the string
        if "category:" not in classified_task or "query:" not in classified_task:
            return {"error": "Malformed classified task"}

        category_part = classified_task.split("category:")[1].split(",")[0].strip()
        query_part = classified_task.split("query:")[1].strip()

        category = category_part
        query = query_part

        # Debug: Print category and query to verify parsing
        print(f"Category: {category}, Query: {query}")

        if category not in ROUTES:
            return {"error": f"Invalid category: {category}"}

        # Strip any leading or trailing quotes from the route URL
        route_url = ROUTES[category].strip("'\"") # Remove single and double quotes and \ since we only send / in the route
        print(f"Sending to route: {route_url}")

        # Send query to external service
        payload = {"query": query}
        headers = {'Content-Type': 'application/json'}

        response = requests.post(route_url, json=payload, headers=headers, verify=False)
        response.raise_for_status()
        
        print(f"Response from {route_url}: {response.json()}")
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Execution error: {str(e)}"}    

# Register the function for both LLM and execution on executor_agent
executor_agent.register_for_llm(name="execute_task", description="Execute a classified sub-query by sending it to the appropriate external model and returning the response")(execute_task)
executor_agent.register_for_execution(name="execute_task")(execute_task)

# Create a GroupChat with all existing agents
groupchat = autogen.GroupChat(
    agents=[
        planner_agent,
        decomposer_agent,
        classifier_agent,
        executor_agent,
        aggregator_agent
    ],
    messages=[]
)

def is_termination_msg(message):
    # Stop if the message is from aggregator_agent and contains "The final answer is:" anywhere in the content
    return message["name"] == "aggregator_agent" and "The final answer is:" in message["content"].strip()

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config, is_termination_msg=is_termination_msg)

def process_query(user_input):
    """Processes the user input through the multi-agent system."""
    try:
        result = manager.initiate_chat(
            planner_agent,  # Start with the planner here manager/chat_manager is taking the user_query/input and sending it to the planner agent
            message=f"User query: {user_input}"
        )
        return {"result": result.chat_history}  # Return chat history for clarity

    except Exception as e:
        return {"error": str(e)}

@app.route("/user_query", methods=["POST"])
def user_query():
    """Handles user input and processes through agents."""
    data = request.json or {}
    user_input = data.get("user_input", "")

    if not user_input:
        return jsonify({"error": "No user input provided"}), 400

    response = process_query(user_input)
    return jsonify({"Processed Query": response}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=8008)
    