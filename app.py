from openai import OpenAI
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import tiktoken

load_dotenv()

client = OpenAI()
GPT_MODEL = "gpt-4-turbo-2024-04-09"

# web scraping
def scrape(url):
    app = FirecrawlApp()

    # Scrape a single URL
    try:
        scraped_data = app.scrape_url(url)
    except Exception as e:
        print("Unable to scrape the url")
        print(f"Exception: {e}")
        return e

    links_scraped.append(url)

    return scraped_data["markdown"]

def search(query, entity_name: str):
    app = FirecrawlApp()

    params = {"pageOptions": {"fetchPageContent": True}}

    # Scrape a single URL
    search_result = app.search(query, params=params)
    search_result_str = str(search_result)

    data_keys_to_search = [obj["name"] for obj in data_points if obj["value"] is None]

    prompt = f"""{search_result_str}
    -----
    About is some search results from the internet about {query}
    Your goal is to find specific list of information about an entity called {entity_name} regarding {data_keys_to_search}.

    Please extract information from the search results above in specific JSON format:

    {{
         "related urls to scrape further": ["url1", "url2", "url3"],
         'info found': [{{
            research_item_1: 'xxxx',
            "reference": url
        }},
        {{
            research_item_2: 'xxxx',
             "reference": url
    }},
    ...]
    }}
    where research_item_1, research_item_2 are the actual research item names you are looking for;
    Only return research_items that you actually found,
    if no research item information found from the content provided, just don't return any

    Extracted JSON:
    """
    response = client.chat.completions.create(
         model=GPT_MODEL, messages=[{"role": "user", "content": prompt}]
)

    result = response.choices[0].message.content

    return result

def update_data(datas_update):
    """
    Update the state with new data points found

    Args:
        state (dict): The current graph state
        datas_update (List[dict]): The new data points found, have to follow the format [{"name": "xxx", "value": yyy}]

    Returns:
        state (dict): The updated graph state
    """
    print(f"Updating the data {datas_update}")

    for data in datas_update:
         for obj in data_points:
             if obj["name"] == data["name"]:
                 obj["value"] = data["value"]
                 obj["reference"] = data["reference"]

    return f"data updated: {data_points}"

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tool_choice, tools, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    
def pretty_print_conversation(message):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }

    if message["role"] == "system":
        print(colored(f"system: {message['content']}", role_to_color[message["role"]]))
    elif message["role"] == "user":
        print(colored(f"user: {message['content']}", role_to_color[message["role"]]))
    elif message["role"] == "assistant" and message.get("tool_calls"):
        print(
            colored(
                f"assistant: {message['tool_calls']}\n", 
                role_to_color[message["role"]],
            )
        )
    elif message["role"] == "assistant" and not message.get("tool_calls"):
        print(colored(f"assistant: {message['content']}", role_to_color[message["role"]]))

tools_list = {"scrape": scrape, "search": search, "update_data": update_data}

def memory_optimise(messages: list):
    system_prompt = messages[0]["content"]

    # token count
    encoding = tiktoken.encoding_for_model(GPT_MODEL)

    if len(messages) > 24 or len(encoding.encode(str(messages))) > 10000:
        latest_messages = messages[-12:]

        token_count_latest_messages = len(encoding.encode(str(latest_messages)))
        print(f"Token count of latest messages: {token_count_latest_messages}")

        index = messages.index(latest_messages[0])
        early_messages = messages[:index]

        prompt = f"""{early_messages}
        -----
        Above is the past history of conversation between user & AI, including actions AI already taken.
        Please summarise the past actions taken so far, what key information learnt & tasks that already completed.
        
        SUMMARY:
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )

        system_prompt = f"""{system_prompt}; Here is a summary of past actions taken so far: {response.choices[0].message.content}"""
        messages = [{"role": "system", "content": system_prompt}] + latest_messages

    else:
        messages.append({"role": "user", "content": (system_prompt + " " + prompt)})

    state = "running"

    for message in messages:
        pretty_print_conversation(message)

    while state == "running":
        chat_response = chat_completion_request(messages, tool_choice=None, tools=tools)

        if isinstance(chat_response, Exception):
            print("Failed to get a valid response:", chat_response)
            state = "finished"
        else:
            current_choice = chat_response.choices[0]
            messages.append(
                {
                    "role": "assistant",
                    "content": current_choice.message.content,
                    "tool_calls": current_choice.message.tool_calls,
                }
            )
            pretty_print_conversation(messages[-1])

            if current_choice.finish_reason == "tool_calls":
                tool_calls = current_choice.message.tool_calls
                for tool_call in tool_calls:
                    # Assuming function and result are defined elsewhere
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function,
                            "content": result,
                        }
                    )
                    pretty_print_conversation(messages[-1])

            if current_choice.finish_reason == "stop":
                state = "finished"

    messages = memory_optimise(messages)

    return messages[-1]["content"]


    # step 1: run agent to do website search
def website_search(entity_name: str, website: str):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "scrape",
                "description": "Scrape a URL for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "the url of the website to scrape",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_data",
                "description": "Save data points found for later retrieval",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "datas_update": {
                            "type": "array",
                            "description": "the data points to update",
                            "items": {
                                "type": "object",
                                "description": "the data point to update, should follow specific json format: {name: xxx, value: yyy}",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "the name of the data point",
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "the value of the data point",
                                    },
                                },
                            },
                        },
                    },
                    "required": ["datas_update"],
                },
            },
        }
    ]

    data_keys_to_search = [obj["name"] for obj in data_points if obj["value"] is None]

    if len(data_keys_to_search) > 0:
        system_prompt = """
    You are a world-class web researcher.
    You will keep scraping URLs based on information you received until information is found.

    You will try as hard as possible to search for all sorts of different queries & sources to find information.
    You do not stop until all information is found, it is very important we find all information, I will guide you.
    Whenever you find a certain data point, use the "update_data" function to save the data point.

    You only answer questions based on results from the scraper, do not make things up.
    You never ask the user for inputs or permissions, you just do your job and provide the results.
    You ONLY run 1 function at a time, do NEVER run multiple functions at the same time.
    """

    prompt = f"""
    Entity to search: {entity_name}

    Links we already scraped: {links_scraped}
   
    Data points to find:
    {data_keys_to_search}
"""

    response = call_agent(prompt, system_prompt, tools, plan=False)

    return response

    links_scraped = []

    data_points = [
        {"name": "catering_offering_for_employees", "value": None, "reference": None},
        {"name": "num_employees", "value": None, "reference": None},
        {"name": "office_locations", "value": None, "reference": None},
]

    entity_name = "Discord"
    website = "https://discord.com/"

    response1 = website_search(entity_name, website)
    response2 = internet_search(entity_name)

    print("------")
    print(f"Data points found: {data_points}")

# Assuming 'entity_name' and 'links_scraped' are already defined elsewhere in your code




