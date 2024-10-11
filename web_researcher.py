import json
import os
from typing import List, Dict, Any
import tiktoken
from openai import OpenAI
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import requests

# TEMPORARY: Direct API key setting (remove this in production)
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your actual OpenAI API key
os.environ["FIRECRAWL_API_KEY"] = "fc-2644a0521b00476b9ae58d195df7caf3"  # Replace with your actual Firecrawl API key
print("WARNING: API keys are set directly in the code. This is for testing only. Remove before production use.")

# Load environment variables (this will use the keys we just set)
load_dotenv()

def get_api_key(key_name):
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"No {key_name} found. Please set the {key_name} environment variable.")
    return api_key

# Get the API keys
openai_api_key = get_api_key("OPENAI_API_KEY")
firecrawl_api_key = get_api_key("FIRECRAWL_API_KEY")
print(f"OpenAI API key loaded: {'Yes' if openai_api_key else 'No'}")
print(f"OpenAI API key length: {len(openai_api_key) if openai_api_key else 'N/A'}")
print(f"Firecrawl API key loaded: {'Yes' if firecrawl_api_key else 'No'}")
print(f"Firecrawl API key length: {len(firecrawl_api_key) if firecrawl_api_key else 'N/A'}")

client = OpenAI(api_key=openai_api_key)
GPT_MODEL = "gpt-4-1106-preview"  # This is the GPT-4 Turbo model
MAX_TOKENS = 4096  # Adjusted for GPT-4 Turbo
MAX_MESSAGES = 24
SUMMARY_MESSAGES = 12

class WebResearcher:
    def __init__(self, entity_name: str, website: str):
        self.entity_name = entity_name
        self.website = website
        self.links_scraped = []
        self.data_points = [
            {"name": "product_or_service", "value": None, "reference": None},
            {"name": "company_size", "value": None, "reference": None},
            {"name": "headquarters_location", "value": None, "reference": None},
            {"name": "email", "value": None, "reference": None},
            {"name": "phone", "value": None, "reference": None},
        ]
        self.firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)

    def scrape(self, url: str) -> str:
        try:
            print(f"Attempting to scrape URL: {url}")
            scraped_data = self.firecrawl_app.scrape_url(url)
            self.links_scraped.append(url)
            print(f"Successfully scraped URL: {url}")
            return scraped_data["markdown"]
        except requests.exceptions.RequestException as e:
            print(f"Request exception while scraping {url}: {e}")
            return str(e)
        except Exception as e:
            print(f"Unexpected error while scraping {url}: {e}")
            return str(e)

    def search(self, query: str) -> str:
        params = {"pageOptions": {"fetchPageContent": True}}
        search_result = self.firecrawl_app.search(query, params=params)
        data_keys_to_search = [obj["name"] for obj in self.data_points if obj["value"] is None]
        
        prompt = f"""{str(search_result)}
        -----
        About is some search results from the internet about {query}
        Your goal is to find specific list of information about an entity called {self.entity_name} regarding {data_keys_to_search}.

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
        return response.choices[0].message.content

    def update_data(self, datas_update: List[Dict[str, Any]]) -> str:
        for data in datas_update:
            for obj in self.data_points:
                if obj["name"] == data["name"]:
                    obj["value"] = data["value"]
                    obj["reference"] = data["reference"]
        return f"data updated: {self.data_points}"

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(self, messages: List[Dict[str, Any]], tool_choice: Any, tools: List[Dict[str, Any]]) -> Any:
        try:
            return client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception as e:
            print(f"Unable to generate ChatCompletion response: {e}")
            return e

    def pretty_print_conversation(self, message: Dict[str, Any]) -> None:
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "tool": "magenta",
        }
        role = message["role"]
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", "")
        
        if role == "assistant" and tool_calls:
            print(colored(f"assistant: {tool_calls}\n", role_to_color[role]))
        else:
            print(colored(f"{role}: {content}", role_to_color[role]))

    def memory_optimise(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        encoding = tiktoken.encoding_for_model(GPT_MODEL)
        
        if len(messages) > MAX_MESSAGES or len(encoding.encode(str(messages))) > MAX_TOKENS:
            latest_messages = messages[-SUMMARY_MESSAGES:]
            early_messages = messages[:-SUMMARY_MESSAGES]
            
            summary_prompt = f"""{early_messages}
            -----
            Above is the past history of conversation between user & AI, including actions AI already taken.
            Please summarise the past actions taken so far, what key information learnt & tasks that already completed.
            
            SUMMARY:
            """
            summary_response = client.chat.completions.create(
                model=GPT_MODEL, messages=[{"role": "user", "content": summary_prompt}]
            )
            
            system_prompt = f"{messages[0]['content']}; Here is a summary of past actions taken so far: {summary_response.choices[0].message.content}"
            return [{"role": "system", "content": system_prompt}] + latest_messages
        
        return messages

    def call_agent(self, prompt: str, system_prompt: str, tools: List[Dict[str, Any]], plan: bool = False) -> str:
        messages = []
        if plan:
            messages.append({"role": "user", "content": f"{system_prompt} {prompt} Let's think step by step, make a plan first"})
            chat_response = self.chat_completion_request(messages, tool_choice="none", tools=tools)
            print(chat_response.choices[0].message.content)
            messages = [
                {"role": "user", "content": f"{system_prompt} {prompt}"},
                {"role": "assistant", "content": chat_response.choices[0].message.content},
            ]
        else:
            messages.append({"role": "user", "content": f"{system_prompt} {prompt}"})

        for message in messages:
            self.pretty_print_conversation(message)

        while True:
            chat_response = self.chat_completion_request(messages, tool_choice=None, tools=tools)
            
            if isinstance(chat_response, Exception):
                print(f"Failed to get a valid response: {chat_response}")
                break
            
            current_choice = chat_response.choices[0]
            messages.append({
                "role": "assistant",
                "content": current_choice.message.content,
                "tool_calls": current_choice.message.tool_calls,
            })
            self.pretty_print_conversation(messages[-1])

            if current_choice.finish_reason == "tool_calls":
                for tool_call in current_choice.message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "scrape":
                        result = self.scrape(function_args["url"])
                    elif function_name == "search":
                        result = self.search(function_args["query"])
                    elif function_name == "update_data":
                        result = self.update_data(function_args["datas_update"])
                    else:
                        result = f"Unknown function: {function_name}"
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": result,
                    })
                    self.pretty_print_conversation(messages[-1])
            
            if current_choice.finish_reason == "stop":
                break

        messages = self.memory_optimise(messages)
        return messages[-1]["content"]

    def website_search(self) -> str:
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
                                    "description": "the data point to update, should follow specific json format: {name: xxx, value: yyy, reference: zzz}",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "the name of the data point",
                                        },
                                        "value": {
                                            "type": "string",
                                            "description": "the value of the data point",
                                        },
                                        "reference": {
                                            "type": "string",
                                            "description": "the reference URL for the data point",
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

        data_keys_to_search = [obj["name"] for obj in self.data_points if obj["value"] is None]

        if data_keys_to_search:
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
            Entity to search: {self.entity_name}

            Links we already scraped: {self.links_scraped}
        
            Data points to find:
            {data_keys_to_search}
            """

            return self.call_agent(prompt, system_prompt, tools, plan=False)
        
        return "All data points have been found."

# Ensure the class is importable
__all__ = ['WebResearcher']
