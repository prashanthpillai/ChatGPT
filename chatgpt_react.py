import openai
import os
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from dotenv import load_dotenv

# Load API credentials
load_dotenv()
openai_organization = os.getenv("OPENAIORG")
openai_api_key = os.getenv("OPENAIAPIKEY")
serp_api_key = os.getenv("SERPAPIKEY")

# Model selection
model_name = "gpt-4"
model_name = "gpt-3.5-turbo"

# LLM agents
llm = OpenAI(model_name=model_name, temperature=0, openai_api_key=openai_api_key)
tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=serp_api_key)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# LLM prediction
question = "find the launch date and landing date of Chandrayaan-3."
agent.run(question)

