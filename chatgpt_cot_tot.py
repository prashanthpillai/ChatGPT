import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

# Load API credentials
load_dotenv()
openai_organization = os.getenv("OPENAIORG")
openai_api_key = os.getenv("OPENAIAPIKEY")
serp_api_key = os.getenv("SERPAPIKEY")

# Model selection
#model_name = "gpt-4"
model_name = "gpt-3.5-turbo"

# LLM agents
llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=openai_api_key)

# Prompt template
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Prediction -  Scenario 1
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = """Bob is in the living room.
He walks to the kitchen, carrying a cup.
He puts a ball in the cup and carries the cup to the bedroom.
He turns the cup upside down, then walks to the garden.
He puts the cup down in the garden, then walks to the garage.
Where is the ball?"""
'''response = llm_chain.run(question)
print('Scenario 1:', response)'''

# Prediction -  Scenario 3 - TOT
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = """
Imagine three different experts are answering this question.
All experts will write down 1 step of their thinking,
then share it with the group.
Then all experts will go on to the next step, etc.
If any expert realises they're wrong at any point then they leave.
The question is...

Bob is in the living room.
He walks to the kitchen, carrying a cup.
He puts a ball in the cup and carries the cup to the bedroom.
He turns the cup upside down, then walks to the garden.
He puts the cup down in the garden, then walks to the garage.
Where is the ball?"""
response = llm_chain.run(question)
print('Scenario 3:', response)