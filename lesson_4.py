
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Initialize the model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    ),
    temperature=0.3
)

# 2. Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sustainability consultant. "
        "Explain environmental topics in simple language "
        "for small business owners."
    ),
    (
        "human",
        "Explain {topic} and give three practical ways "
        "a small business can benefit from it."
    )
])

# 3. Create an output parser
parser = StrOutputParser()

# 4. Connect everything into a chain
chain = prompt | model | parser

# 5. Execute the chain
response = chain.invoke({
    "topic": "solar energy"
})

print(response)
print("\nResponse type:", type(response))
