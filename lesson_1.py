
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# 1. Initialize the chat model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# 2. Create a reusable prompt template
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an environmental science assistant.
        Explain concepts in beginner-friendly language.
        Use factual information and avoid unsupported claims.
        """
    ),
    (
        "human",
        "Explain {topic} and describe its environmental impact."
    )
])

# 3. Supply dynamic input
prompt_value = prompt_template.invoke({
    "topic": "fast fashion"
})

# 4. Send the formatted messages to Gemini
response = model.invoke(prompt_value)

# 5. Display the response
print(response.content)
