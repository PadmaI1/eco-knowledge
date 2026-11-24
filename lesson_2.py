
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

load_dotenv()

# Initialize the model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# Create a prompt with a history placeholder
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an environmental science assistant.
        Explain concepts clearly and accurately.
        Use previous conversation context when answering follow-up questions.
        """
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Store conversation history
history = []

while True:
    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    # Format the prompt using history and latest question
    prompt_value = prompt_template.invoke({
        "history": history,
        "question": question
    })

    # Generate response
    response = model.invoke(prompt_value)

    print("\nAssistant:", response.content)

    # Save both sides of the conversation
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=response.content))
