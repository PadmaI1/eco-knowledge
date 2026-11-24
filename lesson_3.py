
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 1. Define the expected output structure
class EnvironmentalAssessment(BaseModel):
    topic: str = Field(
        description="The environmental topic being assessed"
    )

    summary: str = Field(
        description="A short summary of the provided information"
    )

    benefits: list[str] = Field(
        description="Environmental benefits mentioned in the text"
    )

    risks: list[str] = Field(
        description="Environmental risks or disadvantages mentioned in the text"
    )


# 2. Initialize Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    ),
    temperature=0.2
)

# 3. Bind the output schema to the model
structured_model = model.with_structured_output(
    EnvironmentalAssessment,
    method="json_schema"
)

# 4. Give the model some source information
source_text = """
Solar energy is a renewable energy source obtained from sunlight.
Solar panels convert sunlight into electricity using photovoltaic cells.

Solar energy is clean during operation and reduces dependence on
fossil fuels such as coal and natural gas. It also reduces greenhouse
gas emissions.

However, manufacturing solar panels requires raw materials and energy.
Solar panel disposal and recycling also present challenges.
"""

# 5. Ask the model to analyze the information
result = structured_model.invoke(
    f"""
    Analyze the following environmental information.

    Use only the information provided.
    Do not invent benefits or risks.
    If a category has no supporting information, return an empty list.

    Source information:
    {source_text}
    """
)

# 6. Access the structured result
print("Topic:", result.topic)
print("Summary:", result.summary)

print("\nEnvironmental benefits:")
for benefit in result.benefits:
    print("-", benefit)

print("\nEnvironmental risks:")
for risk in result.risks:
    print("-", risk)

# 7. Convert the Pydantic object into a dictionary
result_dict = result.model_dump()

print("\nDictionary:")
print(result_dict)
