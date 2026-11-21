def calculate_retrieval_metrics(
    retrieved_sources,
    relevant_sources
):
    # Calculate precision and recall
    relevant_docs_retrieved = [source for source in retrieved_sources if source in relevant_sources]
    recall = len(relevant_docs_retrieved) / len(relevant_sources) if relevant_sources else 0
    precision = len(relevant_docs_retrieved) / len(retrieved_sources) if retrieved_sources else 0

    return precision, recall

retrieved_sources = [
    "solar_energy.txt",
    "waste_management.txt",
    "climate_change.txt"
]

relevant_sources = [
    "solar_energy.txt",
    "climate_change.txt"
]

precision, recall = calculate_retrieval_metrics(retrieved_sources, relevant_sources)
print(f"Precision: {precision}, Recall: {recall}")