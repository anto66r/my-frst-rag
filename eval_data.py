# Ground truth test set for WP786.pdf
# Each entry has:
#   question      - what we ask
#   answer        - the correct answer (used by LLM-as-judge)
#   keywords      - words that MUST appear in the retrieved chunks for retrieval to count as a hit

TEST_SET = [
    {
        "question": "By how much did Ireland's average temperature increase between 1890 and 2013?",
        "answer": "Ireland's average temperature increased by approximately 0.9°C between 1890 and 2013.",
        "keywords": ["0.9", "1890", "2013"],
    },
    {
        "question": "What is Ireland's target for reducing greenhouse gas emissions by 2030?",
        "answer": "Ireland aims to achieve a 51% reduction in greenhouse gas emissions by 2030, relative to 2018 levels.",
        "keywords": ["51%", "2030"],
    },
    {
        "question": "What percentage of Ireland's population lives within 10km of the coast?",
        "answer": "Around 60% of Ireland's population lives within 10km of the coast.",
        "keywords": ["60%", "coast"],
    },
    {
        "question": "What legislation sets out Ireland's pathway to a low-carbon economy by 2050?",
        "answer": "The Climate Action and Low Carbon Development (Amendment) Act 2021.",
        "keywords": ["Climate Action", "Low Carbon Development", "2021"],
    },
    {
        "question": "By how much did annual precipitation in Ireland increase between 1989 and 2018?",
        "answer": "Annual precipitation increased by 6% relative to the 30-year period 1961 to 1990.",
        "keywords": ["6%", "precipitation"],
    },
]
