CREATE_SUMMARIES = """
Perform the following tasks on the provided text:

1. **Identify the top 2 most important topics** discussed in the text.
2. **For each identified topic**, generate a concise description using a short phrase or group of keywords that effectively captures the key points related to that topic.

**Return the results in the language of the Input Text as JSON. Do NOT add anything else and use this template without any alterations:**

[
  {
    "topic": "Topic Name",
    "summary": "A concise phrase related to the topic",
    "location": "First 5 words of the corresponding text chunk of the Input Text"
  },
  ...
]

**Input Text:**
"""


CREATE_TOC = """
Perform the following tasks on the provided list of summaries:

1. **Create a table of contents** out of all summaries by **grouping related topics under one short headline**.
2. **Never change the "location" of a summary**.
3. **

**Return the results in the language of the Input Summaries as JSON**. **Do NOT add anything else** and use this template without any alterations:

[
  {
    "headline": "Headline name",
    "topics": [
        "summary": "Provided summary",
        "location": "Provided location of the summary"
        ],
        ...
  },
  ...
]

**Input Summaries:**
"""


CLEAN_SUMMARIES = """
Perform the following tasks on the provided list of summaries:

**Identify overlapping summaries**: Analyze the summaries to find any that heavily overlapping in content or topic.
**Remove duplicates**: Remove any overlapping summaries, ensuring that each topic is represented by only one summary.
**Return a list of unique summaries**: Provide a new list containing only the unique summaries.
**Do NOT change the remaining summaries in any way**

**Return the results in the language of the Input Summaries as JSON**. **Do NOT add anything else** and use this template without any alterations:

[
  {
    "topic": "Topic Name",
    "summary": "Input summary",
    "location": "Input location of the summary"
  },
  ...
]

**Input Summaries:**
"""