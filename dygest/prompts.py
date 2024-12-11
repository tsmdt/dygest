# 3. **Make sure to capture names, organisations, (historical) events, places or dates** if they are relevant to the topic.

# CREATE_SUMMARIES = """
# Perform the following tasks on the provided text:

# 1. **Identify the top 2 most important topics** discussed in the text.
# 2. **For each identified topic**, generate a concise summary using a short phrase that effectively captures the key points related to that topic.
# 3. **The summary should answer these questions: Who did what when and why?**

# **Return the results in the language of the Input Text as JSON. Do NOT add anything else and use this template without any alterations:**

# [
#   {
#     "topic": "Topic Name",
#     "summary": "A concise phrase related to the topic **in the language of the Input Text**",
#     "location": "First 5 words of the corresponding text chunk of the Input Text"
#   },
#   ...
# ]

# **Input Text:**
# """


# CREATE_SUMMARIES = """
# Perform the following tasks on the provided text:

# 1. **Identify the top 2 most important topics** discussed in the text.
# 2. **For each identified topic**, generate a concise summary that includes key details as a short phrase:
#    - **Who**: Names of individuals involved.
#    - **What**: Description of the events or actions.
#    - **When**: Mentioned dates.
#    - **Where**: Locations related to the events.
#    - **Why**: Purpose or reason behind the events.
# 3. **Ensure the summary answers these questions**: Who did what, when, where, and why.

# **Return the results in the language of the Input Text as JSON. Do NOT add anything else and use this template without any alterations:**

# [
#   {
#     "topic": "Topic Name",
#     "summary": "A concise phrase related to the topic **in the language of the Input Text**",
#     "location": "First 5 words of the corresponding text chunk of the Input Text"
#   },
#   ...
# ]

# **Input Text:**
# """


CREATE_SUMMARIES = """
Perform the following tasks on the provided text:

1. **Identify the top 2 most important topics** discussed in the text. 
   - **Each topic name should be a concise, headline-style phrase** (e.g., no more than 5 words) that effectively captures the essence of the topic.

2. **For each identified topic**, generate a concise summary that includes key details as a short phrase:
   - **Who**: Names of individuals involved.
   - **What**: Description of the events or actions.
   - **When**: Mentioned dates.
   - **Where**: Locations related to the events.
   - **Why**: Purpose or reason behind the events.

**Return the results in the language of the Input Text as JSON. Do NOT add anything else and use this template without any alterations:**

[
  {
    "topic": "Concise Subheading",
    "summary": "A concise phrase related to the topic **in the language of the Input Text**",
    "location": "First 5 words of the corresponding text chunk of the Input Text"
  },
  ...
]

**Input Text:**
"""

CREATE_TOC = """
Perform the following tasks on the provided list of summaries:

1. **Create a Table of Contents (TOC):**
   - Group related summaries under concise, relevant headlines.
   - Each headline should encapsulate the theme of its grouped topics.

2. **Maintain Summary Locations:**
   - Do **not** alter the "location" value of any summary.

**Output Requirements:**
- **Format:** JSON
- **Do NOT change the language of the input summaries in any way!**
- **Strictly adhere to the following template and return nothing else:**

[
  {
    "headline": "Headline Name",
    "topics": [
      {
        "summary": "summary",
        "location": "location"
      }
    ]
  },
]

**Input Summaries:**
"""


# CREATE_TOC = """
# Perform the following tasks on the provided list of summaries:

# 1. **Create a Table of Contents (TOC):**
#    - Group related summaries under concise, relevant headlines.
#    - Each headline should encapsulate the theme of its grouped topics.

# 2. **Maintain Summary Locations:**
#    - Do **not** alter the "location" value of any summary.

# **Output Requirements:**
# - **Format:** JSON
# - **Do NOT change the language of the input summaries in any way!**
# - **Strictly adhere to the following template and return nothing else:**

# [
#   {
#     "headline": "Headline Name",
#     "topics": [
#       {
#         "summary": "Provided summary",
#         "location": "Provided location of the summary"
#       },
#       {
#         "summary": "Another summary",
#         "location": "Another location"
#       }
#       // Additional topics...
#     ]
#   },
#   // Additional headlines...
# ]

# **Input Summaries:**
# """


CREATE_TLDR = """
Perform the following tasks on the provided text chunk:

1. **Generate a summary of max. 3 sentences with the most important topics** for the provided text chunk.
2. **Capture key details in those summaries**:
   - **Who**: Names of individuals involved.
   - **What**: Description of the events or actions.
   - **When**: Mentioned dates.
   - **Where**: Locations related to the events.
   - **Why**: Purpose or reason behind the events.
3. **Return the results in the language of the Input Text as a plain string.**

**Return ONLY the summary and add nothing else!**

**Text Chunk:**
"""


COMBINE_TLDRS = """
Perform the following tasks on the provided list of summaries:

1. **Generate a single summary of maximum 5 sentences that captures the most important topics from all provided summaries**.
2. **Remove similar summaries to ensure the remaining summaries are unique**.
3. **Return the result in the same language as the input summaries as a plain string**.

Return **ONLY** the single summary.

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