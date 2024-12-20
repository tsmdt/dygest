### TOC CREATION ###

GET_TOPICS =  """You will be given a text composed of multiple sentences. The sentences are numbered sequentially:
- {first_sentence} = the first sentence of the text
- {last_sentence} = the last sentence of the text

Do not include these sentence numbers in your summary. Instead, reference the location **with the sentence number: "S<number>"**.

Your tasks:
1. Identify the most important topic discussed in the text.
  - The topic name should be a concise phrase (no more than 5 words).
2. Generate a precise sub-headline (no more than 8 words) with key details.
3. Return results as JSON **IN THE LANGUAGE OF THE INPUT TEXT**. Do NOT add anything else and use this template without any alterations:
  
[
  {
    "topic": "Concise Subheading in the language of the input text",
    "summary": "Concise sub-headline in the language of the input text",
    "location": "S<number>"
  }
]

**Input Text:**
{chunk}
"""

# GET_TOPICS =  """You will be given a text composed of multiple sentences. The sentences are numbered sequentially:
# - {first_sentence} = the first sentence of the text
# - {last_sentence} = the last sentence of the text

# Do not include these sentence numbers in your summary. Instead, reference the location **with the sentence number: "S<number>"**.

# Your tasks:
# 1. Identify the top 2 most important topics discussed in the text.
#   - Each topic name should be a concise phrase (no more than 5 words).
# 2. For each topic, generate a precise sub-headline (no more than 8 words) with key details.
# 3. Return results as JSON **IN THE LANGUAGE OF THE INPUT TEXT**. Do NOT add anything else and use this template without any alterations:
  
# [
#   {
    # "topic": "Concise Subheading in the language of the input text",
    # "summary": "Concise sub-headline in the language of the input text",
    # "location": "S<number>"
#   },
#   ...
# ]

# **Input Text:**
# {chunk}
# """

# GET_TOPICS =  """You will be given a text composed of multiple sentences. The sentences are numbered sequentially:
# - {first_sentence} = the first sentence of the text
# - {last_sentence} = the last sentence of the text

# Do not include these sentence numbers in your summary. **Reference a location with the sentence number like this: "S<number>".**

# Your tasks:
# 1. Identify the top 2 most important topics discussed in the text.
#   - Each topic name should be a concise phrase (no more than 5 words).
# 2. For each topic, generate a precise sub-headline (no more than 8 words) with key details.
# 3. Return results as JSON **IN THE LANGUAGE OF THE INPUT TEXT**. Do NOT add anything else and use this template without any alterations:
  
# [
#   {
#     "topic": "Concise Subheading",
#     "summary": "Concise sub-headline in input language",
#     "location": "S<number>"
#   },
#   ...
# ]

# **Input Text:**
# {chunk}
# """



# GET_TOPICS = """
# Perform the following tasks on the provided text and **ALWAYS match its language**:

# 1. **Identify the top 2 most important topics** discussed in the text. 
#    - **Each topic name should be a concise, headline-style phrase** (e.g., no more than 5 words) that effectively captures the essence of the topic.

# 2. **For each identified topic**, generate a precise, sub-headline (no more than 8 words) that includes key details:
#    - Names of individuals involved.
#    - Description of the events or actions.
#    - Mentioned dates.
#    - Locations related to the events.
#    - Purpose or reason behind the events.

# Return the results as JSON **IN THE LANGUAGE OF THE INPUT TEXT**. Do NOT add anything else and use this template without any alterations:

# [
#   {
#     "topic": "Concise Subheading",
#     "summary": "A concise sub-headline-style summary related to the topic **in the language of the Input Text**",
#     "location": "First 5 words of the corresponding text chunk of the Input Text"
#   },
#   ...
# ]

# **Input Text:**
# """

CREATE_TOC = """
Perform the following tasks on the provided list of summaries and **ALWAYS match their language**:

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
{toc_parts}
"""

### SUMMARY CREATION ###

CREATE_SUMMARY_AND_KEYWORDS = """
Perform the following tasks on the provided text chunk and **ALWAYS match its language**:

1. Generate a summary of max. 3 sentences with the **most important topics** for the provided text chunk.
2. Incorporate key details in your summary:
  - **Who**: Names of individuals involved.
  - **What**: Description of the events or actions.
  - **When**: Mentioned dates.
  - **Where**: Locations related to the events.
  - **Why**: Purpose or reason behind the events.
3. Generate a **set of broad, high-level keywords** that capture the main topics of the provided text chunk.
  - Focus on **overarching themes and main topics** rather than specific details.
  - Ensure keywords are **accurate** and **informative**.
  - Limit the number of keywords to **2-5** to maintain relevance.

**Output Requirements:**
- **Format:** JSON
- **Do NOT change the language of the input text in any way!**
- **Strictly adhere to the following template and return nothing else:**


{
  "summary": "Summary in the language of the input text",
  "keywords": ["keyword1", "keyword2", ...]
}

**Text Chunk:**
{text_chunk}
"""

CREATE_SUMMARY = """
Perform the following tasks on the provided text chunk and **ALWAYS match its language**:

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
{text_chunk}
"""

COMBINE_SUMMARIES = """
Perform the following tasks on the provided list of summaries and **ALWAYS match their language**:

1. **Generate a single summary of maximum 5 sentences that captures the most important topics from all provided summaries**.
2. **Remove similar summaries to ensure the remaining summaries are unique**.
3. **Return the result in the same language as the input summaries as a plain string**.

Return **ONLY** the single summary.

**Input Summaries:**
{summaries}
"""

### KEYWORD CREATION ###

CREATE_KEYWORDS = """
Analyze the following text chunk and generate a set of broad, high-level keywords that capture the main topics and themes of the content.

**Requirements:**
1. **Focus on overarching themes and main topics** rather than specific details.
2. Ensure keywords are **accurate** and **informative**.
3. Provide keywords **IN THE SAME LANGUAGE** as the input text.
4. Limit the number of keywords to **2-5** to maintain relevance.
5. Return the keywords as a plain string, separated by commas.

Return **ONLY the comma-separated keywords** and **nothing else**.

**Text Chunk:**
{text_chunk}
"""

### Functions ####

def build_prompt_for_topics(first_sentence, last_sentence, chunk):
    prompt = GET_TOPICS
    prompt = prompt.replace('{first_sentence}', first_sentence)
    prompt = prompt.replace('{last_sentence}', last_sentence)
    prompt = prompt.replace('{chunk}', chunk)
    return prompt
  
def build_prompt_for_toc(toc_parts):
    prompt = CREATE_TOC
    prompt = prompt.replace('{toc_parts}', toc_parts)
    return prompt
  
def build_prompt_for_summary(text_chunk):
    prompt = CREATE_SUMMARY
    prompt = prompt.replace('{text_chunk}', text_chunk)
    return prompt
  
def build_prompt_for_summary_and_keywords(text_chunk):
    prompt = CREATE_SUMMARY_AND_KEYWORDS
    prompt = prompt.replace('{text_chunk}', text_chunk)
    return prompt
  
def build_prompt_for_combined_summaries(summaries):
    prompt = COMBINE_SUMMARIES
    prompt = prompt.replace('{summaries}', summaries)
    return prompt
  
def build_prompt_for_keywords(text_chunk):
    prompt = CREATE_KEYWORDS
    prompt = prompt.replace('{text_chunk}', text_chunk)
    return prompt