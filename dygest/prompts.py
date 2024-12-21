### Function for building prompts with kwargs ###

def build_prompt(template, **kwargs):
    prompt = template
    for key, value in kwargs.items():
        placeholder = '{' + key + '}'
        prompt = prompt.replace(placeholder, value)
    return prompt


### TOC CREATION ###

GET_TOPICS =  """You will be given a text composed of multiple sentences. The sentences are numbered sequentially:
- {first_sentence} = the first sentence of the text
- {last_sentence} = the last sentence of the text

Do not include these sentence numbers in your summary. Instead, reference the location **with the sentence number: "S<number>"**.

Your tasks:
1. Identify the top 2 most important topics discussed in the text.
  - Each topic name should be a concise phrase (no more than 5 words).
2. For each topic, generate a precise sub-headline (no more than 8 words) with key details.
3. Count the sentences to give a **precise sentence number** as location for the identified topics.
4. Return results as JSON **IN {language}**. Do NOT add anything else and use this template without any alterations:
  
[
  {
    "topic": "Concise Subheading in {language}",
    "summary": "Concise sub-headline in {language}",
    "location": "S<number>"
  },
  ...
]

**Input Text:**
{chunk}
"""

CREATE_TOC = """
Perform the following tasks on the provided list of summaries and **ALWAYS match their language**:

1. **Create a Table of Contents (TOC):**
   - Group related summaries under concise, relevant headlines.
   - Each headline should encapsulate the theme of its grouped topics.

2. **Maintain Summary Locations:**
   - Do **not** alter the "location" value of any summary.

**Output Requirements:**
- **Format:** JSON
- **Language:** {language}
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
Perform the following tasks on the provided text:

1. Generate a summary **IN {language}**:
  - max. 3 sentences with the **most important topics** for the provided text.
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
- **Language:** {language}
- **Strictly adhere to the following template and return nothing else:**

{
  "summary": "Summary",
  "keywords": ["keyword1", "keyword2", ...]
}

**Text Chunk:**
{text_chunk}
"""

CREATE_SUMMARY = """
Perform the following tasks on the provided text chunk:

1. Generate a **summary of max. 3 sentences** with the **most important topics** for the provided text chunk.
2. Incorporate key details in your summary:
   - **Who**: Names of individuals involved.
   - **What**: Description of the events or actions.
   - **When**: Mentioned dates.
   - **Where**: Locations related to the events.
   - **Why**: Purpose or reason behind the events.
3. Return the results in **{language}**.

**Return ONLY the summary and add nothing else!**

**Text Chunk:**
{text_chunk}
"""

COMBINE_SUMMARIES = """
Perform the following tasks on the provided list of summaries:

1. Generate a **single summary of maximum 5 sentences** that captures the **most important topics** from all provided summaries.
2. **Remove similar summaries** to ensure the remaining summaries are **unique**.
3. Return the summary in **{language}** as a **plain string**.

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
3. Provide keywords **{language}** as the input text.
4. Limit the number of keywords to **2-5** to maintain relevance.
5. Return the keywords as a plain string, separated by commas.

Return **ONLY the comma-separated keywords** and **nothing else**.

**Text Chunk:**
{text_chunk}
"""