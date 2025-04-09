### JSON SCHEMAS ###

GET_TOPICS_JSON = {
    "name": "get_topics",
    "schema": {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "topic": {
                        "type": "string",
                        "description": "Concise Subheading in {language}"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Concise sub-headline in {language}"
                    },
                    "location": {
                        "type": "string",
                        "description": "S<number>"
                    }
                },
                "required": ["topic", "summary", "location"]
            }
        }
    }
}

CREATE_TOC_JSON = {
    "name": "create_toc",
    "schema": {
        "type": "object",
        "properties": {
            "headline": {
                "type": "string",
                "description": "The main headline for the table of contents"
            },
            "topics": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A brief summary of the topic"
                    },
                    "location": {
                        "type": "string",
                        "description": "location"
                    }
                },
                "required": ["summary", "location"]
            }
        },
        "required": ["headline", "topics"]
    }
}

{
  "summary": "Summary",
  "keywords": ["keyword1", "keyword2", ...]
}


CREATE_SUMMARY_AND_KEYWORDS_JSON = {
    "name": "create_summary_and_keywords",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "Summary",
                "description": "The main headline for the table of contents"
            },
            "keywords": {
                "type": "array",
                "items": {
                    "keyword": {
                        "type": "string",
                        "description": "A descriptive keyword"
                    },

                },
                "required": ["summary", "keywords"]
            }
        },
        "required": ["summary", "keywords"]
    }
}