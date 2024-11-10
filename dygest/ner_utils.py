import re
import copy
import time
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter
from collections import defaultdict


def load_tagger(language: str = 'en', precise: bool = False) -> Classifier:
    """
    Loads flair Classifier depending on a provided language.
    """
    flair_models = {
        'en': 'ner',
        'ar': 'ar-ner',
        'de': 'de-ner',
        'da': 'da-ner',
        'fr': 'fr-ner',
        'es': 'es-ner',
        'nl': 'nl-ner',
    }
    
    start_time = time.time()
    print('... Loading NER tagger')

    # Load Classifier
    try:
        if precise and language in flair_models:
            tagger = Classifier.load(f'{flair_models[language]}-large')
        elif language not in flair_models:
            tagger = Classifier.load('ner-large')
        else:
            tagger = Classifier.load(f'{flair_models[language]}')
    except {} as e:
        print(f'ERROR: {e}')

    end_time = time.time()
    print(f'... Finished loading NER tagger in {end_time - start_time:.2f} seconds.')

    return tagger

def get_flair_entities(text: str, tagger: Classifier) -> list:
    # Split text_chunk into sentences
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(text)

    # Predict entities
    tagger.predict(sentences)

    entities = []
    for sentence in sentences:
        for entity in sentence.get_spans():
            entities.append({
                'start': entity.start_position,
                'end': entity.end_position,
                'ner_tag': entity.tag,
                'text': entity.text
                })

    return entities

def update_entity_positions(
        entities: list[dict], 
        text_doc: str, 
        verbose: bool = False
        ) -> list[dict]:
    """
    Updates the 'start' and 'end' positions of each entity based on their
    occurrences in the document.

    Parameters:
    - entities (list of dict): The list of entity dictionaries to update.
    - doc (str): The document string where entities are located.

    Returns:
    - list of dict: The updated list of entity dictionaries.
    """
    temp_entities = copy.deepcopy(entities)

    # Group entities by their 'text' value
    text_to_entities = defaultdict(list)
    for entity in temp_entities:
        text_to_entities[entity['text']].append(entity)
    
    # Iterate through each unique 'text' and find all matches in 'doc'
    for text, entity_list in text_to_entities.items():
        # Escape the 'text' to handle any special regex characters
        escaped_text = re.escape(text)
        matches = list(re.finditer(escaped_text, text_doc))
        
        # Count the number of matches and entities
        num_matches = len(matches)
        num_entities = len(entity_list)
        
        if verbose:
            # Debugging Output
            print(
                f"Processing '{text}': {num_matches} matches found, " 
                f"{num_entities} entities to update."
                )
            
            # Check for mismatches in counts
            if num_matches < num_entities:
                print(
                    f"Warning: Not enough matches for '{text}'. "
                    f"{num_entities - num_matches} entities will not be updated."
                    )
            elif num_matches > num_entities:
                print(
                    f"Warning: More matches than entities for '{text}'. "
                    f"{num_matches - num_entities} extra matches will be ignored."
                    )
            
        # Assign match positions to entities
        for i, entity in enumerate(entity_list):
            if i < num_matches:
                match = matches[i]
                entity['start'] = match.start()
                entity['end'] = match.end()
                if verbose:
                    print(
                        f"Updated entity {i+1}: start={match.start()}, "
                        f"end={match.end()}"
                        )
            else:
                # Handle entities without corresponding matches
                if verbose:
                    print(
                        f"Entity {i+1} with text '{text}' has no corresponding "
                        f"match in 'doc'."
                        )
    
    return temp_entities
