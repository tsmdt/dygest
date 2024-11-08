import time
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter


def load_tagger(language: str = 'en', precise: bool = False) -> Classifier:
    flair_models = {
        'en': 'ner',
        'ar': 'ar-ner',
        'de': 'de-ner',
        'da': 'da-ner',
        'fr': 'fr-ner',
        'es': 'es-ner',
        'nl': 'nl-ner',
    }
    
    # Load NER tagger
    start_time = time.time()
    print('... Loading NER tagger')

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

def get_flair_entities(text_chunk: str, tagger: Classifier) -> list:
    # Split text_chunk into sentences
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(text_chunk)

    # Predict entities
    tagger.predict(sentences)
    
    entities = []
    for sentence in sentences:
        for entity in sentence.get_spans('ner'):
            flair_label = entity.get_label('ner').value
            if flair_label:
                entities.append({
                    'entity': entity.text,
                    'category': flair_label
                })
    return entities

def remove_duplicate_entities(entities: list) -> list:
    seen_entities = set()
    all_unique_entities = []
    
    for item in entities:
        entity = item['entity']
        if entity not in seen_entities:
            all_unique_entities.append(item)
            seen_entities.add(entity)
    
    return all_unique_entities
