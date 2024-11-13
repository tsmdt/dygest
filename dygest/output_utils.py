import re
from datetime import datetime
from pathlib import Path
from collections import Counter
from bs4 import BeautifulSoup

from dygest import templates
from dygest.translations import UI_TRANSLATIONS


CATEGORY_COLORS = {
    'ORG': '#f5b222',
    'PER': '#7ccfff',
    'LOC': '#1bc89a',
    'DATE': '#f4eb67',
    'MISC': '#ff8222',
}


class HTMLWriter:
    def __init__(self,
            filename: Path, 
            output_filepath: Path,
            text: str, 
            named_entities: list, 
            summaries: list[dict],
            language: str,
            llm_service: str,
            model: str,
            mode: str,
            token_count: int,
            export_metadata: bool
            ):
        self.filename = filename
        self.output_filepath = output_filepath
        self.text = text
        self.named_entities = named_entities
        self.summaries = summaries
        self.language = language
        self.llm_service = llm_service
        self.model = model
        self.mode = mode
        self.token_count = token_count
        self.export_metadata = export_metadata
        # HTML template
        self.soup = BeautifulSoup(templates.HTML_CONTENT, "html.parser")
        self.has_speaker = False

    def write_html(self):
        """
        Builds and saves a HTML page with LLM summaries and Named Entities.
        """
        # Add metadata
        self.add_metadata_to_html()

        # Add main content
        self.add_main_content()

        # Add controls
        self.add_controls()

        # Save HTML to disk
        self.save_html()

    def add_metadata_to_html(self):
        """
        Adds metadata to HTML page.
        """
        div_metadata = self.soup.find('div', class_='metadata')
        if div_metadata:
            # Add filename
            h6_tag = div_metadata.find('h6', class_='metadata-header')
            if h6_tag:
                h6_tag.string = f'{self.filename}'
            
            # Add model and LLM service
            if self.export_metadata:
                div_metadata_content = self.soup.find('div', class_='metadata-content')
                if div_metadata_content:
                    llm_service_tag = self.soup.new_tag('span')
                    llm_service_tag.string = f"Created with {self.model} ({self.llm_service.name})"
                    div_metadata_content.append(llm_service_tag)
                    
                    br_tag = self.soup.new_tag('br')
                    div_metadata_content.append(br_tag)

                    processing_timestamp_tag = self.soup.new_tag('span')
                    processing_timestamp_tag.string = f"Date: {str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}"
                    div_metadata_content.append(processing_timestamp_tag)

                    br_tag = self.soup.new_tag('br')
                    div_metadata_content.append(br_tag)

                    token_count_tag = self.soup.new_tag('span')
                    token_count_tag.string = f"Tokens: {self.token_count:,.0f}".replace(',', '.')
                    div_metadata_content.append(token_count_tag)

                    br_tag = self.soup.new_tag('br')
                    div_metadata_content.append(br_tag)

                    entity_count_tag = self.soup.new_tag('span')
                    ner_count = Counter([x['ner_tag'] for x in self.named_entities])
                    ner_string = f"{len(self.named_entities)} {list(ner_count.items())}"
                    entity_count_tag.string = f"Entities: {ner_string}"
                    div_metadata_content.append(entity_count_tag)


    def add_controls(self):
        # Append additional button controls to HTML
        div_additional_controls = self.soup.find('div', class_='additional-controls')
        
        if div_additional_controls:
            # Add NER Highlighting Button  
            if len(self.named_entities) != 0:
                button_NER_highlighting = self.soup.new_tag(
                    'button',
                    attrs={
                        "id": "toggle-highlighting",
                        "onclick": "toggleHighlighting()"
                    })
                button_NER_highlighting.string = UI_TRANSLATIONS.get(
                    'button_NER_highlighting'
                    ).get(self.language, 'en')
                div_additional_controls.append(button_NER_highlighting)

            # Add Speaker / Timestamp Button 
            if self.has_speaker:
                button_timestamps = self.soup.new_tag(
                    'button',
                    attrs={
                        "id": "toggle-timestamp",
                        "onclick": "toggleTimestamp()"
                    })
                button_timestamps.string = UI_TRANSLATIONS.get(
                    'button_timestamps'
                    ).get(self.language, 'en')
                div_additional_controls.append(button_timestamps)

            # Add button for showing HTML Code
            button_show_html = self.soup.new_tag(
                'button',
                attrs={
                    'id': 'toggle-source',
                    'onclick': 'toggleSource()'
                })
            button_show_html.string = UI_TRANSLATIONS.get(
                'button_show_HTML'
                ).get(self.language, 'en')
            div_additional_controls.append(button_show_html)

        # Add button for saving the edited HTML
        div_document_controls = self.soup.find('div', class_='document-controls')
        button_save = self.soup.new_tag(
            'button',
            attrs={
                'class': 'save',
                'onclick': 'savePage()'
            })
        button_save.string = UI_TRANSLATIONS.get(
            'button_save'
            ).get(self.language, 'en')
        div_document_controls.append(button_save)

    def add_main_content(self):
        """
        Appends main content blocks to a default HTML template.
        """
        # Add summaries heading
        div_summaries = self.soup.find('div', id='summary-content')
        if div_summaries:
            h5_tag = self.soup.new_tag(
                'h5',
                attrs={
                    'style': 'text-align: center;'
                })
            h5_tag.string = UI_TRANSLATIONS.get(
                'heading_topics'
                ).get(self.language, 'en')
            div_summaries.insert_before(h5_tag)

        # TOC
        if self.mode == 'create_toc':
            ol_tag = self.soup.find("ol")
            for item in self.summaries:
                # Create a list element for the headline
                li_tag = self.soup.new_tag("li")
                strong_tag = self.soup.new_tag("strong")
                strong_tag.string = f'{item["headline"]}'
                li_tag.append(strong_tag)
                
                # Create an unordered list for topics
                ul_tag = self.soup.new_tag("ul")

                # Append each topic to the unordered list
                for topic in item['topics']:
                    topic_li = self.soup.new_tag('li')
                    
                    # Get location to create a link to the content
                    location = topic.get('location')
                    if location:
                        link_id = f"location_{location.replace(' ', '_')}"
                        link_tag = self.soup.new_tag("a", href=f"#{link_id}")
                        link_tag.string = f"{topic['summary']}"
                        topic_li.append(link_tag)
                    
                    # Append topics to unordered list
                    ul_tag.append(topic_li)
                
                li_tag.append(ul_tag)
                ol_tag.append(li_tag)

        # Summaries
        elif self.mode == 'clean_summaries':
            ol_tag = self.soup.find("ol")
            for el in self.summaries:
                li_tag = self.soup.new_tag("li")
                
                strong_tag = self.soup.new_tag("strong")
                strong_tag.string = f'{el["topic"]}: '
                
                li_tag.append(strong_tag)
                
                # Create a link to the location in the text
                location = el.get('location')
                if location:
                    link_id = f"location_{location.replace(' ', '_')}"
                    link_tag = self.soup.new_tag("a", href=f"#{link_id}")
                    link_tag.string = f"{el['summary']}"
                    li_tag.append(link_tag)
                
                ol_tag.append(li_tag)
        else:
            print(f'... Wrong prompt template: {self.mode}')
                
        # Prepare to process the text
        events = []
        
        ### 1. Collect NER events
        for entity in self.named_entities:
            entity_start = entity['start']
            entity_end = entity['end']
            entity_text = entity['text']
            entity_ner_tag = entity['ner_tag']

            # Skip certain entities
            if entity_text in ['SPEAKER', 'UNKNOWN', '.']:
                continue

            events.append((entity_start, 'start_entity', entity_ner_tag))
            events.append((entity_end, 'end_entity', None))
        
        ### 2. Collect anchor tag events 
        # TOC
        if self.mode == 'create_toc':
            for item in self.summaries:
                for topic in item['topics']:
                    location = topic.get('location')
                    if location:
                        for match in re.finditer(re.escape(location), self.text):
                            start, _ = match.start(), match.end()
                            anchor_id = f"location_{location.replace(' ', '_')}"
                            events.append((start, 'insert_anchor', anchor_id))
        # Summaries
        else:
            for item in self.summaries:
                location = item.get('location')
                if location:
                    for match in re.finditer(re.escape(location), self.text):
                        start, _ = match.start(), match.end()
                        anchor_id = f"location_{location.replace(' ', '_')}"
                        events.append((start, 'insert_anchor', anchor_id))
        
        ### 3. Match timestamp / speaker patterns of this type: [00:00:06.743] [SPEAKER_09]
        pattern = r'\[\d{2}:\d{2}:\d{2}\.\d{3}\] \[SPEAKER_\d+\]|\[SPEAKER_\d+\]'
        for match in re.finditer(pattern, self.text):
            start, end = match.start(), match.end()
            events.append((start, 'insert_linebreak', None))
            events.append((start, 'start_speaker', None))
            events.append((end, 'end_speaker', None))
            self.has_speaker = True  
        
        # Define event order to handle conflicts
        event_order = {
            'end_entity': 0,
            'end_speaker': 1,
            'insert_anchor': 2,
            'insert_linebreak': 3,
            'start_speaker': 4,
            'start_entity': 5
        }
        events.sort(key=lambda x: (x[0], event_order[x[1]]))
        
        # Build the content with paragraphs
        content_elements = []
        p_tag = self.soup.new_tag('p')
        current_parent = p_tag
        spans_stack = []
        position = 0
        
        for event in events:
            event_position, event_type, data = event
            
            # Add text up to the event position
            if position < event_position:
                text_segment = self.text[position:event_position]
                current_parent.append(text_segment)
            position = event_position
            
            if event_type == 'start_entity':
                # Start a new span for the entity
                span_tag = self.soup.new_tag(
                    "span", 
                    attrs={
                        "class": "ner-entity",
                        "data-color": CATEGORY_COLORS.get(data, ''),
                        "alt": data,
                        "style": f"background-color: {CATEGORY_COLORS.get(data, '')};",
                        "title": data
                    })
                current_parent.append(span_tag)
                spans_stack.append(span_tag)
                current_parent = span_tag
                
            elif event_type == 'end_entity':
                # Close the current entity span
                spans_stack.pop()
                current_parent = spans_stack[-1] if spans_stack else p_tag
                
            elif event_type == 'insert_anchor':
                # Close any open spans before ending the paragraph
                spans_stack.clear()
                current_parent = p_tag  
                
                # Add the current paragraph to the content if it has any children
                if p_tag.contents:
                    content_elements.append(p_tag)
                
                # Only start a new paragraph if no 'start_speaker' events exist
                if not self.has_speaker:
                    p_tag = self.soup.new_tag('p')
                    current_parent = p_tag
                
                # Insert the anchor tag directly into the content
                anchor_tag = self.soup.new_tag(
                    "a", 
                    id=data,
                    attrs={
                        "class": "anchor"
                    })
                content_elements.append(anchor_tag)
            
            elif event_type == 'insert_linebreak':
                # Close any open spans before ending the paragraph
                spans_stack.clear()
                current_parent = p_tag
                
                # Add the current paragraph to the content if it has any children
                if p_tag.contents:
                    content_elements.append(p_tag)
                
                # Start a new paragraph
                p_tag = self.soup.new_tag('p')
                current_parent = p_tag
            
            elif event_type == 'start_speaker':
                span_tag = self.soup.new_tag(
                    "span", 
                    attrs={
                        "class": "timestamp"
                    })
                current_parent.append(span_tag)
                spans_stack.append(span_tag)
                current_parent = span_tag
            
            elif event_type == 'end_speaker':
                # Close the speaker span
                spans_stack.pop()
                current_parent = spans_stack[-1] if spans_stack else p_tag
        
        # Add any remaining text after the last event
        if position < len(self.text):
            text_segment = self.text[position:]
            current_parent.append(text_segment)
        
        # Add the last paragraph to the content if it has any children
        if p_tag.contents:
            content_elements.append(p_tag)
        
        # Replace the old paragraph with the new content
        old_p_tag = self.soup.p
        for element in content_elements:
            old_p_tag.insert_before(element)
        old_p_tag.decompose()

        # Ensure readability of HTML
        self.soup.prettify()

    
    def save_html(self):
        with open(self.output_filepath, 'w', encoding='utf-8') as fout:
            fout.write(str(self.soup))
        print(f"🌞 Saved {self.output_filepath}.")
    