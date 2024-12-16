import re
import csv
import json
from enum import Enum
from dataclasses import dataclass, field
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


class ExportFormats(str, Enum):
    JSON = 'json'
    CSV = 'csv'
    HTML = 'html'


@dataclass
class WriterBase:
    filename: Path
    output_filepath: Path
    input_text: str
    chunk_size: int
    named_entities: list
    toc: list[dict]
    summary: str
    keywords: list[str]
    language: str
    light_model: str
    expert_model: str
    token_count: int


@dataclass
class HTMLWriter(WriterBase):
    """
    Class for writing HTML output.
    """
    export_metadata: bool = field(default=None, init=True)

    # HTML template
    has_speaker: bool = field(default=False, init=True)
    soup = BeautifulSoup(templates.HTML_CONTENT, "html.parser")
    has_speaker = False

    def write(self):
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

    def format_metadata_entry(self, parent, label, value):
        span_tag = self.soup.new_tag('span')
        span_tag.string = f"{label}: {value}"
        br_tag = self.soup.new_tag('br')
        parent.append(span_tag)
        parent.append(br_tag)

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
            
            # Add and format metadata
            if self.export_metadata:
                div_metadata_content = self.soup.find('div', class_='metadata-content')
                if div_metadata_content:
                    # Light LLM
                    self.format_metadata_entry(
                        parent=div_metadata_content, 
                        label="Light LLM (light_model)", 
                        value=self.light_model
                    )
                    # Expert LLM
                    self.format_metadata_entry(
                        parent=div_metadata_content,
                        label="Expert LLM (expert_model)", 
                        value=self.expert_model
                    )
                    # Chunk size
                    self.format_metadata_entry(
                        parent=div_metadata_content, 
                        label="Chunk size", 
                        value=self.chunk_size
                    )
                    # Datetime
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.format_metadata_entry(
                        parent=div_metadata_content,
                        label="Date",
                        value=current_time
                    )
                    # Token count
                    token_count = f"{self.token_count:,.0f}".replace(',', '.')
                    self.format_metadata_entry(
                        parent=div_metadata_content, 
                        label="Tokens",
                        value=token_count
                    )

                    if self.named_entities:
                        ner_count = Counter([x['ner_tag'] for x in self.named_entities])
                        ner_string = f"{len(self.named_entities)} {list(ner_count.items())}"
                        self.format_metadata_entry(
                            parent=div_metadata_content,
                            label="Entities",
                            value=ner_string
                        )

    def add_controls(self):
        # Append additional button controls to HTML
        div_additional_controls = self.soup.find('div', class_='additional-controls')
        
        if div_additional_controls:
            # Add NER Highlighting Button  
            if self.named_entities:
                button_NER_highlighting = self.soup.new_tag(
                    'button',
                    attrs={
                        "id": "toggle-highlighting",
                        "onclick": "toggleHighlighting()"
                    })
                button_NER_highlighting.string = get_translation(
                    key='button_NER_highlighting',
                    language_code=self.language
                )
                div_additional_controls.append(button_NER_highlighting)

            # Add Speaker / Timestamp Button 
            if self.has_speaker:
                button_timestamps = self.soup.new_tag(
                    'button',
                    attrs={
                        "id": "toggle-timestamp",
                        "onclick": "toggleTimestamp()"
                    })
                button_timestamps.string = get_translation(
                    key='button_timestamps',
                    language_code=self.language
                )
                div_additional_controls.append(button_timestamps)

            # Add button for showing HTML Code
            button_show_html = self.soup.new_tag(
                'button',
                attrs={
                    'id': 'toggle-source',
                    'onclick': 'toggleSource()'
                })
            button_show_html.string = get_translation(
                key='button_show_HTML',
                language_code=self.language
                )
            div_additional_controls.append(button_show_html)

        # Add button for saving the edited HTML
        div_document_controls = self.soup.find('div', class_='document-controls')
        button_save = self.soup.new_tag(
            'button',
            attrs={
                'class': 'save',
                'onclick': 'savePage()'
            })
        button_save.string = get_translation(
                key='button_save',
                language_code=self.language
            )
        div_document_controls.append(button_save)

    def add_main_content(self):
        """
        Appends main content blocks to a default HTML template.
        """
        # Add summary / TL;DR
        if self.summary:
            div_tldr = self.soup.find('div', class_='tldr')
            if div_tldr:
                # Create new div element
                div_summary = self.soup.new_tag(
                    'div',
                    attrs={'class': 'tldr-content'}
                )

                # Add title tag
                title_tag = self.soup.new_tag('span')
                title_tag.string = get_translation(
                    key="summary",
                    language_code=self.language
                )       

                # Add summary
                summary_tag = self.soup.new_tag('span')
                summary_tag.string = self.summary.strip()

                # Append new tags
                div_summary.append(title_tag)
                div_summary.append(summary_tag)
                div_tldr.append(div_summary)

        # Add keywords
        if self.keywords:
            div_tldr = self.soup.find('div', class_='tldr')
            if div_tldr:
                # Create new div element
                div_keywords = self.soup.new_tag(
                    'div',
                    attrs={'class': 'tldr-keywords'}
                )

                # Add title tag
                title_tag = self.soup.new_tag('span')
                title_tag.string = get_translation(
                    key="keywords",
                    language_code=self.language
                )       

                # Add keywords string
                keywords_tag = self.soup.new_tag('span')
                keywords_tag.string = (
                    ', '.join(
                        [key if key.isupper() else key.title() for key in self.keywords]
                        )
                    )

                # Append new tags
                div_keywords.append(title_tag)
                div_keywords.append(keywords_tag)
                div_tldr.append(div_keywords)

        
        # Add TOC heading
        if self.toc:
            div_additional_controls = self.soup.find('div', class_='additional-controls')
            div_summaries = self.soup.new_tag('div', attrs={'class': 'summaries'})
            
            div_summaries_content = self.soup.new_tag(
                'div',
                attrs={
                    'id': 'summary-content',
                    'style': 'display: block; text-align: left'
                }
            )
            
            h5_tag = self.soup.new_tag(
                'h5',
                attrs={
                    'style': 'text-align: center;'
                }
            )
            h5_tag.string = get_translation(
                key='heading_topics',
                language_code=self.language
            )
            div_summaries_content.append(h5_tag)
            
            ol_tag = self.soup.new_tag('ol')
            div_summaries_content.append(ol_tag)
            div_summaries.append(div_summaries_content)
            div_additional_controls.insert_after(div_summaries)

            # Add TOC topics and links
            ol_tag = self.soup.find("ol")
            for item in self.toc:
                
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
                        link_tag.string = f"{topic['summary'].rstrip('.,;:?!')}"
                        topic_li.append(link_tag)
                    
                    # Append topics to unordered list
                    ul_tag.append(topic_li)
                
                li_tag.append(ul_tag)
                ol_tag.append(li_tag)


        ### Content Processing ###     
        
        events = []
        
        ### 1. Collect NER events ###
        
        if self.named_entities:
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
        
        ### 2. Collect anchor tag events for TOC ###
        
        inserted_locations = set()
        if self.toc:
            for item in self.toc:
                for topic in item['topics']:
                    location = topic.get('location')
                    if location and location not in inserted_locations:
                        for match in re.finditer(re.escape(location), self.input_text):
                            start, _ = match.start(), match.end()
                            anchor_id = f"location_{location.replace(' ', '_')}"
                            events.append((start, 'insert_anchor', anchor_id))
                            inserted_locations.add(location)
                            break
        
        ### 3. Match timestamp / speaker patterns of this type: [00:00:06.743] [SPEAKER_09] ###
        
        pattern = r'\[\d{2}:\d{2}:\d{2}\.\d{3}\] \[(?:SPEAKER_\d+|UNKNOWN)\]|\[(?:SPEAKER_\d+|UNKNOWN)\]'
        
        for match in re.finditer(pattern, self.input_text):
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
                text_segment = self.input_text[position:event_position]
                current_parent.append(text_segment)
            position = event_position
            
            if event_type == 'start_entity':
                # Save the current parent to the stack
                spans_stack.append(current_parent)

                # Create the outer span for the entity
                outer_span_tag = self.soup.new_tag(
                    "span",
                    attrs={
                        "class": "ner-entity",
                        "data-color": CATEGORY_COLORS.get(data, ''),
                        "style": f"background-color: {CATEGORY_COLORS.get(data, '')};",
                        "title": data,
                        "contenteditable": False
                    })

                # Create the inner span that will hold the text
                inner_span_tag = self.soup.new_tag(
                    "span",
                    attrs={
                        "class": "edits",
                        "tabindex": 0,
                        "contenteditable": True
                    })

                outer_span_tag.append(inner_span_tag)
                current_parent.append(outer_span_tag)

                # Set the current parent to the inner span
                current_parent = inner_span_tag
                
            elif event_type == 'end_entity':
                # Close the current entity span
                if spans_stack:
                    spans_stack.pop()
                    current_parent = spans_stack[-1] if spans_stack else p_tag
                else:
                    print(f'span_stack pop Error for event: {event}')
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
                spans_stack.append(current_parent)
                
                outer_span_tag = self.soup.new_tag(
                    "span", 
                    attrs={
                        "class": "timestamp",
                        "contenteditable": False
                    })
                
                inner_span_tag = self.soup.new_tag(
                    "span",
                    attrs={
                        "class": "edits",
                        "tabindex": 0,
                        "contenteditable": True
                    })
                
                outer_span_tag.append(inner_span_tag)
                current_parent.append(outer_span_tag)
                current_parent = inner_span_tag
            
            elif event_type == 'end_speaker':
                # Close the speaker span
                spans_stack.pop()
                current_parent = spans_stack[-1] if spans_stack else p_tag
        
        # Add any remaining text after the last event
        if position < len(self.input_text):
            text_segment = self.input_text[position:]
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
        self.soup.prettify(formatter='html')

    def save_html(self):
        with open(self.output_filepath, 'w', encoding='utf-8') as fout:
            fout.write(str(self.soup))
        print(f"... 🌞 Saved {self.output_filepath}.")


@dataclass
class CSVWriter(WriterBase):
    """
    Class for writing CSV output.
    """
    def write(self):
        with open(self.output_filepath, 'w', encoding='utf-8') as fout:
            csv_header = [
                'filename',
                'output_filepath',
                'input_text',
                'language',
                'chunk_size',
                'token_count',
                'light_model',
                'expert_model',
                'summary' if self.summary else None,
                'keywords' if self.keywords else None,
                'toc' if self.toc else None
            ]
            writer = csv.writer(fout)
            writer.writerow(csv_header)
            writer.writerow(
                [
                    self.filename,
                    self.output_filepath,
                    self.input_text,
                    self.language,
                    self.chunk_size,
                    self.token_count,
                    self.light_model,
                    self.expert_model,
                    self.summary if self.summary else None,
                    self.keywords if self.keywords else None,
                    self.toc if self.toc else None
                ]
            )
        print(f"... 🌞 Saved {self.output_filepath}.")


@dataclass
class JSONWriter(WriterBase):
    """
    Class for writing JSON output.
    """
    def write(self):
        with open(str(self.output_filepath), 'w', encoding='utf-8') as fout:
            json_dict = {
                'filename': self.filename,
                'output_filepath': str(self.output_filepath),
                'input_text': self.input_text,
                'language': self.language,
                'chunk_size': self.chunk_size,
                'token_count': self.token_count,
                'light_model': self.light_model,
                'expert_model': self.expert_model,
                'summary': self.summary if self.summary else None,
                'keywords': self.keywords if self.keywords else None,
                'toc': self.toc if self.toc else None
            }
            json.dump(json_dict, fout, indent=4)
        print(f"... 🌞 Saved {self.output_filepath}.")


def get_writer(proc):
    """
    Instantiates and returns the appropriate writer Class based on 
    proc.export_format, including shared and format-specific parameters.
    """
    def html_specific_params(proc):
        return {'export_metadata': proc.export_metadata} if proc.export_metadata else {}
    
    def csv_specific_params(proc):
        return {}
    
    def json_specific_params(proc):
        return {}
    
    # Map ExportFormats to (WriterClass, specific_params_function)
    writer_mapping = {
        ExportFormats.HTML: (HTMLWriter, html_specific_params),
        ExportFormats.CSV: (CSVWriter, csv_specific_params),
        ExportFormats.JSON: (JSONWriter, json_specific_params)
    }

    # Retrieve writer_class and specific params function based on export_format
    writer_entry = writer_mapping.get(proc.export_format)
    if not writer_entry:
        raise ValueError(f"... Unknown export format: {proc.export_format}")

    writer_class, specific_params_func = writer_entry

    # Shared params for all writer classes
    shared_params = {
        'filename': proc.filename,
        'input_text': proc.text,
        'chunk_size': proc.chunk_size,
        'named_entities': proc.entities if proc.add_ner else None,
        'toc': proc.toc if proc.add_toc else None,
        'summary': proc.summaries if proc.add_summaries else None,
        'keywords': proc.keywords if proc.add_keywords else None,
        'language': proc.language if proc.add_ner else None,
        'light_model': proc.light_model,
        'expert_model': proc.expert_model,
        'token_count': proc.token_count
    }

    # Get file suffix based on export_format
    suffix = proc.export_format.value.lower()
    output_filepath = proc.output_filepath.with_suffix(f'.{suffix}')
    shared_params['output_filepath'] = output_filepath

    # Get specific params and merge them
    specific_params = specific_params_func(proc)
    specific_params = {k: v for k, v in specific_params.items() if v is not None}
    writer_params = {**shared_params, **specific_params}

    # Return the writer
    return writer_class(**writer_params)

def get_translation(key, language_code, default_language='en'):
    translations = UI_TRANSLATIONS.get(key, {})
    return translations.get(
        language_code, translations.get(default_language, '')
        )