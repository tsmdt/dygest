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


class HTML_Templates(str, Enum):
    TABS = 'tabs'
    PLAIN = 'plain'


class ExportFormats(str, Enum):
    ALL = 'all'
    JSON = 'json'
    CSV = 'csv'
    HTML = 'html'
    

@dataclass
class WriterBase:
    filename: Path = field(default=None, init=True)
    output_filepath: Path = field(default=None, init=True)
    input_text: str = field(default=None, init=True)
    chunk_size: int = field(default=None, init=True)
    chunks: dict = field(default=None, init=True)
    named_entities: list = field(default=None, init=True)
    toc: list[dict] = field(default=None, init=True)
    summary: str = field(default=None, init=True)
    keywords: list[str] = field(default=None, init=True)
    language: str = field(default=None, init=True)
    light_model: str = field(default=None, init=True)
    expert_model: str = field(default=None, init=True)
    token_count: int = field(default=None, init=True)
    sentence_offsets: dict = field(default_factory=dict, init=True)


@dataclass
class HTMLWriter(WriterBase):
    """
    Class for writing HTML output.
    """
    export_metadata: bool = field(default=None, init=True)
    html_template: HTML_Templates = field(default=HTML_Templates.TABS, init=True)
    has_speaker: bool = field(default=False, init=True)
    
    def __post_init__(self):
        # Load an HTML template
        template = self.set_html_template(type=self.html_template)

        # Inject CSS and JavaScript
        html = template.get('html')

        if 'css_path' in template:
            css_content = template.get('css_path').read_text()
            html = html.replace('{css_placeholder}', css_content)

        if 'js_path' in template:
            js_content = template.get('js_path').read_text()
            html = html.replace('{js_placeholder}', js_content)

        # Soupify the template
        self.soup = BeautifulSoup(html, 'html.parser')
        self.has_speaker = False
        
    @staticmethod    
    def set_html_template(type: HTML_Templates) -> dict:
        """
        Return HTML template dict with template, css and js filepaths.
        """
        html_templates = {
            HTML_Templates.TABS: {
                'html': templates.HTML_TABS,
                'css_path': Path('dygest/static/css/tabs.css').resolve(),
                'js_path': Path('dygest/static/tabs.js').resolve()
            },
            HTML_Templates.PLAIN: {
                'html': templates.HTML_PLAIN,
                'css_path': Path('dygest/static/css/plain.css').resolve()
            }
        }
        return html_templates.get(type)

    def _get_write_sequence(self):
        """
        Return an ordered list of callables that must be executed by
        :py:meth:`write`.  
        The concrete order depends on the selected HTML template to allow
        different layouts (e.g. *tabs* vs. *plain*).  
        Optional blocks that are not relevant for the current document
        (e.g. no summary available) are represented as ``None`` and must be
        skipped by the caller.
        """
        if self.html_template == HTML_Templates.TABS:
            return [
                self.add_page_title,
                self.add_summary if self.summary else None,
                self.add_toc if self.toc else None,
                self.add_keywords if self.keywords else None,
                self.add_metadata if self.export_metadata else None,
                lambda: self.add_main_content(content_editable=True)
            ]
        elif self.html_template == HTML_Templates.PLAIN:
            return [
                self.add_page_title,
                self.add_metadata if self.export_metadata else None,
                self.add_summary if self.summary else None,
                self.add_keywords if self.keywords else None,
                self.add_toc if self.toc else None,
                lambda: self.add_main_content(content_editable=False),
            ]
        else:
            # Fallback to the historic default order
            return [
                self.add_page_title,
                self.add_summary if self.summary else None,
                self.add_toc if self.toc else None,
                self.add_keywords if self.keywords else None,
                self.add_metadata,
                self.add_main_content
            ]
        
    def write(self):
        """
        Build and save the HTML document.  
        The order in which the individual content blocks are rendered is
        determined by :py:meth:`_get_write_sequence`, which varies by
        `self.html_template`.
        """
        for block in self._get_write_sequence():
            if block is not None:
                block()

        # Save HTML
        self.save_html()

    def format_metadata_entry(self, parent, label, value):
        span_tag = self.soup.new_tag('span')
        span_tag.string = f"{label}: {value}"
        br_tag = self.soup.new_tag('br')
        parent.append(span_tag)
        parent.append(br_tag)

    def add_page_title(self):
        """
        Add page title to HTML.
        """
        div_page_container = self.soup.find('div', class_='page-container')
        if div_page_container:
            # Add heading with filename as page title
            h1_tag = self.soup.new_tag('h1')
            h1_tag.string = self.filename
            
            # Append h1 tag
            div_page_container.append(h1_tag)
            
    def add_metadata(self):
        """
        Adds metadata to HTML.
        """
        div_page_container = self.soup.find('div', class_='page-container')
        if div_page_container:
            # Create metadata header
            div_metadata = self.soup.new_tag(
                'div',
                attrs={'class': 'metadata'}
            )
            # Add timestamp
            self.format_metadata_entry(
                parent=div_metadata,
                label='Created',
                value=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            # Create metadata content
            div_metadata_content = self.soup.new_tag(
                'div',
                attrs={'class': 'metadata-content'}
            )
            # Light LLM
            self.format_metadata_entry(
                parent=div_metadata_content, 
                label="Light LLM", 
                value=self.light_model
            )
            # Expert LLM
            self.format_metadata_entry(
                parent=div_metadata_content,
                label="Expert LLM", 
                value=self.expert_model
            )
            # Chunk size
            self.format_metadata_entry(
                parent=div_metadata_content, 
                label="Chunk size", 
                value=self.chunk_size
            )
            # Token count
            token_count = f"{self.token_count:,.0f}".replace(',', '.')
            self.format_metadata_entry(
                parent=div_metadata_content, 
                label="Tokens",
                value=token_count
            )
            # NER
            if self.named_entities:
                ner_count = Counter([x['ner_tag'] for x in self.named_entities])
                ner_counter_string = (
                    ', '.join(f"{tag} ({count})" for tag, count in ner_count.items())
                    )
                ner_string = f"{len(self.named_entities)} [{ner_counter_string}]"
                self.format_metadata_entry(
                    parent=div_metadata_content,
                    label="Entities",
                    value=ner_string
                )
                
            # Append new elements
            div_metadata.append(div_metadata_content)
            div_page_container.append(div_metadata)

    def add_controls(self):
        # Add button for saving the edited HTML
        div_document_controls = self.soup.find('div', class_='document-controls')
        if div_document_controls:
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
                div_document_controls.append(button_NER_highlighting)

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
                div_document_controls.append(button_timestamps)

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
            div_document_controls.append(button_show_html)
            
            # Save button
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
              
    def add_summary(self):
        """
        Appends summary content div to a HTML template.
        """
        div_page_container = self.soup.find('div', class_='page-container')
        if div_page_container:
            # Create summary div
            div_summary = self.soup.new_tag(
                'div',
                attrs={'class': 'summary'}
            )    
            # Add summary
            summary_tag = self.soup.new_tag('span')
            summary_tag.string = self.summary.strip()
            
            # Append new elements
            div_summary.append(summary_tag)
            div_page_container.append(div_summary)
            
    def add_keywords(self):
        """
        Appends keywords content div to a HTML template.
        """
        div_page_container = self.soup.find('div', class_='page-container')
        if div_page_container:
            # Create keywords div
            div_keywords = self.soup.new_tag(
                'div',
                attrs={'class': 'keywords'}
            )
            # Add keywords string
            keywords_tag = self.soup.new_tag('span')
            keywords_tag.string = (
                ', '.join(
                    [key if key.isupper() else key.lower() for key in self.keywords]
                    )
                )
            # Append new elements
            div_keywords.append(keywords_tag)
            div_page_container.append(div_keywords)
            
    def add_toc(self):
        """
        Appends table of contents div to a HTML template.
        """
        div_page_container = self.soup.find('div', class_='page-container')
        if div_page_container:
            # Create toc div
            div_toc_wrapper = self.soup.new_tag(
                'div',
                attrs={'class': 'toc-wrapper'}
            )
            div_toc = self.soup.new_tag(
                'div', 
                attrs={'class': 'toc'}
                )
            div_toc_content = self.soup.new_tag(
                'div',
                attrs={
                    'id': 'toc-content',
                    'style': 'display: block; text-align: left'
                }
            )
            
            ol_tag = self.soup.new_tag('ol')
            div_toc_content.append(ol_tag)
            div_toc.append(div_toc_content)
            div_toc_wrapper.append(div_toc)
            div_page_container.append(div_toc_wrapper)

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
                        link_id = f"location-{location}"
                        link_tag = self.soup.new_tag("a", href=f"#{link_id}")
                        link_tag.string = f"{topic['summary'].rstrip('.,;:?!')}"
                        topic_li.append(link_tag)
                    
                    # Append topics to unordered list
                    ul_tag.append(topic_li)
                
                li_tag.append(ul_tag)
                ol_tag.append(li_tag)
        

    def add_main_content(self, content_editable: bool = False):
        """
        Append main content blocks and event to HTML.
        """
        events = []
        
        ### 1. Collect NER events
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
        
        ### 2. Match timestamp / speaker patterns
        pattern = r'\[\d{2}:\d{2}:\d{2}\.\d{3}\] \[(?:SPEAKER_\d+|UNKNOWN)\]|\[(?:SPEAKER_\d+|UNKNOWN)\]'
        
        for match in re.finditer(pattern, self.input_text):
            start, end = match.start(), match.end()
            events.append((start, 'insert_linebreak', None))
            events.append((start, 'start_speaker', None))
            events.append((end, 'end_speaker', None))
            self.has_speaker = True  
        
        ### 3. Collect anchor tag events for TOC (now using sentence_offsets)
        inserted_locations = set()
        if self.toc:
            for item in self.toc:
                for topic in item['topics']:
                    location = topic.get('location')
                    if location and location not in inserted_locations:
                        # Check if location (e.g. "S12") in sentence_offsets
                        if location in self.sentence_offsets:
                            start = self.sentence_offsets[location]
                            anchor_id = f"location-{location}"
                            events.append((start, 'insert_anchor', anchor_id))
                            inserted_locations.add(location)
                            
                            # Do not insert a linebreak if speaker tag was found
                            if self.has_speaker:
                                continue
                            
                            events.append((start, 'insert_linebreak', None))
        
        ### 4. Insert linebreak events for each "\n"
        for match in re.finditer(r'\n{2}', self.input_text):
            events.append((match.start(), 'insert_linebreak', None))
            
                        
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
        p_tag = self.soup.new_tag('p')
        content_elements = []
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
                        "contenteditable": f'{str(content_editable).lower()}'
                    })

                # Create the inner span that will hold the text
                inner_span_tag = self.soup.new_tag(
                    "span",
                    attrs={
                        "class": "edits",
                        "tabindex": 0,
                        "contenteditable": f'{str(content_editable).lower()}'
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
                # Ensure all spans are closed so the anchor doesn't appear 
                # nested incorrectly. If necessary, close spans:
                spans_stack.clear()
                current_parent = p_tag

                # Create and insert the anchor inline
                anchor_tag = self.soup.new_tag(
                    "a",
                    id=data,
                    attrs={
                        "class": "anchor"
                    }
                )
                current_parent.append(anchor_tag)
            
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
                        "contenteditable": f'{str(content_editable).lower()}'
                    })
                
                inner_span_tag = self.soup.new_tag(
                    "span",
                    attrs={
                        "class": "edits",
                        "tabindex": 0,
                        "contenteditable": f'{str(content_editable).lower()}'
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
        
        # Write the main page content
        div_page_container = self.soup.find('div', class_='page-container')
        if div_page_container:            
            div_content = self.soup.new_tag(
                'div',
                attrs={
                    'class': 'content',
                    'contenteditable': f'{str(content_editable).lower()}'
                }
            )
            
            for element in content_elements:
                div_content.append(element)
            
            div_page_container.append(div_content)    
            
        # Transform any markdown tags to HTML
        self.transform_markdown_tags()
        
    def transform_markdown_tags(self):
        """
        Transforms markdown syntax within the main content of the HTML template
        to corresponding HTML tags.
        """
        # Locate the main content div
        main_content = self.soup.find('div', class_='content')
        if not main_content:
            print(f"""... No <div class="content"> in HTML.""")
            return

        markdown_patterns = {
            'horizontal_rule': re.compile(r'^-{3,}$', re.MULTILINE),
            'blockquotes': re.compile(r'^>\s?(.*)', re.MULTILINE),
            'headings': re.compile(r'^(#{1,6})\s*(.+)', re.MULTILINE),
            'ordered_list': re.compile(r'^(\s*)\d+\.\s+(.*)', re.MULTILINE),
            'unordered_list': re.compile(r'^(\s*)[-\*\+]\s+(.*)', re.MULTILINE),
            'images': re.compile(r'!\[([^\]]*)\]\(([^\)]+)\)'),
            'links': re.compile(r'\[([^\]]+)\]\(([^\)]+)\)'),
            'inline_code': re.compile(r'`([^`]*)`'),
            'bold': re.compile(r'(\*\*|__)(.*?)\1'),
            'italics': re.compile(r'(\*|_)(.*?)\1'),
            'strikethrough': re.compile(r'~~(.*?)~~'),
        }

        def replace_heading(match):
            hashes, text = match.groups()
            level = len(hashes)
            return f'<h{level}>{text.strip()}</h{level}>'

        def replace_ordered_list(match):
            indent, item = match.groups()
            return f'{indent}<ol><li>{item.strip()}</li></ol>'

        def replace_unordered_list(match):
            indent, item = match.groups()
            return f'{indent}<ul><li>{item.strip()}</li></ul>'

        # Convert the main_content to a string for regex operations
        content_html = str(main_content)

        # Apply horizontal rule transformation first to avoid conflicts
        content_html = markdown_patterns['horizontal_rule'].sub(
            '<hr/>', 
            content_html
            )

        # Blockquotes
        content_html = markdown_patterns['blockquotes'].sub(
            r'<blockquote>\1</blockquote>', 
            content_html
            )

        # Headings
        content_html = markdown_patterns['headings'].sub(
            replace_heading, 
            content_html
            )

        # Lists
        content_html = markdown_patterns['ordered_list'].sub(
            replace_ordered_list, 
            content_html
            )
        
        content_html = markdown_patterns['unordered_list'].sub(
            replace_unordered_list, 
            content_html
            )

        # Images
        content_html = markdown_patterns['images'].sub(
            r'<img alt="\1" src="\2" />', 
            content_html
            )

        # Links
        content_html = markdown_patterns['links'].sub(
            r'<a href="\2">\1</a>', 
            content_html
            )

        # Inline code
        content_html = markdown_patterns['inline_code'].sub(
            r'<code>\1</code>', 
            content_html
            )

        # Bold text
        content_html = markdown_patterns['bold'].sub(
            r'<strong>\2</strong>', 
            content_html
            )

        # Italics
        content_html = markdown_patterns['italics'].sub(
            r'<em>\2</em>', 
            content_html
            )

        # Strikethrough
        content_html = markdown_patterns['strikethrough'].sub(
            r'<del>\1</del>', 
            content_html
            )

        # Parse the transformed HTML with BeautifulSoup
        transformed_soup = BeautifulSoup(content_html, 'html.parser')
        
        # Clean empty <p> tags
        for p in transformed_soup.find_all('p'):
            # Check if the paragraph has neither text nor any child elements
            if not p.get_text(strip=True) and not p.find(True, recursive=False):
                p.decompose()

        # Replace the original main_content with the transformed content
        main_content.clear()
        for element in transformed_soup.contents:
            main_content.append(element)

        self.soup.prettify(formatter='html')

    def save_html(self):
        with open(self.output_filepath, 'w', encoding='utf-8') as fout:
            fout.write(str(self.soup))
        print(f"... 🌞 Saved HTML → {self.output_filepath}")


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
                'language',
                'chunk_size',
                'token_count',
                'light_model',
                'expert_model',
                'chunks',
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
                    self.language,
                    self.chunk_size,
                    self.token_count,
                    self.light_model,
                    self.expert_model,
                    self.chunks,
                    self.summary if self.summary else None,
                    self.keywords if self.keywords else None,
                    self.toc if self.toc else None
                ]
            )
        print(f"... 🌞 Saved CSV → {self.output_filepath}")


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
                'language': self.language,
                'chunk_size': self.chunk_size,
                'token_count': self.token_count,
                'light_model': self.light_model,
                'expert_model': self.expert_model,
                'chunks': self.chunks,
                'summary': self.summary if self.summary else None,
                'keywords': self.keywords if self.keywords else None,
                'toc': self.toc if self.toc else None
            }
            json.dump(json_dict, fout, indent=4)
        print(f"... 🌞 Saved JSON → {self.output_filepath}")
        

def get_writer(proc):
    """
    Instantiates and returns the appropriate writer Class based on 
    proc.export_format, including shared and format-specific parameters.
    """
    def html_specific_params(proc):
        params = {
            'export_metadata': proc.export_metadata if proc.export_metadata else False,
            'html_template': proc.html_template if proc.html_template else HTML_Templates.TABS
        }
        return params
    
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
        'chunks': proc.chunks,
        'named_entities': proc.entities if proc.add_ner else None,
        'toc': proc.toc if proc.add_toc else None,
        'summary': proc.summaries if proc.add_summaries else None,
        'keywords': proc.keywords if proc.add_keywords else None,
        'language': proc.language_ISO,
        'light_model': proc.light_model,
        'expert_model': proc.expert_model,
        'token_count': proc.token_count,
        'sentence_offsets': proc.sentence_offsets
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
