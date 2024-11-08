import re
from bs4 import BeautifulSoup
from src import templates


# HTML
def create_html(filename, text, named_entities, summaries):
    category_colors = {
        'ORG': '#f5b222',
        'PER': '#7ccfff',
        'LOC': '#1bc89a',
        'DATE': '#f4eb67',
        'MISC': '#ff8222',
    }
    
    soup = BeautifulSoup(templates.HTML_CONTENT, "html.parser")
    
    # Add filename
    div_content = soup.find('div', class_='two-thirds column')
    if div_content:
        h6_tag = div_content.find('h6')
        if h6_tag:
            h6_tag.string = filename
    
    # Add summaries to the header
    ul_tag = soup.find("ol")
    for el in summaries:
        li_tag = soup.new_tag("li")
        
        strong_tag = soup.new_tag("strong")
        strong_tag.string = f'{el["topic"]}: '
        
        li_tag.append(strong_tag)
        
        # Create a link to the location in the text
        location = el.get('location')
        if location:
            link_id = f"location_{location.replace(' ', '_')}"
            link_tag = soup.new_tag("a", href=f"#{link_id}")
            link_tag.string = f"{el['summary']}"
            li_tag.append(link_tag)
        
        ul_tag.append(li_tag)
            
    # Prepare to process the text
    events = []
    has_start_speaker = False  # Flag to indicate if any start_speaker events exist
    
    # Collect named entity events
    for el in named_entities:
        entity = el['entity']
        category = el['category']
        
        # Skip certain entities
        if entity in ['SPEAKER', 'UNKNOWN']:
            continue
        
        for match in re.finditer(re.escape(entity), text):
            start, end = match.start(), match.end()
            events.append((start, 'start_entity', category))
            events.append((end, 'end_entity', None))
    
    # Collect anchor tag events
    for el in summaries:
        location = el.get('location')
        if location:
            for match in re.finditer(re.escape(location), text):
                start, _ = match.start(), match.end()
                anchor_id = f"location_{location.replace(' ', '_')}"
                events.append((start, 'insert_anchor', anchor_id))
    
    # Match timestamp / speaker patterns of this type: [00:00:06.743] [SPEAKER_09]
    pattern = r'\[\d{2}:\d{2}:\d{2}\.\d{3}\] \[.*\]'
    for match in re.finditer(pattern, text):
        start, end = match.start(), match.end()
        events.append((start, 'insert_linebreak', None))
        events.append((start, 'start_speaker', None))
        events.append((end, 'end_speaker', None))
        has_start_speaker = True  
    
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
    p_tag = soup.new_tag('p')
    current_parent = p_tag
    spans_stack = []
    position = 0
    
    for event in events:
        event_position, event_type, data = event
        
        # Add text up to the event position
        if position < event_position:
            text_segment = text[position:event_position]
            current_parent.append(text_segment)
        position = event_position
        
        if event_type == 'start_entity':
            # Start a new span for the entity
            span_tag = soup.new_tag(
                "span", 
                attrs={
                    "class": "ner-entity",
                    "data-color": category_colors.get(data, ''),
                    "alt": data,
                    "style": f"background-color: {category_colors.get(data, '')};"
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
            current_parent = p_tag  # Ensure current_parent is the paragraph tag
            
            # Add the current paragraph to the content if it has any children
            if p_tag.contents:
                content_elements.append(p_tag)
            
            # Only start a new paragraph if no 'start_speaker' events exist
            if not has_start_speaker:
                p_tag = soup.new_tag('p')
                current_parent = p_tag
            
            # Insert the anchor tag directly into the content
            anchor_tag = soup.new_tag(
                "a", 
                id=data,
                attrs={
                    "class": "anchor"
                })
            content_elements.append(anchor_tag)
        
        elif event_type == 'insert_linebreak':
            # Close any open spans before ending the paragraph
            spans_stack.clear()
            current_parent = p_tag  # Ensure current_parent is the paragraph tag
            
            # Add the current paragraph to the content if it has any children
            if p_tag.contents:
                content_elements.append(p_tag)
            
            # Start a new paragraph
            p_tag = soup.new_tag('p')
            current_parent = p_tag
        
        elif event_type == 'start_speaker':
            span_tag = soup.new_tag(
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
    if position < len(text):
        text_segment = text[position:]
        current_parent.append(text_segment)
    
    # Add the last paragraph to the content if it has any children
    if p_tag.contents:
        content_elements.append(p_tag)
    
    # Replace the old paragraph with the new content
    old_p_tag = soup.p
    for element in content_elements:
        old_p_tag.insert_before(element)
    old_p_tag.decompose()
    
    return str(soup)


def save_html(html_content, filepath):
    with open(filepath, 'w', encoding='utf-8') as fout:
        fout.write(html_content)
    print(f"→ Saved {filepath}.")
    