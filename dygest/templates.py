CREATE_SUMMARIES_BACKUP = """
Perform the following tasks on the provided text:

1. **Identify the top 2 most important topics** discussed in the text.
2. **For each identified topic**, generate one summary of exactly 1 sentence of max. 25 words that effectively captures the key points related to that topic.

**Return the results in the language of the Input Text as JSON. Do NOT add anything else and use this template without any alterations:**

[
  {
    "topic": "Topic Name",
    "summary": "A concise summary of 1 sentence related to the topic",
    "location": "First 5 words of the corresponding text chunk of the Input Text"
  },
  ...
]

**Input Text:**
"""


CREATE_SUMMARIES = """
Perform the following tasks on the provided text:

1. **Identify the top 2 most important topics** discussed in the text.
2. **For each identified topic**, generate a concise description using a short phrase or group of keywords that effectively captures the key points related to that topic.

**Return the results in the language of the Input Text as JSON. Do NOT add anything else and use this template without any alterations:**

[
  {
    "topic": "Topic Name",
    "summary": "A concise phrase related to the topic",
    "location": "First 5 words of the corresponding text chunk of the Input Text"
  },
  ...
]

**Input Text:**
"""


CLEAN_SUMMARIES = """
Perform the following tasks on the provided list of summaries:

**Identify overlapping summaries**: Analyze the summaries to find any that heavily overlapping in content or topic.
**Remove duplicates**: Remove any overlapping summaries, ensuring that each topic is represented by only one summary.
**Return a list of unique summaries**: Provide a new list containing only the unique summaries.
**Do NOT change the remaining summaries in any way**

**Return the results in the language of the Input Summaries as **. **Do NOT add anything else** and use this template without any alterations:

[
  {
    "topic": "Topic Name",
    "summary": "A concise summary of exactly 1 sentence related to the topic",
    "location": "First 5 words of the corresponding text chunk of the Input Text"
  },
  ...
]

**Input Summaries:**
"""


HTML_CONTENT = """
<html>
<head>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css">
<style>
    body {
        background-color: #ffffff;
        font-family: Arial, sans-serif;
        color: #000000;
        font-size: 18px; 
        margin: 20px;
    }
    
    a {
        color: #3d64bb; 
        text-decoration: none;
    }

    a:hover {
        color: #296af6; 
        text-decoration: underline; 
    }
    
    .anchor {
        text-decoration: underline;
    }

    button {
        background-color: #03a9f4;
        font-weight: 100;
        color: white;
        border: none;
        padding: 10px 10px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 15px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 20px;
        transition: 0.1s;
        text-transform: none;
        line-height: 1;
        vertical-align: middle;
        border-bottom: 2px solid #0074aa;
    }

    button:hover {
        background-color: #0289c7;
        color: white;
        border-top: 2px solid #0078b0;
        border-bottom: 2px solid #0078b0;
    }

    button:active {
        background-color: #03a9f4;
        color: white;
    }

    button:focus {
        background-color: #03a9f4;
        color: white;
    }
    
    button.save {
        background-color: #1bc89a;
        border-bottom: 2px solid #14916f;
    }

    button.save:hover {
        background-color: #15a17b;
        color: white;
        border-top: 2px solid #117b5f;
        border-bottom: 2px solid #117b5f;
    }
    
    .controls {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: center;
    }
    
    div[contenteditable="true"] {
        border-radius: 10px;
        padding: 10px;
        min-height: 100px; 
    }
    
    div[contenteditable="true"]:focus {
        background-color: #f6f6f6;
        border-radius: 10px;
        padding: 10px;
        min-height: 100px; 
    }
    
    .font-size-controls button {
        display: inline-block;
        width: auto;
        margin: 5px 2px;
    }

    .additional-controls {
        display: inline-block;
        width: auto;
        margin: 5px 2px;
    }
    
    h5 {
        font-size: 22px;
    }
    
    .metadata-header {
        font-family: 'Courier New', Courier, monospace;
        margin-left: 12px;
        cursor: pointer; 
        position: relative; 
        white-space: pre-wrap;
        overflow-wrap: break-word;
    }

    .metadata-content {
        font-family: 'Courier New', Courier, monospace;
        font-size: 15px;
        display: none;
        margin-left: 12px; 
        transition: max-height 0.3s ease, opacity 0.3s ease;
        overflow: hidden;
        opacity: 0;
        max-height: 0;
        margin-bottom: 10px;
    }

    .metadata.expanded .metadata-content {
        display: block;
        opacity: 1;
        max-height: 500px; 
    }

    .ner-entity {
        cursor: default;
    }
    
    .sidebar {
        border-radius: 10px;
    }

    .summaries {
        background-color: #f6f6f6; 
        padding: 10px;
        border: 2px solid #e5e5e5; 
        border-radius: 10px;
    }

    #summary-content {
        font-size: 15px;
    }
    
    .timestamp {
        color: #f41ad7; 
        font-family: 'Courier New', Courier, monospace; 
        font-size: 16px;
        cursor: default;
    }
</style>
</head>
<body>
<div class="container">
    <div class="row">
    
        <!-- Sidebar -->
        <div class="one-third column sidebar">
            <div class="controls">
            
                <!-- Control Font-Size -->
                <div class="font-size-controls">
                    <button onclick="decreaseFontSize()">aa</button>
                    <button onclick="increaseFontSize()">AA</button>
                </div>
                
                <div class="additional-controls">
                    <!-- Additional Buttons (Timestamps, NER, saving) -->
                </div>
                
            </div>
            <div class="summaries">
                <div id="summary-content" style="display: block">
                    <ol>
                        <!-- Summary items go here -->
                    </ol>
                </div>
            </div>
        </div>
        
        
        <div class="two-thirds column">

            <!-- Metadata -->
            <div class="metadata">
                <h6 class="metadata-header"></h6>
                <div class="metadata-content">
                    
                </div>
            </div>

            <!-- Main Content -->
            <div class="content" contenteditable="true">
                <p></p>
            </div>
        </div>
    </div>
</div>
<script>
    // Variables to keep track of the state
    var isTimestampVisible = true;
    var isHighlightingEnabled = true;
    var isSourceView = false;

    function toggleSource() {
        var contentArea = document.querySelector('.content');
        if (!isSourceView) {
            // Switch to source code view
            var htmlContent = contentArea.innerHTML;

            // Create a textarea to display the HTML code
            var textarea = document.createElement('textarea');
            textarea.value = htmlContent;
            textarea.style.width = '100%';
            textarea.style.height = '100%';
            textarea.id = 'source-editor';
            textarea.style.fontFamily = 'monospace';
            textarea.style.fontSize = '16px';
            textarea.style.lineHeight = '2';
            textarea.style.padding = '10px';
            textarea.style.border = '1px solid #ccc';
            textarea.style.backgroundColor = '#f6f6f6';
            textarea.style.borderRadius = '10px';

            // Replace the content div with the textarea
            contentArea.parentNode.replaceChild(textarea, contentArea);

            isSourceView = true;
        } else {
            // Switch back to visual view
            var textarea = document.getElementById('source-editor');
            var htmlContent = textarea.value;

            // Create a new content div
            var contentDiv = document.createElement('div');
            contentDiv.className = 'content';
            contentDiv.setAttribute('contenteditable', 'true');
            contentDiv.innerHTML = htmlContent;

            // Replace the textarea with the content div
            textarea.parentNode.replaceChild(contentDiv, textarea);

            isSourceView = false;
        }
    }

    function toggleTimestamp() {
        // Toggle the state variable
        isTimestampVisible = !isTimestampVisible;

        const timestampElements = document.querySelectorAll('.timestamp');
        timestampElements.forEach(function(element) {
            if (isTimestampVisible) {
                element.style.display = '';
            } else {
                element.style.display = 'none';
            }
        });
        var btn = document.getElementById('toggle-timestamp');
        btn.classList.toggle('active');
    }

    function toggleHighlighting() {
        // Toggle the state variable
        isHighlightingEnabled = !isHighlightingEnabled;

        const highlightedElements = document.querySelectorAll('.ner-entity');
        highlightedElements.forEach(function(element) {
            if (isHighlightingEnabled) {
                element.style.backgroundColor = element.getAttribute('data-color');
            } else {
                element.style.backgroundColor = '';
            }
        });
        var btn = document.getElementById('toggle-highlighting');
        btn.classList.toggle('active');
    }

    function changeFontSize(delta) {
        let elements = document.querySelectorAll('.content, .content *');
        elements.forEach(function(element) {
            let style = window.getComputedStyle(element).getPropertyValue('font-size');
            let currentSize = parseFloat(style); 
            let newSize = currentSize + delta;
            if (newSize >= 6 && newSize <= 48) { 
                element.style.fontSize = newSize + 'px';
            }
        });
    }

    function increaseFontSize() {
        changeFontSize(1); 
    }

    function decreaseFontSize() {
        changeFontSize(-1);
    }
    
    function savePage() {
        const htmlContent = document.documentElement.outerHTML;

        // Get the current page's filename
        let filename = window.location.pathname.split('/').pop();

        // If filename is empty (e.g., the URL ends with a '/'), set a default filename
        if (!filename) {
            filename = 'page';
        } else {
            // Remove any query parameters or hash fragments from the filename
            filename = filename.split('?')[0].split('#')[0];

            // Remove the file extension if it exists
            const dotIndex = filename.lastIndexOf('.');
            if (dotIndex > -1) {
                filename = filename.substring(0, dotIndex);
            }
        }

        filename = filename + '_edit.html';

        const blob = new Blob([htmlContent], { type: 'text/html' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = filename; 

        document.body.appendChild(link);
        link.click();

        document.body.removeChild(link);
    }

    document.addEventListener('DOMContentLoaded', function() {
    // Select all metadata headers
    const metadataHeaders = document.querySelectorAll('.metadata-header');

    metadataHeaders.forEach(header => {
        header.addEventListener('click', function() {
            // Toggle the 'expanded' class on the parent .metadata div
            const metadataDiv = this.parentElement;
            metadataDiv.classList.toggle('expanded');
            });
        });
    });
</script>
</body>
</html>
"""
