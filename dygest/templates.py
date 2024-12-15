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

    #back-to-top {
        position: fixed;
        bottom: 20px;
        right: 20px;
        display: none; 
        z-index: 99; 
        background-color: #03a9f4;
        font-weight: 100;
        color: white;
        border: none;
        padding: 10px 10px;
        text-align: center;
        text-decoration: none;
        font-size: 15px;
        cursor: pointer;
        border-radius: 20px;
        transition: 0.1s;
        text-transform: none;
        line-height: 1;
        vertical-align: middle;
        border-bottom: 2px solid #0074aa;
    }

    #back-to-top:hover {
        background-color: #0289c7;
        border-top: 2px solid #0078b0;
        border-bottom: 2px solid #0078b0;
    }

    .content {
        line-height: 1.75;
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
    
    .edits {
        border: 0;
    }

    .edits:focus {
        border: 3px solid deeppink;
        background-color: rgb(255, 215, 215);
        padding: 4px;
        color: black;
        border-radius: 10px;
        outline: none; 
    }

    .document-controls button {
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
        font-family: monospace;
        margin-left: 12px;
        cursor: pointer; 
        position: relative; 
        white-space: pre-wrap;
        overflow-wrap: break-word;
    }

    .metadata-content {
        font-family: monospace;
        font-size: 15px;
        display: none;
        margin-left: 10px; 
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
        font-family: monospace; 
        font-size: 16px;
        cursor: default;
    }

    .tldr {
        color: #666666;
        font-family: monospace;
        margin-left: 10px;
        margin-bottom: 15px;
        font-family: monospace;
        font-size: 15px;
    }

    .tldr-content {
        font-style: italic;
    }

    ol ul {
        list-style: none;
        font-size: 95%;
        line-break: loose;
        padding: 0;
        margin-top: 5px;
        margin-bottom: 5px;
        margin-left: 14px;
    }

    ul li {
        margin: 0; 
        padding: 0; 
        position: relative;
        padding-left: 10px; 
    }

    ul li::before {
        content: "•";
        position: absolute;
        left: 0; 
        margin-right: 5px;
        color: black;
        top: 0; 
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
                <div class="document-controls">
                    <button onclick="decreaseFontSize()">aa</button>
                    <button onclick="increaseFontSize()">AA</button>
                </div>
                
                <div class="additional-controls">
                    <!-- Additional Buttons (Timestamps, NER, saving) -->
                </div>
                
            </div>
        </div>
        
        <div class="two-thirds column">

            <!-- Metadata -->
            <div class="metadata">
                <h6 class="metadata-header"></h6>
                <div class="metadata-content"></div>
            </div>
            
            <!-- TL;DR -->
            <div class="tldr">
            </div>

            <!-- Main Content -->
            <div class="content" contenteditable="true">
                <p></p>
            </div>
        </div>
    </div>
</div>
<button onclick="scrollToTop()" id="back-to-top">Top</button>
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
        let elements = document.querySelectorAll('.content');
        elements.forEach(function(element) {
            let currentSize;

            // Try to read the current font size from the inline style
            if (element.style.fontSize) {
                currentSize = parseFloat(element.style.fontSize);
                console.log('Current font size from inline style:', currentSize, 'px');
            } else {
                // If not set, use the initial font size (e.g., 18px)
                currentSize = 18;
                console.log('No inline font size set. Using initial font size:', currentSize, 'px');
            }

            let newSize = currentSize + delta;
            console.log('New font size:', newSize, 'px');

            if (newSize >= 6 && newSize <= 48) {
                element.style.fontSize = newSize + 'px';
                console.log('Font size updated to:', newSize, 'px');
            } else {
                console.log('New font size out of bounds:', newSize, 'px');
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
    // Select metadata headers
    const metadataHeaders = document.querySelectorAll('.metadata-header');

    metadataHeaders.forEach(header => {
        header.addEventListener('click', function() {
            // Toggle the 'expanded' class on the parent .metadata div
            const metadataDiv = this.parentElement;
            metadataDiv.classList.toggle('expanded');
            });
        });
    });

    // Back to Top Button
    window.onscroll = function() { scrollFunction() };

    function scrollFunction() {
        var backToTopButton = document.getElementById("back-to-top");
        if (document.body.scrollTop > 500 || document.documentElement.scrollTop > 500) {
            backToTopButton.style.display = "block";
        } else {
            backToTopButton.style.display = "none";
        }
    }

    function scrollToTop() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
</script>
</body>
</html>
"""
