"""
Core HTML renderer for PyTorch model analysis.
"""
import os
import jinja2
import torch
import re
from typing import Dict, List, Tuple, Optional, Union, Any

class HTMLRenderer:
    """HTML rendering class for PyTorch model analysis"""
    
    @staticmethod
    def get_base_template(title="PyTorch Formula Analysis"):
        """Return the base HTML template"""
        template = """<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- KaTeX for faster rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css" integrity="sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js" integrity="sha384-cpW21h6RZv/phavutF+AuVYrr+dA8xD9zs6FwLpaCct6O9ctzYFfFr4dgmgccOTx" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "$", right: "$", display: false},
                    {left: "\\\\[", right: "\\\\]", display: true},
                    {left: "\\\\(", right: "\\\\)", display: false}
                ],
                throwOnError: false
            });
        });
    </script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; border-bottom: 1px solid #3498db; padding-bottom: 5px; }
        h3 { color: #2980b9; }
        h4 { color: #27ae60; }
        .layer-summary { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .forward-pass { border-left: 5px solid #2ecc71; padding-left: 15px; }
        .backward-pass { border-left: 5px solid #e74c3c; padding-left: 15px; }
        .tensor-table { border-collapse: collapse; margin: 10px 0; width: auto; }
        .tensor-table th, .tensor-table td { border: 1px solid #ddd; padding: 8px; text-align: right; }
        .tensor-table th { background-color: #f2f2f2; }
        .tensor-header { background-color: #e3f2fd; }
        .channel-separator { border-bottom: 2px solid #3498db; }
        .batch-separator { border-bottom: 3px double #e74c3c; }
        .formula { background-color: #f9f9f9; padding: 10px; border-radius: 5px; overflow-x: auto; margin: 15px 0; }
        .general-formula { font-weight: bold; }
        .math-container { overflow-x: auto; padding: 10px 0; }
        .collapsible { 
            background-color: #f1f1f1;
            cursor: pointer;
            padding: 10px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 15px;
            border-radius: 5px;
        }
        .active, .collapsible:hover { background-color: #e0e0e0; }
        .content { 
            padding: 0 18px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
            background-color: #ffffff;
            border-radius: 0 0 5px 5px;
        }
        /* Improved responsive design */
        @media screen and (max-width: 768px) {
            body { margin: 10px; }
            .tensor-table { width: 100%; overflow-x: auto; display: block; }
            .math-container { overflow-x: auto; max-width: 100%; }
            .tab button { padding: 10px 12px; font-size: 15px; }
        }
        .visualization-container { 
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
            margin: 20px 0;
        }
        .visualization-item {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .visualization-item img {
            max-width: 100%;
            display: block;
        }
        .visualization-caption {
            padding: 10px;
            background-color: #f9f9f9;
            text-align: center;
            font-weight: bold;
        }
        .tab {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            border-radius: 5px 5px 0 0;
        }
        .tab button {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 17px;
        }
        .tab button:hover { background-color: #ddd; }
        .tab button.active { background-color: #ccc; }
        .tabcontent {
            display: none;
            padding: 6px 12px;
            border: 1px solid #ccc;
            border-top: none;
            border-radius: 0 0 5px 5px;
        }
    </style>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Collapsible sections
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                }
            });
        }
        
        // Tabs
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
            
            // Re-render KaTeX in the newly displayed tab
            if (typeof renderMathInElement === 'function') {
                renderMathInElement(document.getElementById(tabName), {
                    delimiters: [
                        {left: "$$", right: "$$", display: true},
                        {left: "$", right: "$", display: false},
                        {left: "\\\\[", right: "\\\\]", display: true},
                        {left: "\\\\(", right: "\\\\)", display: false}
                    ],
                    throwOnError: false
                });
            }
        }
        
        // Open the first tab by default
        if (document.getElementsByClassName("tablinks").length > 0) {
            document.getElementsByClassName("tablinks")[0].click();
        }
        
        // Expose the openTab function globally
        window.openTab = openTab;
    });
    </script>
</head>
<body>
    {{ content }}
</body>
</html>"""
        return template
    
    @staticmethod
    def wrap_in_template(content, title="PyTorch Formula Analysis"):
        """Wrap content in the HTML template"""
        env = jinja2.Environment()
        template = env.from_string(HTMLRenderer.get_base_template())
        return template.render(title=title, content=content)
    
    @staticmethod
    def save_html(html_content, output_path):
        """Save HTML content to a file"""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return output_path
    
    @staticmethod
    def create_collapsible_section(title, content, is_open=False):
        """Create a collapsible section in HTML"""
        html = f"""
        <button class="collapsible" {"class='active'" if is_open else ""}>{title}</button>
        <div class="content" {"style='max-height:none;'" if is_open else ""}>
            {content}
        </div>
        """
        return html
    
    @staticmethod
    def create_tabs(tabs_dict):
        """Create a tabbed interface in HTML
        
        Args:
            tabs_dict: Dictionary with tab titles as keys and content as values
        """
        html = '<div class="tab">'
        tab_content = ""
        
        for i, (title, content) in enumerate(tabs_dict.items()):
            tab_id = f"tab_{title.replace(' ', '_').lower()}"
            html += f'<button class="tablinks" onclick="openTab(event, \'{tab_id}\')">{title}</button>'
            tab_content += f'<div id="{tab_id}" class="tabcontent">{content}</div>'
        
        html += '</div>'
        html += tab_content
        
        return html
    
    @staticmethod
    def create_visualization_grid(images_dict):
        """Create a grid of visualizations
        
        Args:
            images_dict: Dictionary with image titles as keys and image paths as values
        """
        html = '<div class="visualization-container">'
        
        for title, image_path in images_dict.items():
            html += f"""
            <div class="visualization-item">
                <img src="{image_path}" alt="{title}">
                <div class="visualization-caption">{title}</div>
            </div>
            """
        
        html += '</div>'
        
        return html
    
    @staticmethod
    def format_latex_formula(formula, display_mode=False):
        """
        Format a LaTeX formula for proper KaTeX rendering
        
        Args:
            formula: The LaTeX formula to format
            display_mode: Whether to use display mode (centered, larger) or inline mode
            
        Returns:
            Properly formatted LaTeX formula for HTML/KaTeX
        """
        # Make sure the formula doesn't already have delimiters
        formula = formula.strip()
        
        # Remove existing delimiters if present
        if (formula.startswith('$') and formula.endswith('$')) or \
           (formula.startswith('\\(') and formula.endswith('\\)')) or \
           (formula.startswith('\\[') and formula.endswith('\\]')):
            # Extract the formula without delimiters
            if formula.startswith('$') and formula.endswith('$'):
                formula = formula[1:-1]
            elif formula.startswith('\\(') and formula.endswith('\\)'):
                formula = formula[2:-2]
            elif formula.startswith('\\[') and formula.endswith('\\]'):
                formula = formula[2:-2]
        
        # For KaTeX, use simpler delimiters
        if display_mode:
            return f'$${formula}$$'
        else:
            return f'${formula}$'
    
    @staticmethod
    def create_formula_section(title, formula, explanation=None, display_mode=True):
        """
        Create a nicely formatted section for a mathematical formula
        
        Args:
            title: Title of the formula section
            formula: LaTeX formula to display
            explanation: Optional explanation text
            display_mode: Whether to use display mode for the formula
            
        Returns:
            HTML for a formula section
        """
        formatted_formula = HTMLRenderer.format_latex_formula(formula, display_mode)
        
        html = f'<div class="formula">'
        if title:
            html += f'<h4>{title}</h4>'
        
        html += f'<div class="math-container">{formatted_formula}</div>'
        
        if explanation:
            html += f'<p>{explanation}</p>'
        
        html += '</div>'
        
        return html
