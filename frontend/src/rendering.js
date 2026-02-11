/**
 * ICEBURG Rendering Module
 * ========================
 * Handles markdown, code syntax highlighting, and math rendering
 * 
 * Dependencies: marked, hljs, katex (loaded from CDN as globals)
 */

// Initialize marked with code highlighting
if (typeof marked !== 'undefined') {
    marked.setOptions({
        highlight: function (code, lang) {
            if (typeof hljs === 'undefined') {
                return code; // Fallback if hljs not loaded
            }
            if (lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, { language: lang }).value;
                } catch (err) {
                    return hljs.highlightAuto(code).value;
                }
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true
    });

    // Custom renderer for charts
    const renderer = new marked.Renderer();
    const originalCode = renderer.code.bind(renderer);
    renderer.code = function (code, lang, escaped) {
        if (lang === 'chart' || lang === 'chartjs') {
            try {
                JSON.parse(code); // Validate JSON
                const chartId = 'chart-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
                return `<div class="chart-container"><canvas id="${chartId}" data-chart='${code.replace(/'/g, "&apos;")}'></canvas></div>`;
            } catch (e) {
                return originalCode(code, lang, escaped);
            }
        }
        return originalCode(code, lang, escaped);
    };
    marked.setOptions({ renderer });
}

/**
 * Render LaTeX math equations in content
 * @param {string} content - Content with LaTeX math (inline: $...$, block: $$...$$)
 * @returns {string} Content with rendered math
 */
export function renderMath(content) {
    if (typeof katex === 'undefined') {
        return content; // Fallback if katex not loaded
    }
    
    // Render inline math: $...$
    content = content.replace(/\$([^$]+)\$/g, (match, formula) => {
        try {
            return katex.renderToString(formula, { throwOnError: false, displayMode: false });
        } catch (e) {
            return match;
        }
    });
    
    // Render block math: $$...$$
    content = content.replace(/\$\$([^$]+)\$\$/g, (match, formula) => {
        try {
            return katex.renderToString(formula, { throwOnError: false, displayMode: true });
        } catch (e) {
            return match;
        }
    });
    
    return content;
}

/**
 * Render markdown text to HTML
 * @param {string} text - Markdown text
 * @returns {string} HTML string
 */
export function renderMarkdown(text) {
    if (typeof marked === 'undefined') {
        return text; // Fallback if marked not loaded
    }
    return marked.parse(text);
}

/**
 * Render code block with syntax highlighting
 * @param {string} code - Code text
 * @param {string} language - Language identifier (optional)
 * @returns {string} HTML string with highlighted code
 */
export function renderCode(code, language = '') {
    if (typeof hljs === 'undefined') {
        return `<pre><code>${escapeHtml(code)}</code></pre>`;
    }
    
    try {
        if (language && hljs.getLanguage(language)) {
            return `<pre><code class="language-${language}">${hljs.highlight(code, { language }).value}</code></pre>`;
        } else {
            return `<pre><code>${hljs.highlightAuto(code).value}</code></pre>`;
        }
    } catch (e) {
        return `<pre><code>${escapeHtml(code)}</code></pre>`;
    }
}

/**
 * Highlight code blocks in a DOM element
 * @param {HTMLElement} element - Element containing code blocks
 */
export function highlightCodeBlocks(element) {
    if (typeof hljs === 'undefined') {
        return; // Skip if hljs not loaded
    }
    
    element.querySelectorAll('pre code').forEach((block) => {
        try {
            hljs.highlightElement(block);
        } catch (e) {
            // Ignore highlighting errors
        }
    });
}

/**
 * Render complete message content (markdown + code + math)
 * @param {string} content - Raw message content (markdown)
 * @returns {string} Fully rendered HTML
 */
export function renderMessage(content) {
    if (!content) return '';
    
    // Step 1: Render markdown
    let html = renderMarkdown(content);
    
    // Step 2: Render math equations
    html = renderMath(html);
    
    return html;
}

/**
 * Render message content into a DOM element
 * @param {HTMLElement} element - Target element
 * @param {string} content - Raw message content (markdown)
 * @param {function} onRendered - Optional callback after rendering
 */
export function renderMessageIntoElement(element, content, onRendered) {
    if (!element) return;
    
    // Render content
    const html = renderMessage(content);
    element.innerHTML = html;
    
    // Highlight code blocks after rendering
    highlightCodeBlocks(element);
    
    // Call callback if provided
    if (onRendered) {
        // Use requestAnimationFrame to ensure DOM is ready
        requestAnimationFrame(() => {
            onRendered(element);
        });
    }
}

/**
 * Escape HTML special characters
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
