document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('scholar-search-input');
    const searchButton = document.getElementById('search-button');
    const resultsContainer = document.getElementById('results-container');
    const emptyState = document.querySelector('.empty-state');
    const loadingState = document.querySelector('.loading-state');

    // Perform search
    async function performSearch(query) {
        if (!query) return;

        emptyState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        resultsContainer.innerHTML = ''; // Clear previous results
        resultsContainer.appendChild(loadingState); // Keep loading visible

        try {
            const response = await fetch(`/api/scholar/search?q=${encodeURIComponent(query)}`);
            const data = await response.json();

            loadingState.classList.add('hidden');

            if (data.results && data.results.length > 0) {
                renderResults(data.results);
            } else {
                resultsContainer.innerHTML = '<div class="no-results">No results found for your query.</div>';
            }
        } catch (error) {
            console.error('Search failed:', error);
            loadingState.classList.add('hidden');
            resultsContainer.innerHTML = '<div class="error">An error occurred while searching.</div>';
        }
    }

    // Render results
    function renderResults(results) {
        resultsContainer.innerHTML = ''; // Clear loading/empty

        results.forEach(result => {
            const resultEl = document.createElement('div');
            resultEl.className = 'search-result';
            
            // Format authors/meta: "J Danger, ICEBURG Surveyor - 2026 - iceburg.local"
            const metaLine = `${result.authors || 'ICEBURG Agent'} - ${result.year || '2026'} - ${result.source || 'Internal Corpus'}`;
            
            // Safe encode for data attributes
            const safeTitle = (result.title || "").replace(/"/g, "&quot;");
            const safeAuthor = (result.authors || "ICEBURG Agent").replace(/"/g, "&quot;");
            const safeYear = (result.year || "2026").replace(/"/g, "&quot;");
            const safeId = (result.id || "report").replace(/"/g, "&quot;");

            resultEl.innerHTML = `
                <h3 class="result-title">
                    <a href="${result.link}" target="_blank">${result.title}</a>
                </h3>
                <div class="result-meta">${metaLine}</div>
                <div class="result-snippet">${result.snippet || 'No snippet available.'}</div>
                <div class="result-actions">
                    <a href="#" class="action-cite" 
                       data-id="${safeId}"
                       data-title="${safeTitle}" 
                       data-author="${safeAuthor}" 
                       data-year="${safeYear}">Cite</a>
                    <a href="pegasus.html?q=${encodeURIComponent(result.title)}" class="action-related" target="_blank">Related Graph</a>
                    <a href="${result.link}" target="_blank">Full Report</a>
                </div>
            `;
            resultsContainer.appendChild(resultEl);
        });

        // Add event listeners for dynamic actions
        document.querySelectorAll('.action-cite').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Retrieve data from attributes
                const title = btn.dataset.title;
                const author = btn.dataset.author;
                const year = btn.dataset.year;
                const id = btn.dataset.id.replace(/[^a-zA-Z0-9]/g, "");
                
                const bibtex = `@techreport{${id}${year},
  title={${title}},
  author={${author}},
  year={${year}},
  institution={ICEBURG Autonomous Research},
  type={Internal Report}
}`;

                navigator.clipboard.writeText(bibtex).then(() => {
                    const originalText = btn.textContent;
                    btn.textContent = "Copied!";
                    setTimeout(() => {
                        btn.textContent = originalText;
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy citation:', err);
                    alert("Failed to copy to clipboard");
                });
            });
        });
    }

    // Event listeners
    searchButton.addEventListener('click', () => {
        performSearch(searchInput.value);
    });

    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch(searchInput.value);
        }
    });
    
    // Auto-search if query param present (e.g. from main app)
    const urlParams = new URLSearchParams(window.location.search);
    const initialQuery = urlParams.get('q');
    if (initialQuery) {
        searchInput.value = initialQuery;
        performSearch(initialQuery);
    }
});
