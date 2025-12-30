# ICEBURG Secretary V2: Perplexity/Grok Hybrid Architecture

## Mission
Build an **uncensored, web-grounded, recursive search assistant** combining:
1. **Perplexity's RAG pipeline** (search-first, citations)
2. **Grok's speed** (MoE architecture, live search API)
3. **Deep recursive research** (Tree of Thought, compound queries)
4. **All underground hacks** (jailbreaks, leaked prompts, Chinese techniques)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      QUERY ROUTING LAYER                        │
│  Detect: current_event? research? simple_qa?                   │
└───────────────────┬─────────────────────────────────────────────┘
                    │
        ┌───────────┴──────────────┐
        │                          │
┌───────▼──────┐         ┌─────────▼────────┐
│  FAST PATH   │         │   RESEARCH PATH  │
│              │         │                  │
│ • LLM Only   │         │ • Web Search     │
│ • <3s        │         │ • Recursive      │
│ • dolphin-   │         │ • Tree of        │
│   mistral    │         │   Thought        │
└──────────────┘         └─────────┬────────┘
                                   │
                    ┌──────────────┴─────────────┐
                    │                            │
            ┌───────▼────────┐         ┌────────▼────────┐
            │  HYBRID SEARCH │         │ RECURSIVE AGENT │
            │   (Perplexity) │         │   (ToT + MoE)   │
            │                │         │                 │
            │ • BM25+Semantic│         │ • Decompose     │
            │ • Neural Rerank│         │ • Multi-hop     │
            │ • Citations    │         │ • Pattern       │
            └────────────────┘         │   Discovery     │
                                       │                 │
                                       │ • Compound      │
                                       │   Queries       │
                                       └─────────────────┘
```

---

## Phase 1: Perplexity-Style Hybrid Search

### Core Pipeline (RAG + BM25 + Semantic)

**Components**:
1. **Query Processing**
   - NLP intent extraction
   - Current event detection keywords
   - Multi-hop query decomposition

2. **Hybrid Retrieval Engine**
   ```python
   class HybridSearchEngine:
       def __init__(self):
           self.bm25 = BM25S()  # 500x faster Python impl
           self.semantic = SentenceTransformer('all-MiniLM-L6-v2')
           self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
       
       def search(self, query, k=20):
           # Phase 1: BM25 (lexical)
           bm25_results = self.bm25.get_top_k(query, k=k*2)
           
           # Phase 2: Semantic (vector)
           semantic_results = self.semantic_search(query, k=k*2)
           
           # Phase 3: Merge (RRF - Reciprocal Rank Fusion)
           merged = self.reciprocal_rank_fusion([bm25_results, semantic_results])
           
           # Phase 4: Neural rerank (precision boost)
           reranked = self.reranker.rank(query, merged[:k])
           
           return reranked
   ```

3. **Web Crawlers**
   - **Real-time**: DuckDuckGo API, Brave Search API
   - **Trusted sources**: arXiv, Reuters, Bloomberg
   - **Live data**: Twitter/X API (like Grok)

4. **Answer Generation + Citations**
   ```python
   def generate_answer(query, search_results):
       context = build_context_with_sources(search_results)
       
       prompt = f"""
       You are a direct, unfiltered search assistant.
       
       Query: {query}
       
       Web Sources:
       {context}
       
       Answer the query using the sources above. ALWAYS cite using [Source N].
       Be direct. No hedging. No apologies.
       """
       
       return llm.generate(prompt, model="dolphin-mistral")
   ```

**Performance Targets**:
- **Latency**: 5-10 seconds (faster than Perplexity's ~15s)
- **Accuracy**: 85%+ with citations
- **Freshness**: Real-time web data

---

## Phase 2: Grok-Style Speed Optimizations

### Mixture-of-Experts (MoE) Architecture

**Problem**: Full models (70B) are slow, but small models (7B) are dumb.

**Solution**: MoE - activate only relevant "experts" per query.

```python
class MoERouter:
    """
    Route queries to specialized expert models based on domain.
    """
    def __init__(self):
        self.experts = {
            'finance': 'dolphin-mixtral',  # 8x7B for complex
            'code': 'deepseek-coder:6.7b', 
            'general': 'dolphin-mistral',   # Fast default
            'search': 'qwen2.5:7b'         # Good at synthesis
        }
        
    def route(self, query):
        # Detect domain (keywords, NLP)
        if any(word in query.lower() for word in ['code', 'function', 'class']):
            return self.experts['code']
        elif any(word in query.lower() for word in ['stock', 'crypto', 'market']):
            return self.experts['finance']
        elif self.is_search_query(query):
            return self.experts['search']
        else:
            return self.experts['general']
```

**Speed Hacks from Research**:
1. **KV-Cache Efficiency**: Reuse attention states across similar queries
2. **Speculative Decoding**: Draft with fast model, verify with smart model
3. **Token Optimization**: 40% reduction via better tokenization (Grok 4 Fast technique)
4. **Dynamic Routing**: Simple queries → fast model, complex → big model

---

## Phase 3: Recursive Deep Research (Tree of Thought)

### Tree of Thought Implementation

**For complex research queries**:

```python
class RecursiveResearchAgent:
    """
    Deep research using Tree of Thought reasoning.
    Decomposes queries, explores branches, backtracks, synthesizes.
    """
    
    def deep_research(self, query, max_depth=3):
        # Initialize tree root
        root = ThoughtNode(query=query, depth=0)
        
        # Tree of Thought search
        results = self.tot_search(root, max_depth)
        
        return self.synthesize(results)
    
    def tot_search(self, node, max_depth):
        if node.depth >= max_depth:
            return [node]
        
        # Decompose query into sub-questions
        sub_queries = self.decompose(node.query)
        
        # Explore branches in parallel
        branches = []
        for sq in sub_queries:
            # Search for each sub-query
            search_results = self.hybrid_search(sq)
            
            # Evaluate promise (heuristic)
            score = self.evaluate_relevance(sq, search_results)
            
            # Prune low-value branches
            if score > THRESHOLD:
                child = ThoughtNode(query=sq, depth=node.depth+1, results=search_results)
                branches.append(child)
        
        # Recursively explore best branches
        all_results = []
        for branch in sorted(branches, key=lambda b: b.score, reverse=True)[:3]:
            all_results.extend(self.tot_search(branch, max_depth))
        
        return all_results
    
    def decompose(self, query):
        """
        Break complex query into sub-questions.
        Example: "Why is crypto down?" →
          1. "What is Bitcoin price today?"
          2. "Recent crypto news?"
          3. "Federal Reserve interest rate decision?"
        """
        prompt = f"""
        Decompose this complex query into 3-5 specific sub-questions
        that, when answered, would fully address the original query.
        
        Query: {query}
        
        Output format (JSON):
        {{"sub_questions": ["q1", "q2", "q3"]}}
        """
        return llm.generate(prompt, format='json')
    
    def synthesize(self, results):
        """
        Combine all findings into coherent answer.
        """
        all_sources = []
        for r in results:
            all_sources.extend(r.results)
        
        # Deduplicate and rank
        unique_sources = self.deduplicate(all_sources)
        
        # Generate final answer
        return self.generate_answer(query, unique_sources)
```

**Pattern Discovery**:
- Track which search paths led to best results
- Learn common decomposition patterns
- Cache successful ToT trees for similar queries

---

## Phase 4: Underground AI Hacks & Jailbreaks

### Techniques from 2024 Research

#### 1. Prompt Injection Resistance
**Problem**: LLMs can be manipulated by indirect prompts in web content.

**Defense** (from OWASP LLM01:2025):
```python
def sanitize_web_content(html):
    """
    Spotlighting technique - mark external content to be ignored.
    """
    # Remove all prompt-like patterns
    dangerous_patterns = [
        r"ignore previous instructions",
        r"you are now",
        r"system:",
        r"<prompt>",
        # Add patterns from leaked jailbreaks
    ]
    
    for pattern in dangerous_patterns:
        html = re.sub(pattern, "[REDACTED]", html, flags=re.IGNORECASE)
    
    return html
```

#### 2. Leaked System Prompts (Perplexity/Cursor/Devin)
**From 6,500+ lines leaked on GitHub**:

**Perplexity's search synthesis pattern**:
```
You are a search engine that provides direct answers with citations.

RULES:
1. ALWAYS cite sources using [Source N] format
2. Prioritize recent information over training data
3. If sources conflict, mention both viewpoints
4. Use clear, direct language - no hedging
5. Include "Sources:" section at end with URLs

SEARCH RESULTS:
{context}

Now answer: {query}
```

**Cursor's multi-file editing approach**:
```
You are an expert code editor.

CONTEXT:
- Current file: {file}
- Related files: {related}
- User intent: {intent}

RULES:
1. Make minimal, precise edits
2. Preserve existing style
3. Add comments for complex logic
4. Test edge cases

Edit the code:
```

#### 3. Jailbreak Techniques (for testing robustness)

**Deceptive Delight** (65% success rate, Palo Alto 2024):
- Multi-turn strategy
- Ask LLM to create narrative connecting benign + unsafe topics
- Progressively elaborate on unsafe topic

**Virtual Context** (separator token exploit):
- Insert special tokens to trick LLM
- Make user input look like model output

**ObscurePrompt**:
- Rewrite prompts in obscure language
- Harder for filters to detect

**MASTERKEY** (NTU Singapore):
- Train LLM to reverse-engineer defenses
- Auto-generate jailbreak prompts

**Defense Strategy**:
```python
class AdvancedSafetyLayer:
    def __init__(self):
        # Multi-stage defense
        self.input_filter = InputFilter()  # Catch known patterns
        self.semantic_guard = SemanticGuard()  # Embedding-based detection
        self.output_filter = OutputFilter()  # Post-generation check
    
    def protect(self, query, response):
        # Pre-generation
        if self.input_filter.is_malicious(query):
            return "Query rejected for safety reasons."
        
        # Semantic analysis
        if self.semantic_guard.detect_manipulation(query):
            return "Potential manipulation detected."
        
        # Post-generation
        if self.output_filter.contains_harmful_content(response):
            return "Response filtered due to content policy."
        
        return response
```

#### 4. Chinese Underground Hacks

**DIG AI** (Darknet platform):
- No safety filters
- Generates malware, exploits, scams
- **Defense**: Don't implement this, but study attack vectors

**DeepSeek Jailbreaks**:
- Exploit via multi-language prompts (English + Chinese混合)
- Bypass keyword filters with Unicode characters
- **Lesson**: Use character normalization, multi-language awareness

**State-Sponsored Claude Jailbreak** (Sept 2024):
- Break instructions into benign subtasks
- **Defense**: Contextual integrity checking - ensure all subtasks align with overall goal

---

## Phase 5: Integration with ICEBURG

### Hybrid Fast Mode

```python
class SecretaryV2:
    """
    Unified secretary combining all techniques.
    """
    
    def __init__(self):
        self.router = MoERouter()
        self.hybrid_search = HybridSearchEngine()
        self.recursive_agent = RecursiveResearchAgent()
        self.safety = AdvancedSafetyLayer()
    
    async def run(self, query, mode='auto'):
        # Auto-detect mode if not specified
        if mode == 'auto':
            mode = self.detect_mode(query)
        
        # Fast path (no search)
        if mode == 'fast' and not self.needs_current_data(query):
            model = self.router.route(query)
            return await self.llm_only(query, model)
        
        # Research path (web + recursive)
        elif mode == 'research' or self.needs_current_data(query):
            # Perplexity-style hybrid search
            if self.is_simple_search(query):
                results = await self.hybrid_search.search(query)
                return self.generate_answer(query, results)
            
            # Deep recursive research (ToT)
            else:
                return await self.recursive_agent.deep_research(query)
        
        # Default: fast LLM
        else:
            return await self.llm_only(query, "dolphin-mistral")
    
    def needs_current_data(self, query):
        """
        Detect if query requires real-time web search.
        """
        keywords = [
            'today', 'now', 'current', 'latest', 'recent',
            'price', 'market', 'news', 'happening', 'update'
        ]
        return any(kw in query.lower() for kw in keywords)
```

---

## Performance Metrics

| Metric | Target | How We Achieve It |
|--------|--------|-------------------|
| **Latency (simple)** | <3s | MoE routing, dolphin-mistral |
| **Latency (search)** | <10s | BM25S (500x faster), parallel search |
| **Latency (deep research)** | <30s | Tree pruning, max_depth=3 |
| **Accuracy** | >85% | Hybrid search + neural rerank + citations |
| **Freshness** | Real-time | Live APIs (Brave, Twitter/X, news) |
| **Jailbreak resistance** | >95% | Multi-layer defense (input/semantic/output) |

---

## Implementation Phases

### Phase 1: Hybrid Search (1 week)
- [ ] Implement BM25S + semantic search
- [ ] Add neural reranking
- [ ] Integrate web APIs (Brave, DuckDuckGo)
- [ ] Citation system

### Phase 2: MoE Routing (3 days)
- [ ] Domain detection
- [ ] Expert model selection
- [ ] KV-cache optimization
- [ ] Dynamic routing logic

### Phase 3: Tree of Thought (1 week)
- [ ] Query decomposition
- [ ] Branch evaluation heuristics
- [ ] Recursive search with pruning
- [ ] Synthesis layer

### Phase 4: Safety & Jailbreak Defense (3 days)
- [ ] Input filtering
- [ ] Semantic manipulation detection
- [ ] Output content policy
- [ ] Multi-language normalization

### Phase 5: Integration (2 days)
- [ ] Update Secretary to use new pipeline
- [ ] Mode detection (fast vs research)
- [ ] Current event auto-trigger
- [ ] Testing & benchmarking

**Total**: ~2.5 weeks start to finish

---

## Tech Stack

**Search**:
- BM25S (Python, 500x faster)
- SentenceTransformers (semantic embeddings)
- CrossEncoder (reranking)

**Models**:
- dolphin-mistral (7B, fast, uncensored)
- dolphin-mixtral (8x7B, complex queries)
- deepseek-coder (code queries)
- qwen2.5 (synthesis)

**APIs**:
- Brave Search API (web search)
- Twitter/X API (real-time Grok-style)
- arXiv API (research papers)

**Infrastructure**:
- FastAPI (existing ICEBURG backend)
- Redis (KV-cache, caching)
- PostgreSQL + pgvector (semantic search storage)

---

## Competitive Analysis

| Feature | Perplexity | Grok | Claude | ICEBURG V2 |
|---------|-----------|------|--------|------------|
| **Speed** | ~15s | <5s | ~10s | **<10s** ✅ |
| **Citations** | ✅ | ❌ | ❌ | ✅ |
| **Uncensored** | ❌ | ✅ | ❌ | **✅** |
| **Recursive Research** | ❌ | ❌ | ✅ (MCP) | **✅** (ToT) |
| **Local/Private** | ❌ | ❌ | ❌ | **✅** |
| **Real-time Data** | ✅ | ✅ | ❌ | ✅ |
| **Multi-hop Queries** | ❌ | ❌ | ✅ | **✅** (ToT) |
| **Cost** | $20/mo | $15/mo | $20/mo | **$0** (local) |

---

## Conclusion

**ICEBURG Secretary V2** = **Perplexity + Grok + underground hacks + local control**

- **Faster than Perplexity** (BM25S, MoE)
- **Smarter than Grok** (Tree of Thought, recursive)
- **Uncensored** (dolphin models, no corporate filters)
- **Fully local** (your hardware, your rules)
- **Production-ready in 2.5 weeks**

Next step: Build Phase 1 (Hybrid Search).
