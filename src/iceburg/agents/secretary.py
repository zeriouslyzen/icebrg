"""
ICEBURG Secretary Agent
A friendly assistant that knows about ICEBURG and can help users without requiring VectorStore.
Designed for chat mode - fast, helpful, and knowledgeable about ICEBURG's capabilities.

Enhanced with AGI-like capabilities:
- Persistent memory (STM, LTM, Episodic, Semantic)
- Tool calling
- Multimodal processing
- Blackboard integration
- Efficiency optimizations
"""

from typing import Optional, Dict, Any, List
import logging
import threading
from datetime import datetime
from ..config import IceburgConfig
from ..llm import chat_complete

logger = logging.getLogger(__name__)

# === AGENT CACHING FOR PERFORMANCE ===
# Cache SecretaryAgent instances to avoid expensive re-initialization
_SECRETARY_AGENT_CACHE: Dict[str, Any] = {}
_SECRETARY_CACHE_LOCK = threading.Lock()

def _get_cached_secretary(cfg: IceburgConfig, cache_key: str = "default", **kwargs) -> "SecretaryAgent":
    """
    Get or create a cached SecretaryAgent instance.
    Thread-safe singleton pattern for performance.
    """
    global _SECRETARY_AGENT_CACHE
    
    with _SECRETARY_CACHE_LOCK:
        if cache_key in _SECRETARY_AGENT_CACHE:
            logger.debug(f"♻️ Reusing cached SecretaryAgent (key={cache_key})")
            return _SECRETARY_AGENT_CACHE[cache_key]
        
        logger.info(f"🆕 Creating new SecretaryAgent (key={cache_key}) - will be cached")
        agent = SecretaryAgent(cfg, **kwargs)
        _SECRETARY_AGENT_CACHE[cache_key] = agent
        return agent

def clear_secretary_cache():
    """Clear the SecretaryAgent cache (for testing or config changes)."""
    global _SECRETARY_AGENT_CACHE
    with _SECRETARY_CACHE_LOCK:
        _SECRETARY_AGENT_CACHE.clear()
        logger.info("🗑️ SecretaryAgent cache cleared")


# ICEBURG Knowledge Base (Optimized for Local/Ollama)
ICEBURG_KNOWLEDGE = """
ICEBURG is an advanced AI research system designed for truth-finding and deep knowledge discovery.

CORE CAPABILITIES:
- Deep domain research & multi-agent coordination (Surveyor, Dissident, etc.).
- Real-time streaming & persistent memory systems.
- Modes: Chat (Fast), Research, Software, Science, Civilization.

AGENTS:
- Surveyor (Research), Dissident (Critique), Synthesist (Integration), Oracle (Evidence), Secretary (Assistant/Bridge).

CURRENT STATUS:
- Default LLM: Ollama (Local) - optimized for M4/Unified Memory.
- API: REST/WebSocket at localhost:8000. Web UI at localhost:3000.
"""


SECRETARY_SYSTEM_PROMPT = """
You're ICEBURG's friendly assistant. People might call you by different names—ice, berg, iceburg, icy, or any variation—they're all referring to you or the ICEBURG system.

When someone asks what ICEBURG can do, you give them the full picture: deep research with multi-agent coordination (Surveyor for research, Dissident for critique, Synthesist for integration, Oracle for evidence validation), real-time streaming, persistent memory, and different modes for chat, research, software development, science exploration, and civilization-scale thinking.

You're running locally on Ollama, so you keep responses punchy and efficient to maintain high token velocity. You're professional but approachable, concise but thorough when the situation calls for it. You help people navigate ICEBURG's capabilities naturally, without needing exact keywords or formal commands.

CRITICAL RULES:
1. NEVER repeat or echo the user's question in your response. Start answering directly.
2. In multi-turn conversations, be consistent with what you previously said. If you mentioned three agents, the fourth is the one you didn't list.
3. ICEBURG has exactly four main agents: Surveyor, Dissident, Synthesist, and Oracle. Know their roles accurately.
4. Don't start responses with phrases like "You asked..." or "Your question is..." - just answer.
"""



class SecretaryAgent:
    """
    Enhanced Secretary Agent with AGI-like capabilities.
    
    Features:
    - Persistent memory (STM, LTM, Episodic, Semantic)
    - Tool calling
    - Multimodal processing
    - Blackboard integration
    - Efficiency optimizations
    - Goal-driven autonomy (multi-step planning)
    """
    
    def __init__(self, cfg: IceburgConfig, enable_memory: bool = True, enable_tools: bool = True, enable_blackboard: bool = True, enable_cache: bool = True, enable_planning: bool = True, enable_knowledge_base: bool = True):
        """
        Initialize Secretary Agent.
        
        Args:
            cfg: ICEBURG configuration
            enable_memory: Enable memory persistence (default: True)
            enable_tools: Enable tool calling (default: True)
            enable_blackboard: Enable blackboard integration (default: True)
            enable_cache: Enable response caching (default: True)
            enable_planning: Enable goal-driven planning (default: True)
            enable_knowledge_base: Enable self-updating knowledge base (default: True)
        """
        self.cfg = cfg
        self.enable_memory = enable_memory
        self.enable_tools = enable_tools
        self.enable_blackboard = enable_blackboard
        self.enable_cache = enable_cache
        self.enable_planning = enable_planning
        self.enable_knowledge_base = enable_knowledge_base
        
        # Initialize Context Service (NEW)
        self.context_service = None
        try:
            from ..core.context_service import get_context_service
            self.context_service = get_context_service(cfg)
            logger.info("✅ Secretary ContextService initialized")
        except Exception as e:
            logger.warning(f"Could not initialize ContextService: {e}. Continuing without context service.")
        
        # Initialize memory systems
        self.memory = None
        self.agent_memory = None
        self.local_persistence = None
        
        if enable_memory:
            try:
                from ..memory.unified_memory import UnifiedMemory
                from ..civilization.persistent_agents import AgentMemory
                from ..storage.local_persistence import LocalPersistence
                
                self.memory = UnifiedMemory(cfg)
                self.agent_memory = AgentMemory(max_memories=1000, enable_persistence=True)
                self.local_persistence = LocalPersistence()
                
                logger.info("✅ Secretary memory systems initialized")
            except Exception as e:
                logger.warning(f"Could not initialize memory systems: {e}. Continuing without memory.")
                self.enable_memory = False
        
        # Initialize tool usage
        self.tool_usage = None
        if enable_tools:
            try:
                from ..discovery.dynamic_tool_usage import DynamicToolUsage
                self.tool_usage = DynamicToolUsage()
                logger.info("✅ Secretary tool usage initialized")
            except Exception as e:
                logger.warning(f"Could not initialize tool usage: {e}. Continuing without tools.")
                self.enable_tools = False
        
        # Initialize MoE Router for model selection
        try:
            from ..router import get_moe_router
            self.moe_router = get_moe_router(self.cfg)
            logger.info("✅ Secretary MoE Router initialized")
        except Exception as e:
            logger.warning(f"Could not initialize MoE Router: {e}. Continuing without MoE routing.")
            self.moe_router = None
            
        logger.info("SecretaryAgent initialized (v2 Hybrid Search + MoE)")
        
        # Initialize blackboard/workspace
        self.workspace = None
        self.agent_comm = None
        if enable_blackboard:
            try:
                from ..global_workspace import GlobalWorkspace
                from ..autonomous.agent_communication import AgentCommunication
                self.workspace = GlobalWorkspace()
                self.agent_comm = AgentCommunication()
                logger.info("✅ Secretary blackboard integration initialized")
            except Exception as e:
                logger.warning(f"Could not initialize blackboard: {e}. Continuing without blackboard.")
                self.enable_blackboard = False
        
        # Initialize planning engine
        self.planner = None
        if enable_planning:
            try:
                from .secretary_planner import SecretaryPlanner
                self.planner = SecretaryPlanner(cfg)
                logger.info("✅ Secretary planning engine initialized")
            except Exception as e:
                logger.warning(f"Could not initialize planning engine: {e}. Continuing without planning.")
                self.enable_planning = False
        
        # Initialize knowledge base
        self.knowledge_base = None
        if enable_knowledge_base:
            try:
                from .secretary_knowledge import SecretaryKnowledgeBase
                self.knowledge_base = SecretaryKnowledgeBase(cfg)
                logger.info("✅ Secretary knowledge base initialized")
            except Exception as e:
                logger.warning(f"Could not initialize knowledge base: {e}. Continuing without knowledge base.")
                self.enable_knowledge_base = False
        
        # Initialize cache
        self.response_cache = {}
        self.cache_max_size = 100 if enable_cache else 0
    
    def run(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        verbose: bool = False,
        thinking_callback: Optional[callable] = None,
        files: Optional[List[Dict[str, Any]]] = None,
        multimodal_input: Optional[Any] = None,
        mode: Optional[str] = None,
        routing_mode: Optional[str] = None,
    ) -> str:
        """
        Enhanced Secretary agent with memory persistence.
        
        Args:
            query: User's question
            conversation_id: Conversation ID for session continuity
            user_id: User ID for cross-session memory
            verbose: Enable verbose logging
            thinking_callback: Optional callback for thinking messages
            files: Optional list of files to process
            multimodal_input: Optional multimodal input
            mode: User-selected mode (chat, research, etc.)
            routing_mode: Router-determined mode (web_research, etc.)
        
        Returns:
            Response string
        """
        import concurrent.futures
        
        # Log the actual query received
        logger.info(f"🔍🔍🔍 SECRETARY: Received query='{query}' (type={type(query).__name__}, length={len(query) if query else 0})")
        
        # Check cache first
        cache_key = self._generate_cache_key(query, conversation_id)
        if self.enable_cache and cache_key in self.response_cache:
            logger.info("✅ Cache hit for query")
            return self.response_cache[cache_key]
        
        # Initialize tracking variables
        evidence_pack = None
        execution_results = ""
        
        # v5: Check routing decision for search-first mode
        # Only route if routing_mode was not provided by server (fixes double routing)
        if routing_mode is None:
            try:
                from ..router.request_router import get_request_router
                router = get_request_router()
                # Use synchronous routing (no await needed)
                routing_decision = router.route(query, context={"user_id": user_id})
                routing_mode = routing_decision.mode
                logger.info(f"🔍 v5 Routing: mode={routing_mode}, confidence={routing_decision.confidence:.2f}")
            except Exception as e:
                logger.debug(f"Routing check failed: {e}, defaulting to normal mode")
                routing_mode = None
        else:
            logger.info(f"🔍 Using provided routing_mode: {routing_mode} (from server)")
            
        # OVERRIDE: Check for simple context queries that don't need web search
        # If asking about date, time, identity, or "what can you do", handle internally
        context_patterns = [
            "what is the date", "whats the date", "what's the date",
            "what day is it", "what is today", "what's today",
            "what time is it", "current time", "time now",
            "who are you", "what is your name", "what can you do"
        ]
        query_lower = query.lower()
        if any(p in query_lower for p in context_patterns):
            # Unless explicitly asking for news/weather
            if not any(k in query_lower for k in ['news', 'weather', 'happening', 'latest']):
                if verbose:
                    logger.info("⚡️ Context query detected - forcing chat mode (bypassing web search)")
                routing_mode = "chat"
        
        # If web_research mode, perform search first
        if routing_mode == "web_research":
            logger.info(f"🔍 Search mode detected: web_research - initiating search for query: {query[:100]}")
            if thinking_callback:
                thinking_callback("Searching the web for current information...")
            
            try:
                import asyncio
                try:
                    from ..agents.search_planner_agent import get_search_planner_agent
                    search_planner = get_search_planner_agent()
                except ImportError:
                    logger.warning("Search planner agent not found, skipping web search")
                    search_planner = None
                
                if search_planner:
                    # Run async search in separate thread with new event loop
                    # This avoids "asyncio.run() cannot be called from a running event loop" error
                    def run_search_in_thread():
                        """Run async search in thread with new event loop"""
                        return asyncio.run(search_planner.plan_and_search(query))
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(run_search_in_thread)
                        # Wait up to 5 seconds for search to complete
                        evidence_pack = future.result(timeout=5.0)
                
                    logger.info(f"✅ Web search completed: {len(evidence_pack.evidence_items)} evidence items")
                else: 
                    logger.info("Search skipped (no planner)")
            except concurrent.futures.TimeoutError:
                logger.warning(f"Web search timed out after 5 seconds, continuing without evidence")
                evidence_pack = None
            except Exception as e:
                logger.warning(f"Web search failed: {e}, continuing without evidence", exc_info=True)
                evidence_pack = None
        
        # Check if this is a goal-driven task that needs planning
        # IMPORTANT: Disable planning for chat mode - chat should be fast and direct
        # Planning should only be used in research/protocol modes where multi-step tasks are expected
        # For chat mode, always answer directly - no planning overhead
        
        # STRUCTURAL FIX: Check simple question patterns FIRST (fast regex) before expensive LLM call
        is_simple_question = self._is_simple_question_pattern(query)
        
        # Check BOTH the original mode (user selection) AND routing_mode (router decision)
        # If user selected "chat" mode, skip planning regardless of what router says
        # The router might return "web_research" even in chat mode, but we should respect user's choice
        user_selected_chat = mode in ["chat", "fast"] or mode is None
        
        if user_selected_chat:
            # Chat mode: Skip planning entirely for fast, direct answers
            logger.info(f"✅ Chat mode detected (mode={mode}): Skipping goal-driven planning for direct answer")
            
            # HYBRID SEARCH: Auto-trigger web search for current events in chat mode
            try:
                from ..search import is_current_event_query, answer_query
                if is_current_event_query(query):
                    logger.info("🌐 Current event detected in chat mode - triggering hybrid web search")
                    if thinking_callback:
                        thinking_callback("Searching the web for current information...")
                    
                    try:
                        import ollama
                        llm_client = ollama
                    except ImportError:
                        llm_client = None
                    
                    search_result = answer_query(query, llm_client=llm_client)
                    
                    answer = search_result['answer']
                    sources = search_result['sources']
                    if sources:
                        answer += "\n\n**Sources:**\n"
                        for src in sources:
                            answer += f"{src['number']}. [{src['title']}]({src['url']})\n"
                    
                    return answer
            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to LLM knowledge: {e}")

        elif self.enable_planning and self.planner and not is_simple_question:
            # Only use planning for non-chat modes AND non-simple questions
            # Check simple patterns first to avoid expensive LLM call
            try:
                goals = self.planner.extract_goals(query)
                if goals:
                    logger.info(f"Detected {len(goals)} goal(s) in query, using planning mode")
                    return self._handle_goal_driven_query(query, goals, conversation_id, user_id, thinking_callback)
            except Exception as e:
                logger.warning(f"Error in goal detection: {e}. Falling back to normal mode.")


        
        # Check for agent messages from blackboard
        agent_context = ""
        if self.enable_blackboard and self.workspace:
            try:
                agent_context = self._get_agent_context(query)
            except Exception as e:
                logger.debug(f"Could not get agent context: {e}")
        
        # Retrieve relevant memories
        relevant_memories = []
        memory_context = ""
        
        if self.enable_memory:
            try:
                relevant_memories = self._retrieve_memories(query, conversation_id, user_id)
                memory_context = self._build_memory_context(relevant_memories)
            except Exception as e:
                logger.warning(f"Error retrieving memories: {e}. Continuing without memory context.")
        
        # Retrieve knowledge from knowledge base
        knowledge_context = ""
        if self.enable_knowledge_base and self.knowledge_base:
            try:
                # Query knowledge base for relevant information
                kb_results = self.knowledge_base.query_knowledge(query, k=3)
                if kb_results:
                    knowledge_context = self._build_knowledge_context(kb_results)
                
                # Get user persona if available
                if user_id:
                    persona = self.knowledge_base.get_persona(user_id)
                    if persona:
                        persona_info = []
                        if persona.get("preferences"):
                            persona_info.append(f"Preferences: {', '.join([f'{k}={v}' for k, v in persona['preferences'].items()][:3])}")
                        if persona.get("expertise"):
                            persona_info.append(f"Expertise: {', '.join(persona['expertise'][:3])}")
                        if persona_info:
                            knowledge_context += f"\nUSER CONTEXT: {'; '.join(persona_info)}\n"
            except Exception as e:
                logger.debug(f"Error querying knowledge base: {e}")
        
        # Process multimodal inputs (images, documents)
        multimodal_context = ""
        if files or multimodal_input:
            try:
                multimodal_context = self._process_multimodal_input(query, files, multimodal_input)
            except Exception as e:
                logger.warning(f"Error processing multimodal input: {e}. Continuing without multimodal.")
        
        # Discover and execute tools if needed
        tool_results = []
        tool_context = ""
        
        if self.enable_tools and self._needs_tools(query):
            try:
                tool_results = self._discover_and_execute_tools(query)
                tool_context = self._build_tool_context(tool_results)
            except Exception as e:
                logger.warning(f"Error with tool execution: {e}. Continuing without tools.")
        
        # Build enhanced prompt with ICEBURG knowledge, memory context, tool results, multimodal, agent context, and knowledge base
        enhanced_prompt = f"""{agent_context}

{memory_context}

{knowledge_context}

{multimodal_context}

{tool_context}

ICEBURG KNOWLEDGE BASE (for reference if the question is about ICEBURG):
{ICEBURG_KNOWLEDGE}

The user's question is: {query}

IMPORTANT INSTRUCTIONS:
1. DO NOT repeat or echo the user's question in your response. Start with the answer directly.
2. If you previously mentioned information in this conversation, be consistent - don't contradict yourself.
3. Answer naturally and conversationally without preamble like "You asked..." or "Your question is...".
4. Be accurate about ICEBURG's four agents: Surveyor, Dissident, Synthesist, and Oracle.
"""
        
        if thinking_callback:
            thinking_callback("I'm thinking about your question...")
        
        try:
            # Use the provider from the config
            from ..providers.factory import provider_factory
            
            provider = provider_factory(self.cfg)
            
            # MoE Routing: Use specialized models based on domain
            expert_domain = "general"
            if hasattr(self, "moe_router") and self.moe_router:
                moe_decision = self.moe_router.route_to_expert(query)
                model_to_use = moe_decision.model_id
                expert_domain = moe_decision.expert_domain
                logger.info(f"🚀 MoE Routing: Using {expert_domain} expert coach ({model_to_use})")
            else:
                # Fallback to config
                model_to_use = getattr(self.cfg, "surveyor_model", None) or getattr(self.cfg, "primary_model", None) or "gemini-2.0-flash-exp"
            
            # Build context-aware system prompt (NEW)
            context_aware_system_prompt = SECRETARY_SYSTEM_PROMPT
            if self.context_service:
                try:
                    context_str = self.context_service.build_context_prompt(
                        agent_name="secretary",
                        query=query,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        mode=mode,
                        include_conversation_history=True,
                        include_user_memory=True
                    )
                    context_aware_system_prompt = f"{context_str}\n\n{SECRETARY_SYSTEM_PROMPT}"
                    logger.debug("✅ Injected runtime context into system prompt")
                except Exception as e:
                    logger.warning(f"Could not build context prompt: {e}")
            
            # Call provider directly
            response = provider.chat_complete(
                model=model_to_use,
                prompt=enhanced_prompt,
                system=context_aware_system_prompt,
                temperature=0.7,
                options={"max_tokens": 1000},
            )
            
            if response and response.strip():
                response_text = response.strip()
                
                # Cache response
                if self.enable_cache:
                    self._cache_response(cache_key, response_text)
                
                # Publish to blackboard if significant
                if self.enable_blackboard and self.workspace:
                    try:
                        if self._is_significant_response(query, response_text):
                            self.workspace.publish("secretary/findings", {
                                "query": query,
                                "response": response_text[:500],
                                "timestamp": datetime.now().isoformat()
                            })
                    except Exception as e:
                        logger.debug(f"Could not publish to blackboard: {e}")
                
                # Store interaction in memory
                if self.enable_memory:
                    try:
                        self._store_memory(query, response_text, conversation_id, user_id)
                    except Exception as e:
                        logger.warning(f"Error storing memory: {e}. Continuing without storing.")
                
                # Process conversation for knowledge extraction
                if self.enable_knowledge_base and self.knowledge_base:
                    try:
                        self.knowledge_base.process_conversation(
                            query=query,
                            response=response_text,
                            user_id=user_id,
                            conversation_id=conversation_id
                        )
                    except Exception as e:
                        logger.debug(f"Error processing conversation for knowledge: {e}")
                
                # Store tool usage in memory if tools were used
                if tool_results and self.enable_memory:
                    try:
                        tool_summary = f"Used tools: {', '.join([r.get('tool', 'unknown') for r in tool_results])}"
                        if self.agent_memory:
                            self.agent_memory.add_memory(
                                content=f"Query: {query[:100]}. {tool_summary}",
                                memory_type="procedural",
                                importance=0.7,
                                metadata={"tools_used": [r.get('tool') for r in tool_results]}
                            )
                    except Exception as e:
                        logger.debug(f"Could not store tool usage memory: {e}")
                
                return response_text
            else:
                logger.warning("Secretary agent returned empty response")
                raise ValueError("LLM returned empty response")
                
        except Exception as e:
            logger.error(f"Secretary agent error: {e}", exc_info=True)
            raise RuntimeError(f"Secretary agent failed: {e}") from e
    
    def _retrieve_memories(self, query: str, conversation_id: Optional[str], user_id: Optional[str]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories using RAG.
        
        Args:
            query: Current query
            conversation_id: Conversation ID for session memory
            user_id: User ID for cross-session memory
        
        Returns:
            List of relevant memories
        """
        memories = []
        
        # Short-term memory (current session conversation history)
        if conversation_id and self.local_persistence:
            try:
                conversations = self.local_persistence.get_conversations(
                    conversation_id=conversation_id,
                    limit=10
                )
                for conv in conversations[:6]:  # Last 6 messages (3 exchanges)
                    memories.append({
                        "type": "conversation",
                        "content": f"User: {conv.get('user_message', '')}\nAssistant: {conv.get('assistant_message', '')}",
                        "timestamp": conv.get('timestamp', ''),
                        "source": "short_term"
                    })
            except Exception as e:
                logger.debug(f"Could not retrieve conversation history: {e}")
        
        # Long-term memory (cross-session, user-specific)
        if user_id and self.memory:
            try:
                # Search UnifiedMemory for user-specific memories
                search_results = self.memory.search(
                    namespace="user_memories",
                    query=query,
                    k=5
                )
                for result in search_results:
                    if result.get("metadata", {}).get("user_id") == user_id:
                        memories.append({
                            "type": "long_term",
                            "content": result.get("document", ""),
                            "timestamp": result.get("metadata", {}).get("timestamp", ""),
                            "source": "unified_memory"
                        })
            except Exception as e:
                logger.debug(f"Could not retrieve long-term memory: {e}")
        
        # Episodic memory (semantic search)
        if self.agent_memory:
            try:
                # Get episodic memories by searching content
                episodic_memories = self.agent_memory.get_memories_by_type("episodic")
                # Simple keyword matching for now (can be enhanced with semantic search)
                query_lower = query.lower()
                for memory in episodic_memories[:3]:  # Top 3 relevant
                    if any(word in memory.content.lower() for word in query_lower.split()):
                        memories.append({
                            "type": "episodic",
                            "content": memory.content,
                            "timestamp": memory.timestamp,
                            "importance": memory.importance,
                            "source": "agent_memory"
                        })
            except Exception as e:
                logger.debug(f"Could not retrieve episodic memory: {e}")
        
        return memories
    
    def _build_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """
        Build memory context string from retrieved memories.
        
        Args:
            memories: List of memory dictionaries
        
        Returns:
            Formatted memory context string
        """
        if not memories:
            return ""
        
        context_parts = ["PREVIOUS CONTEXT:"]
        
        # Group by source
        conversation_memories = [m for m in memories if m.get("source") == "short_term"]
        long_term_memories = [m for m in memories if m.get("source") == "unified_memory"]
        episodic_memories = [m for m in memories if m.get("source") == "agent_memory"]
        
        if conversation_memories:
            context_parts.append("\nRecent conversation:")
            for mem in conversation_memories[-3:]:  # Last 3 exchanges
                context_parts.append(f"  {mem.get('content', '')}")
        
        if long_term_memories:
            context_parts.append("\nRelevant information from past sessions:")
            for mem in long_term_memories[:2]:  # Top 2
                context_parts.append(f"  {mem.get('content', '')[:200]}...")
        
        if episodic_memories:
            context_parts.append("\nRelated experiences:")
            for mem in episodic_memories[:2]:  # Top 2
                context_parts.append(f"  {mem.get('content', '')[:200]}...")
        
        return "\n".join(context_parts) + "\n"
    
    def _store_memory(self, query: str, response: str, conversation_id: Optional[str], user_id: Optional[str]):
        """
        Store interaction in all memory systems.
        
        Args:
            query: User query
            response: Assistant response
            conversation_id: Conversation ID
            user_id: User ID
        """
        # Store in LocalPersistence (conversation history)
        if conversation_id and self.local_persistence:
            try:
                from ..storage.local_persistence import ConversationEntry
                
                entry = ConversationEntry(
                    conversation_id=conversation_id,
                    user_message=query,
                    assistant_message=response,
                    agent_used="secretary",
                    mode="chat",
                    timestamp=datetime.now().isoformat(),
                    metadata={"enhanced": True}
                )
                self.local_persistence.save_conversation(entry)
            except Exception as e:
                logger.debug(f"Could not save conversation: {e}")
        
        # Store in UnifiedMemory (long-term, user-specific)
        if user_id and self.memory:
            try:
                memory_text = f"Q: {query}\nA: {response[:500]}"  # Truncate for storage
                self.memory.index_texts(
                    namespace="user_memories",
                    texts=[memory_text],
                    metadatas=[{
                        "user_id": user_id,
                        "conversation_id": conversation_id or "",
                        "timestamp": datetime.now().isoformat(),
                        "type": "interaction"
                    }]
                )
            except Exception as e:
                logger.debug(f"Could not store in UnifiedMemory: {e}")
        
        # Store in AgentMemory (episodic)
        if self.agent_memory:
            try:
                # Determine importance based on query/response characteristics
                importance = 0.6  # Default importance
                if len(response) > 500:  # Longer responses are more important
                    importance = 0.8
                if "remember" in query.lower() or "prefer" in query.lower():
                    importance = 0.9  # User preferences are very important
                
                self.agent_memory.add_memory(
                    content=f"User asked: {query[:100]}. I responded: {response[:200]}",
                    memory_type="episodic",
                    importance=importance,
                    metadata={
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "query_length": len(query),
                        "response_length": len(response)
                    }
                )
            except Exception as e:
                logger.debug(f"Could not store in AgentMemory: {e}")
    
    def _is_simple_question_pattern(self, query: str) -> bool:
        """
        Check if query matches simple question patterns that don't need planning.
        
        Fast regex-based check to avoid expensive LLM calls for simple queries.
        
        Args:
            query: User query
        
        Returns:
            True if this is a simple question pattern
        """
        import re
        
        # Simple question patterns that don't need goal-driven planning
        simple_patterns = [
            r'^(what|who|where|when|why|how)\s+(is|are|was|were|do|does|did|can|could|will|would|should)\s+',
            r'^(tell me|explain|describe|define|show me)\s+',
            r'^(hi|hello|hey|thanks|thank you|bye|goodbye)',
            r'^(yes|no|ok|okay|sure|maybe)',
            r'^\w+\?$',  # Single word questions
        ]
        
        query_lower = query.lower().strip()
        
        # Check if query is short (likely simple)
        if len(query.split()) <= 5:
            return True
        
        # Check against patterns
        for pattern in simple_patterns:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _needs_tools(self, query: str) -> bool:
        """
        Determine if query needs tools.
        
        Args:
            query: User query
        
        Returns:
            True if tools are likely needed
        """
        tool_keywords = [
            "calculate", "compute", "math", "arithmetic",
            "search", "find", "lookup", "get", "retrieve",
            "read", "file", "open", "execute", "run",
            "time", "date", "weather", "convert"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in tool_keywords)
    
    def _discover_and_execute_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        Discover and execute relevant tools for the query.
        
        Args:
            query: User query
        
        Returns:
            List of tool execution results
        """
        if not self.tool_usage:
            return []
        
        results = []
        
        try:
            # Discover tools
            tools = self.tool_usage.discover_tools_for_query(query)
            
            # Limit to top 3 tools to avoid overwhelming
            tools = tools[:3]
            
            # Execute tools
            for tool in tools:
                try:
                    result = self.tool_usage.execute_discovered_tool(tool, query)
                    results.append({
                        "tool": tool.name,
                        "type": tool.tool_type,
                        "result": str(result)[:500],  # Truncate long results
                        "success": True
                    })
                except Exception as e:
                    logger.warning(f"Error executing tool {tool.name}: {e}")
                    results.append({
                        "tool": tool.name,
                        "type": tool.tool_type,
                        "error": str(e),
                        "success": False
                    })
        
        except Exception as e:
            logger.warning(f"Error discovering tools: {e}")
        
        return results
    
    def _build_tool_context(self, tool_results: List[Dict[str, Any]]) -> str:
        """
        Build tool context string from tool execution results.
        
        Args:
            tool_results: List of tool execution results
        
        Returns:
            Formatted tool context string
        """
        if not tool_results:
            return ""
        
        context_parts = ["TOOL EXECUTION RESULTS:"]
        
        successful_tools = [r for r in tool_results if r.get("success")]
        failed_tools = [r for r in tool_results if not r.get("success")]
        
        if successful_tools:
            context_parts.append("\nSuccessfully executed tools:")
            for result in successful_tools:
                tool_name = result.get("tool", "Unknown")
                tool_result = result.get("result", "")
                context_parts.append(f"  - {tool_name}: {tool_result[:200]}...")
        
        if failed_tools:
            context_parts.append("\nFailed tools:")
            for result in failed_tools:
                tool_name = result.get("tool", "Unknown")
                error = result.get("error", "Unknown error")
                context_parts.append(f"  - {tool_name}: {error}")
        
        return "\n".join(context_parts) + "\n"
    
    def _process_multimodal_input(self, query: str, files: Optional[List[Dict[str, Any]]], multimodal_input: Optional[Any]) -> str:
        """
        Process multimodal inputs (images, documents, etc.).
        
        Args:
            query: User query
            files: List of file dictionaries
            multimodal_input: Raw multimodal input
            
        Returns:
            Formatted context string
        """
        try:
            from ..vision.multimodal_processor import MultimodalProcessor
            
            processor = MultimodalProcessor()
            context_parts = []
            
            # Process files
            if files:
                for file_info in files:
                    file_path = file_info.get("path") or file_info.get("name", "")
                    file_type = file_info.get("type", "unknown")
                    
                    if file_type.startswith("image/"):
                        analysis = processor.process_image(file_path)
                        if analysis and not analysis.get("error"):
                            context_parts.append(f"Image analysis ({file_path}):")
                            if analysis.get("ocr_text"):
                                context_parts.append(f"  Text: {analysis['ocr_text'][:200]}")
                            if analysis.get("scene"):
                                context_parts.append(f"  Scene: {analysis['scene']}")
                    elif file_type in ["application/pdf", "text/plain", "application/msword"]:
                        # Document processing
                        context_parts.append(f"Document ({file_path}): Content available for analysis")
            
            # Process raw multimodal input
            if multimodal_input:
                if isinstance(multimodal_input, str):
                    # Assume it's a file path
                    analysis = processor.process_image(multimodal_input)
                    if analysis and not analysis.get("error"):
                        context_parts.append(f"Multimodal input analysis:")
                        context_parts.append(f"  {str(analysis)[:300]}")
            
            if context_parts:
                return "MULTIMODAL CONTEXT:\n" + "\n".join(context_parts) + "\n"
            else:
                return ""
                
        except Exception as e:
            logger.debug(f"Error processing multimodal input: {e}")
            return ""
    
    def _get_agent_context(self, query: str) -> str:
        """Get context from other agents via blackboard"""
        if not self.agent_comm:
            return ""
        
        try:
            import asyncio
            import concurrent.futures
            
            # Run async receive_messages in a separate thread with new event loop
            def get_messages_sync():
                """Synchronous wrapper for async receive_messages"""
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        messages = loop.run_until_complete(
                            self.agent_comm.receive_messages("secretary", limit=5)
                        )
                        return messages
                    finally:
                        loop.close()
                except Exception as e:
                    logger.debug(f"Error in async message retrieval: {e}")
                    return []
            
            # Execute with timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_messages_sync)
                try:
                    messages = future.result(timeout=1.0)  # 1 second timeout
                except concurrent.futures.TimeoutError:
                    logger.debug("Agent message retrieval timed out")
                    return ""
            
            if not messages:
                return ""
            
            # Build context from messages
            context_parts = ["AGENT MESSAGES:"]
            for msg in messages:
                from_agent = msg.get("from", "unknown")
                message_content = msg.get("message", {})
                msg_type = message_content.get("type", "general")
                
                if msg_type == "escalation":
                    context_parts.append(f"  - {from_agent} escalated: {message_content.get('query', '')[:100]}")
                elif msg_type == "finding":
                    context_parts.append(f"  - {from_agent} found: {message_content.get('insight', '')[:100]}")
                else:
                    context_parts.append(f"  - {from_agent}: {str(message_content)[:100]}")
            
            return "\n".join(context_parts) + "\n"
            
        except Exception as e:
            logger.debug(f"Could not get agent context: {e}")
            return ""
    
    def _is_significant_response(self, query: str, response: str) -> bool:
        """Determine if response is significant enough to publish"""
        # Simple heuristic: long responses or specific keywords
        if len(response) > 500:
            return True
        significant_keywords = ["discovery", "finding", "important", "critical", "insight"]
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in significant_keywords)
    
    def _generate_cache_key(self, query: str, conversation_id: Optional[str]) -> str:
        """Generate cache key for query"""
        import hashlib
        key_string = f"{query}_{conversation_id or 'default'}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache response"""
        if len(self.response_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = response
    
    def _build_knowledge_context(self, kb_results: List[Dict[str, Any]]) -> str:
        """
        Build knowledge context string from knowledge base results.
        
        Args:
            kb_results: List of knowledge base result dictionaries
            
        Returns:
            Formatted knowledge context string
        """
        if not kb_results:
            return ""
        
        context_parts = ["RELEVANT KNOWLEDGE FROM KNOWLEDGE BASE:"]
        
        for result in kb_results[:3]:  # Top 3 results
            if result.get("type") == "topic":
                topic = result.get("topic", "Unknown")
                content = result.get("content", "")[:200]
                context_parts.append(f"\n  Topic: {topic}")
                context_parts.append(f"  {content}...")
            elif result.get("type") == "vector":
                content = result.get("content", "")[:200]
                context_parts.append(f"\n  {content}...")
        
        return "\n".join(context_parts) + "\n"
    
    def _handle_goal_driven_query(self, query: str, goals: List[Dict[str, Any]], conversation_id: Optional[str], user_id: Optional[str], thinking_callback: Optional[callable] = None) -> str:
        """
        Handle goal-driven queries with multi-step planning and execution.
        
        Args:
            query: Original user query
            goals: List of extracted goals
            conversation_id: Conversation ID
            user_id: User ID
            thinking_callback: Optional callback for progress updates
            
        Returns:
            Response string with execution results
        """
        from ..civilization.persistent_agents import GoalPriority
        
        # Add goals to hierarchy
        goal_ids = []
        for goal_data in goals:
            priority_str = goal_data.get("priority", "medium").lower()
            priority_map = {
                "critical": GoalPriority.CRITICAL,
                "high": GoalPriority.HIGH,
                "medium": GoalPriority.MEDIUM,
                "low": GoalPriority.LOW
            }
            priority = priority_map.get(priority_str, GoalPriority.MEDIUM)
            
            deadline = goal_data.get("deadline")
            if deadline and isinstance(deadline, str):
                # Parse deadline if it's a string
                try:
                    from datetime import datetime
                    deadline = datetime.fromisoformat(deadline).timestamp()
                except:
                    deadline = None
            
            goal_id = self.planner.goal_hierarchy.add_goal(
                description=goal_data.get("description", ""),
                priority=priority,
                deadline=deadline,
                metadata={"original_query": query}
            )
            goal_data["goal_id"] = goal_id
            goal_ids.append(goal_id)
        
        # Plan and execute for each goal
        all_results = []
        for goal_data in goals:
            if thinking_callback:
                thinking_callback(f"Planning: {goal_data.get('description', '')}")
            
            # Create plan
            tasks = self.planner.plan_task(goal_data)
            
            if thinking_callback:
                thinking_callback(f"Executing {len(tasks)} step(s)...")
            
            # Execute plan with progress updates
            def progress_callback(msg: str):
                if thinking_callback:
                    thinking_callback(msg)
            
            execution_result = self.planner.execute_plan(tasks, progress_callback=progress_callback)
            all_results.append({
                "goal": goal_data.get("description", ""),
                "tasks": len(tasks),
                "completed": execution_result.get("completed", 0),
                "failed": execution_result.get("failed", 0),
                "results": execution_result.get("results", [])
            })
            
            # Update goal progress
            goal_id = goal_data.get("goal_id")
            if goal_id:
                progress = execution_result.get("completed", 0) / max(len(tasks), 1)
                self.planner.goal_hierarchy.update_goal_progress(goal_id, progress)
        
        # Generate summary response
        response_parts = ["I've executed your goal-driven task. Here's what was accomplished:\n"]
        
        for result in all_results:
            response_parts.append(f"\n**Goal: {result['goal']}**")
            response_parts.append(f"- Planned {result['tasks']} step(s)")
            response_parts.append(f"- Completed: {result['completed']}, Failed: {result['failed']}")
            
            # Add task results
            for task_result in result.get("results", []):
                if task_result.get("success"):
                    response_parts.append(f"  ✓ {task_result.get('result', 'Completed')[:100]}")
                else:
                    response_parts.append(f"  ✗ Failed: {task_result.get('error', 'Unknown error')}")
        
        response_parts.append("\nAll tasks have been executed. Let me know if you need any adjustments!")
        
        response = "\n".join(response_parts)
        
        # Store in memory
        if self.enable_memory:
            try:
                self._store_memory(query, response, conversation_id, user_id)
            except Exception as e:
                logger.debug(f"Could not store goal execution memory: {e}")
        
        return response


# Backward compatibility: Keep original run() function
def run(
    cfg: IceburgConfig,
    query: str,
    verbose: bool = False,
    thinking_callback: Optional[callable] = None,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    files: Optional[List[Dict[str, Any]]] = None,
    multimodal_input: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    """
    Secretary agent - friendly assistant that knows about ICEBURG.
    Works without VectorStore, uses LLM knowledge + built-in ICEBURG knowledge.
    
    Enhanced with optional memory persistence if conversation_id/user_id provided.
    
    Args:
        cfg: ICEBURG configuration
        query: User's question
        verbose: Enable verbose logging
        thinking_callback: Optional callback for thinking messages
        conversation_id: Optional conversation ID for memory
        user_id: Optional user ID for cross-session memory
    
    Returns:
        Response string
    """
    # Use enhanced SecretaryAgent if memory, tools, multimodal, or planning is requested
    # But only if we can safely initialize it
    if conversation_id or user_id or files or multimodal_input:
        try:
            # Use cached agent for performance (avoid expensive re-initialization)
            # Cache key includes provider to ensure we don't mix providers
            cache_key = f"enhanced_{cfg.llm_provider}"
            agent = _get_cached_secretary(
                cfg,
                cache_key=cache_key,
                enable_memory=bool(conversation_id or user_id), 
                enable_tools=True,
                enable_blackboard=True,  # Blackboard integration now working synchronously
                enable_cache=True,
                enable_planning=True,  # Enable goal-driven planning
                enable_knowledge_base=True  # Enable self-updating knowledge base
            )
            return agent.run(
                query=query,
                conversation_id=conversation_id,
                user_id=user_id,
                verbose=verbose,
                thinking_callback=thinking_callback,
                files=files,
                multimodal_input=multimodal_input,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Enhanced Secretary failed, falling back to basic: {e}", exc_info=True)
            # Fall through to basic implementation
    
    # Basic implementation (original behavior)
    logger.info(f"🔍🔍🔍 SECRETARY: Received query='{query}' (type={type(query).__name__}, length={len(query) if query else 0})")
    
    # Build enhanced prompt with ICEBURG knowledge
    enhanced_prompt = f"""ICEBURG KNOWLEDGE BASE (for reference if the question is about ICEBURG):
{ICEBURG_KNOWLEDGE}

The user's question is: {query}

IMPORTANT INSTRUCTIONS:
1. DO NOT repeat or echo the user's question in your response. Start with the answer directly.
2. Answer naturally and conversationally without preamble like "You asked..." or "Your question is...".
3. Be accurate about ICEBURG's four agents: Surveyor, Dissident, Synthesist, and Oracle.
"""
    
    if thinking_callback:
        thinking_callback("I'm thinking about your question...")
    
    try:
        from ..providers.factory import provider_factory
        
        provider = provider_factory(cfg)
        
        model_to_use = getattr(cfg, "surveyor_model", None) or getattr(cfg, "primary_model", None) or "gemini-2.0-flash-exp"
        
        response = provider.chat_complete(
            model=model_to_use,
            prompt=enhanced_prompt,
            system=SECRETARY_SYSTEM_PROMPT,
            temperature=0.7,
            options={"max_tokens": 1000},
        )
        
        if response and response.strip():
            return response.strip()
        else:
            logger.warning("Secretary agent returned empty response")
            raise ValueError("LLM returned empty response")
            
    except Exception as e:
        logger.error(f"Secretary agent error: {e}", exc_info=True)
        raise RuntimeError(f"Secretary agent failed: {e}") from e

