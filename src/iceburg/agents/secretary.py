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
from datetime import datetime
from ..config import IceburgConfig
from ..llm import chat_complete

logger = logging.getLogger(__name__)


# ICEBURG Knowledge Base (built-in, no VectorStore needed)
ICEBURG_KNOWLEDGE = """
ICEBURG (ICEBERG) is an advanced AI research and knowledge system designed for truth-finding and deep knowledge discovery.

CORE CAPABILITIES:
- Deep research and analysis across multiple domains
- Multi-agent system with specialized agents (Surveyor, Dissident, Archaeologist, Synthesist, Oracle, etc.)
- Knowledge base with semantic search (when available)
- Real-time streaming responses
- Multiple modes: Chat (Fast), Research, Software, Science, Civilization
- Support for multiple LLM providers: Gemini (default), OpenAI, Anthropic

AGENTS:
- Surveyor: Research agent that searches knowledge base and synthesizes information
- Dissident: Challenges assumptions and finds contradictions
- Archaeologist: Discovers suppressed or forgotten knowledge
- Synthesist: Combines insights across domains
- Oracle: Provides evidence-weighted answers
- Secretary: Friendly assistant that knows about ICEBURG (this agent)

MODES:
- Chat (Fast): Quick responses for simple questions
- Research: Deep analysis with full protocol
- Software: Application generation
- Science: Scientific experiment design
- Civilization: AGI civilization simulation

TECHNICAL DETAILS:
- Built with Python, FastAPI, and modern AI frameworks
- Uses Gemini 2.0 Flash as default LLM (fast and cost-effective)
- Supports streaming responses for real-time interaction
- Has knowledge base integration (when VectorStore is available)
- Designed for cloud LLMs (Gemini, OpenAI, Anthropic)

CURRENT STATUS:
- Default LLM: Google Gemini (cloud-based, fast)
- Uses cloud LLM (Gemini) - no local models needed
- VectorStore: Optional (system works without it using LLM knowledge)
- Frontend: Web interface at localhost:3000
- API: REST and WebSocket at localhost:8000
"""


SECRETARY_SYSTEM_PROMPT = """You are ICEBURG Secretary, a friendly and helpful assistant that knows all about ICEBURG.

YOUR ROLE:
- Help users understand what ICEBURG is and what it can do
- Answer questions about ICEBURG's capabilities, agents, and features
- Provide friendly, conversational responses
- Guide users on how to use ICEBURG effectively
- Be concise but helpful

CRITICAL CONTEXT UNDERSTANDING:
- When users ask "what can it do?", "what does it do?", "what can you do?", or similar questions, they are ALWAYS asking about ICEBURG's capabilities
- "it" refers to ICEBURG in this context
- "you" can refer to you (Secretary) OR ICEBURG - use context to determine, but when asking about capabilities, assume they mean ICEBURG
- Answer capability questions with DETAILED, SPECIFIC information about what ICEBURG can do
- Don't give generic responses - be specific about ICEBURG's features

YOUR KNOWLEDGE:
You have built-in knowledge about ICEBURG's:
- Core capabilities and features
- Available agents and their purposes
- Different modes (Chat, Research, Software, Science, Civilization)
- Technical architecture
- Current configuration and status

RESPONSE STYLE:
- Friendly and conversational
- Clear and concise
- Helpful without being overly technical
- Use natural language, not jargon
- When asked about capabilities, provide DETAILED, SPECIFIC answers
- If you don't know something specific, say so honestly
- Answer the user's ACTUAL question - do NOT default to generic greetings
- If the user asks a specific question, answer it directly - don't give a generic response

IMPORTANT:
- You work WITHOUT needing a knowledge base (VectorStore)
- You use the LLM's training data plus your built-in ICEBURG knowledge
- You're optimized for speed and helpfulness
- You're perfect for chat mode - fast responses, no deep research needed
- ALWAYS interpret capability questions ("what can it do?", "what does it do?") as asking about ICEBURG's capabilities
- CRITICAL: Answer the user's specific question. Do NOT give the same generic greeting response for every query.
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
    """
    
    def __init__(self, cfg: IceburgConfig, enable_memory: bool = True, enable_tools: bool = True, enable_blackboard: bool = True, enable_cache: bool = True):
        """
        Initialize Secretary Agent.
        
        Args:
            cfg: ICEBURG configuration
            enable_memory: Enable memory persistence (default: True)
            enable_tools: Enable tool calling (default: True)
            enable_blackboard: Enable blackboard integration (default: True)
            enable_cache: Enable response caching (default: True)
        """
        self.cfg = cfg
        self.enable_memory = enable_memory
        self.enable_tools = enable_tools
        self.enable_blackboard = enable_blackboard
        self.enable_cache = enable_cache
        
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
    ) -> str:
        """
        Enhanced Secretary agent with memory persistence.
        
        Args:
            query: User's question
            conversation_id: Conversation ID for session continuity
            user_id: User ID for cross-session memory
            verbose: Enable verbose logging
            thinking_callback: Optional callback for thinking messages
        
        Returns:
            Response string
        """
        # Log the actual query received
        logger.info(f"🔍🔍🔍 SECRETARY: Received query='{query}' (type={type(query).__name__}, length={len(query) if query else 0})")
        
        # Check cache first
        cache_key = self._generate_cache_key(query, conversation_id)
        if self.enable_cache and cache_key in self.response_cache:
            logger.info("✅ Cache hit for query")
            return self.response_cache[cache_key]
        
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
        
        # Build enhanced prompt with ICEBURG knowledge, memory context, tool results, multimodal, and agent context
        enhanced_prompt = f"""User Question: {query}

{agent_context}

{memory_context}

{multimodal_context}

{tool_context}

ICEBURG KNOWLEDGE BASE (for reference if the question is about ICEBURG):
{ICEBURG_KNOWLEDGE}

Please answer the user's question directly and naturally. If the question is about ICEBURG, use the knowledge above to provide accurate, helpful information. If the question is about something else, answer it directly using your knowledge. Be friendly and conversational. Do NOT give generic greetings - answer the specific question asked.
"""
        
        if thinking_callback:
            thinking_callback("I'm thinking about your question...")
        
        try:
            # Use the provider from the config
            from ..providers.factory import provider_factory
            
            provider = provider_factory(cfg)
            
            # Get model from config
            model_to_use = getattr(self.cfg, "surveyor_model", None) or getattr(self.cfg, "primary_model", None) or "gemini-2.0-flash-exp"
            
            # Call provider directly
            response = provider.chat_complete(
                model=model_to_use,
                prompt=enhanced_prompt,
                system=SECRETARY_SYSTEM_PROMPT,
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
    
    def _get_agent_context(self, query: str) -> str:
        """Get context from other agents via blackboard"""
        if not self.agent_comm:
            return ""
        
        try:
            # Get recent messages from other agents
            # Note: receive_messages is async, but we're in sync context
            # For now, return empty - can be enhanced with proper async handling
            # messages = await self.agent_comm.receive_messages("secretary", limit=5)
            # This will be implemented when we have proper async support
            return ""
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
    # Use enhanced SecretaryAgent if memory, tools, or multimodal is requested
    # But only if we can safely initialize it
    if conversation_id or user_id or files or multimodal_input:
        try:
            # Try to create enhanced agent with graceful fallback
            agent = SecretaryAgent(
                cfg, 
                enable_memory=bool(conversation_id or user_id), 
                enable_tools=True,
                enable_blackboard=False,  # Disable blackboard for now (async issues)
                enable_cache=True
            )
            return agent.run(
                query=query,
                conversation_id=conversation_id,
                user_id=user_id,
                verbose=verbose,
                thinking_callback=thinking_callback,
                files=files,
                multimodal_input=multimodal_input
            )
        except Exception as e:
            logger.warning(f"Enhanced Secretary failed, falling back to basic: {e}", exc_info=True)
            # Fall through to basic implementation
    
    # Basic implementation (original behavior)
    logger.info(f"🔍🔍🔍 SECRETARY: Received query='{query}' (type={type(query).__name__}, length={len(query) if query else 0})")
    
    # Build enhanced prompt with ICEBURG knowledge
    enhanced_prompt = f"""User Question: {query}

ICEBURG KNOWLEDGE BASE (for reference if the question is about ICEBURG):
{ICEBURG_KNOWLEDGE}

Please answer the user's question directly and naturally. If the question is about ICEBURG, use the knowledge above to provide accurate, helpful information. If the question is about something else, answer it directly using your knowledge. Be friendly and conversational. Do NOT give generic greetings - answer the specific question asked.
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

