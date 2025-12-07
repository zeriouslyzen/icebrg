"""
Secretary Agent Knowledge Base
Self-updating knowledge base that extracts, stores, and retrieves knowledge from conversations.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SecretaryKnowledgeBase:
    """
    Knowledge base manager for Secretary agent.
    
    Features:
    - Automatic knowledge extraction from conversations
    - Topic-based markdown file creation
    - User persona storage and updates
    - Topic indexes and cross-references
    - Vector store integration for semantic search
    """
    
    def __init__(self, cfg, knowledge_base_dir: Optional[Path] = None):
        """
        Initialize knowledge base.
        
        Args:
            cfg: ICEBURG configuration
            knowledge_base_dir: Base directory for knowledge base (default: data/secretary_knowledge)
        """
        self.cfg = cfg
        
        if knowledge_base_dir is None:
            base_dir = Path(cfg.data_dir) if hasattr(cfg, 'data_dir') else Path("./data")
            knowledge_base_dir = base_dir / "secretary_knowledge"
        
        self.kb_dir = Path(knowledge_base_dir)
        self.topics_dir = self.kb_dir / "topics"
        self.personas_dir = self.kb_dir / "personas"
        self.indexes_dir = self.kb_dir / "indexes"
        self.summaries_dir = self.kb_dir / "summaries"
        
        # Create directories
        for dir_path in [self.topics_dir, self.personas_dir, self.indexes_dir, self.summaries_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.kb_dir / "metadata.json"
        self._load_metadata()
        
        # Initialize vector store integration
        self.memory = None
        try:
            from ..memory.unified_memory import UnifiedMemory
            self.memory = UnifiedMemory(cfg)
            logger.info("✅ Knowledge base vector store initialized")
        except Exception as e:
            logger.warning(f"Could not initialize vector store: {e}. Continuing without vector search.")
    
    def _load_metadata(self):
        """Load knowledge base metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}. Starting fresh.")
                self.metadata = {
                    "version": "1.0",
                    "created": datetime.now().isoformat(),
                    "topics": {},
                    "personas": {},
                    "last_updated": datetime.now().isoformat()
                }
        else:
            self.metadata = {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "topics": {},
                "personas": {},
                "last_updated": datetime.now().isoformat()
            }
            self._save_metadata()
    
    def _save_metadata(self):
        """Save knowledge base metadata."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save metadata: {e}")
    
    def extract_knowledge(self, conversation: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract knowledge from conversation.
        
        Args:
            conversation: Conversation text (query + response)
            user_id: Optional user ID for persona updates
            
        Returns:
            Dictionary with extracted knowledge
        """
        from ..providers.factory import provider_factory
        
        provider = provider_factory(self.cfg)
        model_to_use = getattr(self.cfg, "surveyor_model", None) or getattr(self.cfg, "primary_model", None) or "gemini-2.0-flash-exp"
        
        extraction_prompt = f"""Analyze this conversation and extract key knowledge:

Conversation:
{conversation}

Extract and return JSON with:
{{
    "topics": ["topic1", "topic2", ...],  // Main topics discussed
    "facts": ["fact1", "fact2", ...],     // Key facts or information
    "preferences": {{                     // User preferences mentioned
        "key": "value"
    }},
    "expertise": ["area1", "area2", ...], // Areas of expertise mentioned
    "important": true/false               // Whether this is important knowledge
}}

Return ONLY valid JSON, no other text."""

        try:
            response = provider.chat_complete(
                model=model_to_use,
                prompt=extraction_prompt,
                system="You are a knowledge extraction system. Extract structured knowledge from conversations.",
                temperature=0.3,
                options={"max_tokens": 500},
            )
            
            # Parse JSON response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            knowledge = json.loads(response)
            
            # Add metadata
            knowledge["extracted_at"] = datetime.now().isoformat()
            knowledge["user_id"] = user_id
            
            return knowledge
            
        except Exception as e:
            logger.warning(f"Error extracting knowledge: {e}")
            # Return minimal structure
            return {
                "topics": [],
                "facts": [],
                "preferences": {},
                "expertise": [],
                "important": False,
                "extracted_at": datetime.now().isoformat(),
                "user_id": user_id
            }
    
    def store_topic(self, topic: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Store topic in knowledge base.
        
        Args:
            topic: Topic name
            content: Topic content
            metadata: Additional metadata
        """
        # Sanitize topic name for filename
        safe_topic = re.sub(r'[^\w\s-]', '', topic).strip()
        safe_topic = re.sub(r'[-\s]+', '-', safe_topic)
        topic_file = self.topics_dir / f"{safe_topic}.md"
        
        # Read existing content if file exists
        existing_content = ""
        if topic_file.exists():
            try:
                with open(topic_file, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
            except Exception as e:
                logger.warning(f"Could not read existing topic file: {e}")
        
        # Merge content
        if existing_content:
            # Add new content with timestamp
            new_section = f"\n\n---\n\n## Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{content}"
            full_content = existing_content + new_section
        else:
            # Create new topic file
            full_content = f"# {topic}\n\n{content}"
            if metadata:
                full_content += f"\n\n## Metadata\n\n```json\n{json.dumps(metadata, indent=2)}\n```"
        
        # Write topic file
        try:
            with open(topic_file, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            # Update metadata
            if topic not in self.metadata["topics"]:
                self.metadata["topics"][topic] = {
                    "created": datetime.now().isoformat(),
                    "file": str(topic_file.relative_to(self.kb_dir))
                }
            self.metadata["topics"][topic]["last_updated"] = datetime.now().isoformat()
            self.metadata["topics"][topic]["metadata"] = metadata or {}
            self._save_metadata()
            
            logger.info(f"✅ Stored topic: {topic}")
            
        except Exception as e:
            logger.error(f"Error storing topic: {e}")
    
    def query_knowledge(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query knowledge base semantically.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of relevant knowledge entries
        """
        results = []
        
        # Try vector search first
        if self.memory:
            try:
                vector_results = self.memory.search(
                    namespace="secretary_knowledge",
                    query=query,
                    k=k
                )
                for result in vector_results:
                    results.append({
                        "type": "vector",
                        "content": result.get("document", ""),
                        "metadata": result.get("metadata", {}),
                        "score": result.get("distance", 0.0)
                    })
            except Exception as e:
                logger.debug(f"Vector search failed: {e}")
        
        # Fallback to topic file search
        if not results:
            query_lower = query.lower()
            for topic, topic_info in self.metadata.get("topics", {}).items():
                if query_lower in topic.lower():
                    topic_file = self.kb_dir / topic_info.get("file", "")
                    if topic_file.exists():
                        try:
                            with open(topic_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            results.append({
                                "type": "topic",
                                "topic": topic,
                                "content": content[:500],  # First 500 chars
                                "metadata": topic_info.get("metadata", {})
                            })
                        except Exception as e:
                            logger.debug(f"Could not read topic file: {e}")
        
        return results[:k]
    
    def update_persona(self, user_id: str, traits: Dict[str, Any]):
        """
        Update user persona.
        
        Args:
            user_id: User ID
            traits: Persona traits to update
        """
        persona_file = self.personas_dir / f"{user_id}.json"
        
        # Load existing persona
        persona = {}
        if persona_file.exists():
            try:
                with open(persona_file, 'r', encoding='utf-8') as f:
                    persona = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load persona: {e}")
        
        # Update persona
        persona.update(traits)
        persona["last_updated"] = datetime.now().isoformat()
        if "created" not in persona:
            persona["created"] = datetime.now().isoformat()
        
        # Save persona
        try:
            with open(persona_file, 'w', encoding='utf-8') as f:
                json.dump(persona, f, indent=2, ensure_ascii=False)
            
            # Update metadata
            if user_id not in self.metadata["personas"]:
                self.metadata["personas"][user_id] = {
                    "created": datetime.now().isoformat()
                }
            self.metadata["personas"][user_id]["last_updated"] = datetime.now().isoformat()
            self._save_metadata()
            
            logger.info(f"✅ Updated persona for user: {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating persona: {e}")
    
    def get_persona(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user persona.
        
        Args:
            user_id: User ID
            
        Returns:
            Persona dictionary or None
        """
        persona_file = self.personas_dir / f"{user_id}.json"
        
        if persona_file.exists():
            try:
                with open(persona_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load persona: {e}")
        
        return None
    
    def build_index(self):
        """
        Build topic index and cross-references.
        
        Creates an index file that maps topics to files and cross-references.
        """
        index = {
            "topics": {},
            "cross_references": {},
            "last_built": datetime.now().isoformat()
        }
        
        # Build topic index
        for topic, topic_info in self.metadata.get("topics", {}).items():
            index["topics"][topic] = {
                "file": topic_info.get("file", ""),
                "created": topic_info.get("created", ""),
                "last_updated": topic_info.get("last_updated", "")
            }
        
        # Build cross-references (simple: topics that share words)
        topic_words = {}
        for topic in index["topics"].keys():
            words = set(topic.lower().split())
            topic_words[topic] = words
        
        for topic1, words1 in topic_words.items():
            for topic2, words2 in topic_words.items():
                if topic1 != topic2:
                    common_words = words1.intersection(words2)
                    if common_words:
                        if topic1 not in index["cross_references"]:
                            index["cross_references"][topic1] = []
                        if topic2 not in index["cross_references"][topic1]:
                            index["cross_references"][topic1].append({
                                "topic": topic2,
                                "common_words": list(common_words)
                            })
        
        # Save index
        index_file = self.indexes_dir / "topic_index.json"
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ Built topic index")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
    
    def process_conversation(self, query: str, response: str, user_id: Optional[str] = None, conversation_id: Optional[str] = None):
        """
        Process a conversation and extract/store knowledge.
        
        Args:
            query: User query
            response: Assistant response
            user_id: User ID
            conversation_id: Conversation ID
        """
        conversation = f"User: {query}\nAssistant: {response}"
        
        # Extract knowledge
        knowledge = self.extract_knowledge(conversation, user_id)
        
        # Store topics
        for topic in knowledge.get("topics", []):
            if topic:
                # Create topic content from facts
                facts = [f for f in knowledge.get("facts", []) if topic.lower() in f.lower() or f.lower() in topic.lower()]
                content = "\n".join([f"- {fact}" for fact in facts[:5]])  # Top 5 facts
                if not content:
                    content = f"Information about {topic} from conversation."
                
                self.store_topic(
                    topic=topic,
                    content=content,
                    metadata={
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "extracted_at": knowledge.get("extracted_at")
                    }
                )
        
        # Update persona if preferences or expertise mentioned
        if knowledge.get("preferences") or knowledge.get("expertise"):
            persona_updates = {}
            if knowledge.get("preferences"):
                persona_updates["preferences"] = knowledge["preferences"]
            if knowledge.get("expertise"):
                persona_updates["expertise"] = knowledge["expertise"]
            
            if user_id and persona_updates:
                self.update_persona(user_id, persona_updates)
        
        # Store in vector store for semantic search
        if self.memory and knowledge.get("important", False):
            try:
                summary = f"Topics: {', '.join(knowledge.get('topics', []))}. Facts: {' '.join(knowledge.get('facts', [])[:3])}"
                self.memory.index_texts(
                    namespace="secretary_knowledge",
                    texts=[summary],
                    metadatas=[{
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "topics": knowledge.get("topics", []),
                        "timestamp": knowledge.get("extracted_at")
                    }]
                )
            except Exception as e:
                logger.debug(f"Could not store in vector store: {e}")
        
        # Rebuild index periodically (every 10 topics)
        if len(self.metadata.get("topics", {})) % 10 == 0:
            self.build_index()
        
        return knowledge

