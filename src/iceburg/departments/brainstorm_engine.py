"""
ICEBURG Brainstorm Engine

Advanced brainstorming system that enables autonomous scaling of thinking capabilities
through distributed intelligence and collaborative problem-solving.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
import random

class BrainstormMode(Enum):
    DIVERGENT = "divergent"  # Generate many ideas
    CONVERGENT = "convergent"  # Refine and select best ideas
    COLLABORATIVE = "collaborative"  # Multi-agent collaboration
    ITERATIVE = "iterative"  # Iterative refinement

class IdeaType(Enum):
    SOLUTION = "solution"
    APPROACH = "approach"
    STRATEGY = "strategy"
    INNOVATION = "innovation"
    OPTIMIZATION = "optimization"

@dataclass
class BrainstormIdea:
    idea_id: str
    content: str
    idea_type: IdeaType
    confidence: float
    source_agent: str
    tags: List[str]
    dependencies: List[str] = None
    refinement_count: int = 0
    collaboration_score: float = 0.0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class BrainstormSession:
    session_id: str
    problem: str
    context: Dict[str, Any]
    mode: BrainstormMode
    participants: List[str]
    ideas: List[BrainstormIdea]
    session_goals: List[str]
    start_time: float
    end_time: Optional[float] = None
    status: str = "active"
    
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

class BrainstormEngine:
    """
    Advanced brainstorming engine that enables:
    - Autonomous idea generation and refinement
    - Multi-agent collaborative thinking
    - Scalable department coordination
    - Advanced problem-solving strategies
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, BrainstormSession] = {}
        self.idea_pool: Dict[str, BrainstormIdea] = {}
        self.collaboration_network: Dict[str, List[str]] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.knowledge_base: Dict[str, Any] = {}
        
    def start_brainstorm_session(self, problem: str, context: Dict[str, Any] = None,
                               mode: BrainstormMode = BrainstormMode.COLLABORATIVE,
                               participants: List[str] = None,
                               goals: List[str] = None) -> BrainstormSession:
        """Start a new brainstorming session"""
        session_id = f"brainstorm_{uuid.uuid4().hex[:8]}"
        
        session = BrainstormSession(
            session_id=session_id,
            problem=problem,
            context=context or {},
            mode=mode,
            participants=participants or [],
            ideas=[],
            session_goals=goals or [],
            start_time=time.time()
        )
        
        self.active_sessions[session_id] = session
        return session
    
    def generate_ideas_autonomously(self, session_id: str, num_ideas: int = 10) -> List[BrainstormIdea]:
        """Generate ideas autonomously using various strategies"""
        if session_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[session_id]
        ideas = []
        
        for i in range(num_ideas):
            idea = self._generate_single_idea(session, i)
            if idea:
                ideas.append(idea)
                session.ideas.append(idea)
                self.idea_pool[idea.idea_id] = idea
        
        return ideas
    
    def _generate_single_idea(self, session: BrainstormSession, iteration: int) -> BrainstormIdea:
        """Generate a single idea using various brainstorming techniques"""
        idea_id = f"idea_{uuid.uuid4().hex[:8]}"
        
        # Select brainstorming technique based on mode and iteration
        technique = self._select_technique(session.mode, iteration)
        
        # Generate idea content
        content = self._apply_technique(technique, session.problem, session.context)
        
        # Determine idea type and confidence
        idea_type = self._classify_idea_type(content)
        confidence = self._calculate_confidence(content, session)
        
        # Generate tags
        tags = self._extract_tags(content)
        
        idea = BrainstormIdea(
            idea_id=idea_id,
            content=content,
            idea_type=idea_type,
            confidence=confidence,
            source_agent=f"autonomous_agent_{iteration % 3}",
            tags=tags
        )
        
        return idea
    
    def _select_technique(self, mode: BrainstormMode, iteration: int) -> str:
        """Select appropriate brainstorming technique"""
        techniques = {
            BrainstormMode.DIVERGENT: [
                "free_association", "random_word", "scamper", "six_thinking_hats"
            ],
            BrainstormMode.CONVERGENT: [
                "evaluation_matrix", "weighted_scoring", "consensus_building"
            ],
            BrainstormMode.COLLABORATIVE: [
                "round_robin", "silent_brainstorming", "idea_building"
            ],
            BrainstormMode.ITERATIVE: [
                "refinement_cycle", "feedback_loop", "progressive_enhancement"
            ]
        }
        
        available_techniques = techniques.get(mode, ["free_association"])
        return available_techniques[iteration % len(available_techniques)]
    
    def _apply_technique(self, technique: str, problem: str, context: Dict[str, Any]) -> str:
        """Apply a specific brainstorming technique"""
        technique_implementations = {
            "free_association": self._free_association,
            "random_word": self._random_word_technique,
            "scamper": self._scamper_technique,
            "six_thinking_hats": self._six_thinking_hats,
            "evaluation_matrix": self._evaluation_matrix,
            "round_robin": self._round_robin,
            "refinement_cycle": self._refinement_cycle
        }
        
        implementation = technique_implementations.get(technique, self._free_association)
        return implementation(problem, context)
    
    def _free_association(self, problem: str, context: Dict[str, Any]) -> str:
        """Free association brainstorming"""
        associations = [
            "What if we approached this completely differently?",
            "How would this work in a parallel universe?",
            "What would the opposite approach look like?",
            "How can we make this 10x better?",
            "What if we had unlimited resources?"
        ]
        
        return f"Free association for '{problem}': {random.choice(associations)}"
    
    def _random_word_technique(self, problem: str, context: Dict[str, Any]) -> str:
        """Random word technique for creative connections"""
        random_words = [
            "butterfly", "ocean", "mountain", "lightning", "crystal",
            "forest", "diamond", "volcano", "rainbow", "galaxy"
        ]
        
        word = random.choice(random_words)
        return f"Random word '{word}' applied to '{problem}': How can {word} inspire a solution?"
    
    def _scamper_technique(self, problem: str, context: Dict[str, Any]) -> str:
        """SCAMPER technique (Substitute, Combine, Adapt, Modify, Put to other uses, Eliminate, Reverse)"""
        scamper_actions = [
            "Substitute: What can we replace in the current approach?",
            "Combine: How can we merge different solutions?",
            "Adapt: What can we borrow from other domains?",
            "Modify: How can we change the current approach?",
            "Put to other uses: What else can this solution do?",
            "Eliminate: What can we remove or simplify?",
            "Reverse: How can we do the opposite?"
        ]
        
        action = random.choice(scamper_actions)
        return f"SCAMPER for '{problem}': {action}"
    
    def _six_thinking_hats(self, problem: str, context: Dict[str, Any]) -> str:
        """Six Thinking Hats technique"""
        hats = [
            "White Hat (Facts): What are the objective facts about this problem?",
            "Red Hat (Emotions): How do we feel about this problem?",
            "Black Hat (Caution): What are the potential problems and risks?",
            "Yellow Hat (Optimism): What are the benefits and opportunities?",
            "Green Hat (Creativity): What creative solutions can we generate?",
            "Blue Hat (Control): How do we organize our thinking process?"
        ]
        
        hat = random.choice(hats)
        return f"Six Thinking Hats for '{problem}': {hat}"
    
    def _evaluation_matrix(self, problem: str, context: Dict[str, Any]) -> str:
        """Evaluation matrix for idea assessment"""
        criteria = ["feasibility", "impact", "cost", "time", "risk"]
        return f"Evaluation matrix for '{problem}': Assess against {', '.join(criteria)}"
    
    def _round_robin(self, problem: str, context: Dict[str, Any]) -> str:
        """Round-robin collaborative brainstorming"""
        return f"Round-robin for '{problem}': Collaborative idea building with multiple perspectives"
    
    def _refinement_cycle(self, problem: str, context: Dict[str, Any]) -> str:
        """Iterative refinement approach"""
        return f"Refinement cycle for '{problem}': Iterative improvement of existing ideas"
    
    def _classify_idea_type(self, content: str) -> IdeaType:
        """Classify the type of idea based on content"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["solution", "solve", "fix"]):
            return IdeaType.SOLUTION
        elif any(word in content_lower for word in ["approach", "method", "way"]):
            return IdeaType.APPROACH
        elif any(word in content_lower for word in ["strategy", "plan", "roadmap"]):
            return IdeaType.STRATEGY
        elif any(word in content_lower for word in ["innovative", "creative", "novel"]):
            return IdeaType.INNOVATION
        elif any(word in content_lower for word in ["optimize", "improve", "enhance"]):
            return IdeaType.OPTIMIZATION
        else:
            return IdeaType.SOLUTION
    
    def _calculate_confidence(self, content: str, session: BrainstormSession) -> float:
        """Calculate confidence score for an idea"""
        base_confidence = 0.5
        
        # Length bonus
        length_bonus = min(len(content) / 100, 0.3)
        
        # Specificity bonus
        specificity_bonus = 0.1 if any(word in content.lower() for word in ["specific", "concrete", "detailed"]) else 0
        
        # Innovation bonus
        innovation_bonus = 0.2 if any(word in content.lower() for word in ["innovative", "creative", "novel"]) else 0
        
        return min(base_confidence + length_bonus + specificity_bonus + innovation_bonus, 1.0)
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract relevant tags from idea content"""
        tag_keywords = {
            "technical": ["code", "algorithm", "system", "architecture"],
            "creative": ["design", "art", "visual", "aesthetic"],
            "business": ["revenue", "market", "customer", "profit"],
            "social": ["community", "user", "human", "social"],
            "innovative": ["breakthrough", "revolutionary", "cutting-edge", "advanced"]
        }
        
        tags = []
        content_lower = content.lower()
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def collaborate_on_ideas(self, session_id: str, idea_ids: List[str]) -> List[BrainstormIdea]:
        """Enable collaboration on specific ideas"""
        if session_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[session_id]
        collaborated_ideas = []
        
        for idea_id in idea_ids:
            if idea_id in self.idea_pool:
                original_idea = self.idea_pool[idea_id]
                collaborated_idea = self._collaborate_on_idea(original_idea, session)
                collaborated_ideas.append(collaborated_idea)
        
        return collaborated_ideas
    
    def _collaborate_on_idea(self, original_idea: BrainstormIdea, session: BrainstormSession) -> BrainstormIdea:
        """Collaborate on a specific idea to improve it"""
        # Create a refined version
        refined_content = f"Collaborative refinement of: {original_idea.content}"
        
        # Add collaboration insights
        collaboration_insights = [
            "How can we combine this with other ideas?",
            "What are the potential improvements?",
            "How can we make this more practical?",
            "What are the implementation challenges?"
        ]
        
        refined_content += f" Collaboration insights: {random.choice(collaboration_insights)}"
        
        # Create new idea with collaboration
        collaborated_idea = BrainstormIdea(
            idea_id=f"collab_{original_idea.idea_id}",
            content=refined_content,
            idea_type=original_idea.idea_type,
            confidence=min(original_idea.confidence + 0.1, 1.0),
            source_agent="collaborative_agent",
            tags=original_idea.tags + ["collaborative"],
            dependencies=[original_idea.idea_id],
            refinement_count=original_idea.refinement_count + 1,
            collaboration_score=0.8
        )
        
        return collaborated_idea
    
    def scale_brainstorming(self, session_id: str, scale_factor: int) -> List[BrainstormIdea]:
        """Scale up brainstorming by generating more ideas with different approaches"""
        if session_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[session_id]
        scaled_ideas = []
        
        # Generate ideas with different techniques
        for i in range(scale_factor):
            # Vary the brainstorming mode
            modes = [BrainstormMode.DIVERGENT, BrainstormMode.CONVERGENT, BrainstormMode.COLLABORATIVE]
            mode = modes[i % len(modes)]
            
            # Generate ideas with this mode
            ideas = self.generate_ideas_autonomously(session_id, 3)
            scaled_ideas.extend(ideas)
        
        return scaled_ideas
    
    def synthesize_solutions(self, session_id: str) -> Dict[str, Any]:
        """Synthesize all ideas into comprehensive solutions"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        if not session.ideas:
            return {"synthesis": "No ideas to synthesize"}
        
        # Group ideas by type
        ideas_by_type = {}
        for idea in session.ideas:
            idea_type = idea.idea_type.value
            if idea_type not in ideas_by_type:
                ideas_by_type[idea_type] = []
            ideas_by_type[idea_type].append(idea)
        
        # Find best ideas in each category
        best_ideas = {}
        for idea_type, ideas in ideas_by_type.items():
            best_idea = max(ideas, key=lambda x: x.confidence)
            best_ideas[idea_type] = best_idea
        
        # Generate synthesis
        synthesis = {
            "problem": session.problem,
            "total_ideas": len(session.ideas),
            "ideas_by_type": {k: len(v) for k, v in ideas_by_type.items()},
            "best_ideas": best_ideas,
            "synthesis_recommendation": self._generate_synthesis_recommendation(best_ideas),
            "implementation_roadmap": self._generate_implementation_roadmap(best_ideas),
            "collaboration_insights": self._analyze_collaboration_patterns(session.ideas)
        }
        
        return synthesis
    
    def _generate_synthesis_recommendation(self, best_ideas: Dict[str, BrainstormIdea]) -> str:
        """Generate a synthesis recommendation from best ideas"""
        if not best_ideas:
            return "No ideas available for synthesis"
        
        recommendations = []
        for idea_type, idea in best_ideas.items():
            recommendations.append(f"{idea_type.title()}: {idea.content[:100]}...")
        
        return f"Synthesis recommendation: Combine {' + '.join(recommendations)}"
    
    def _generate_implementation_roadmap(self, best_ideas: Dict[str, BrainstormIdea]) -> List[str]:
        """Generate implementation roadmap from best ideas"""
        roadmap = [
            "Phase 1: Initial analysis and planning",
            "Phase 2: Prototype development and testing",
            "Phase 3: Full implementation and deployment",
            "Phase 4: Monitoring and optimization"
        ]
        
        return roadmap
    
    def _analyze_collaboration_patterns(self, ideas: List[BrainstormIdea]) -> Dict[str, Any]:
        """Analyze collaboration patterns in the ideas"""
        collaboration_analysis = {
            "total_collaborations": len([i for i in ideas if i.collaboration_score > 0]),
            "average_collaboration_score": sum(i.collaboration_score for i in ideas) / len(ideas) if ideas else 0,
            "most_collaborative_ideas": [i.idea_id for i in ideas if i.collaboration_score > 0.5],
            "refinement_frequency": sum(i.refinement_count for i in ideas) / len(ideas) if ideas else 0
        }
        
        return collaboration_analysis
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a brainstorming session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "problem": session.problem,
            "mode": session.mode.value,
            "participants": len(session.participants),
            "total_ideas": len(session.ideas),
            "session_duration": session.duration(),
            "status": session.status,
            "goals": session.session_goals
        }
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a brainstorming session and return summary"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        session.end_time = time.time()
        session.status = "completed"
        
        # Generate final synthesis
        synthesis = self.synthesize_solutions(session_id)
        
        return {
            "session_summary": self.get_session_status(session_id),
            "final_synthesis": synthesis,
            "session_completed": True
        }
