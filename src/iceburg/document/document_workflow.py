from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DocumentElement:
    """Represents a document element (manuscript, abstract, methods)"""
    document_type: str
    title: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime


class DocumentWorkflow:
    """Generates publication-ready documents and manuscripts"""
    
    def __init__(self):
        self.document_counter = 0
        self.available_types = [
            "manuscript", "abstract", "methods", "pre_registration", "citation"
        ]
    
    def generate_manuscript(self, title: str, research_data: Dict[str, Any]) -> DocumentElement:
        """Generate a complete research manuscript"""
        
        manuscript_id = self._generate_document_id()
        
        # Extract key components from research data
        abstract = research_data.get("abstract", "")
        methods = research_data.get("methods", {})
        results = research_data.get("results", {})
        conclusions = research_data.get("conclusions", [])
        
        manuscript_content = {
            "manuscript_id": manuscript_id,
            "title": title,
            "abstract": abstract,
            "keywords": research_data.get("keywords", []),
            "sections": {
                "introduction": self._generate_introduction(research_data),
                "methods": self._generate_methods_section(methods),
                "results": self._generate_results_section(results),
                "discussion": self._generate_discussion_section(research_data),
                "conclusions": self._generate_conclusions_section(conclusions)
            },
            "references": research_data.get("references", []),
            "format": "Publication-ready manuscript",
            "word_count": self._estimate_word_count(research_data)
        }
        
        metadata = {
            "generator": "Iceberg Document Workflow",
            "version": "1.0",
            "recommendations": [
                "Review and edit for clarity",
                "Ensure proper citation format",
                "Check journal-specific requirements",
                "Verify figure and table references"
            ]
        }
        
        return DocumentElement(
            document_type="manuscript",
            title=title,
            content=manuscript_content,
            metadata=metadata,
            timestamp=datetime.utcnow()
        )
    
    def generate_abstract(self, research_data: Dict[str, Any]) -> DocumentElement:
        """Generate a publication-ready abstract"""
        
        abstract_id = self._generate_document_id()
        
        # Create structured abstract
        abstract_content = {
            "abstract_id": abstract_id,
            "background": research_data.get("background", ""),
            "objective": research_data.get("objective", ""),
            "methods": research_data.get("methods_summary", ""),
            "results": research_data.get("key_results", ""),
            "conclusions": research_data.get("conclusions", ""),
            "word_count": len(research_data.get("abstract", "").split()),
            "format": "Structured abstract (Background, Objective, Methods, Results, Conclusions)"
        }
        
        metadata = {
            "generator": "Iceberg Document Workflow",
            "version": "1.0",
            "recommendations": [
                "Keep within journal word limits",
                "Highlight key findings",
                "Use clear, concise language",
                "Include relevant keywords"
            ]
        }
        
        return DocumentElement(
            document_type="abstract",
            title="Research Abstract",
            content=abstract_content,
            metadata=metadata,
            timestamp=datetime.utcnow()
        )
    
    def generate_pre_registration(self, study_design: Dict[str, Any]) -> DocumentElement:
        """Generate a study pre-registration document"""
        
        prereg_id = self._generate_document_id()
        
        prereg_content = {
            "preregistration_id": prereg_id,
            "study_title": study_design.get("title", ""),
            "hypothesis": study_design.get("hypothesis", ""),
            "methods": {
                "participants": study_design.get("participants", {}),
                "materials": study_design.get("materials", []),
                "procedure": study_design.get("procedure", []),
                "analysis_plan": study_design.get("analysis_plan", {})
            },
            "power_analysis": study_design.get("power_analysis", {}),
            "timeline": study_design.get("timeline", ""),
            "format": "OSF/ClinicalTrials.gov compatible"
        }
        
        metadata = {
            "generator": "Iceberg Document Workflow",
            "version": "1.0",
            "recommendations": [
                "Submit to appropriate registry",
                "Include all planned analyses",
                "Specify stopping rules",
                "Document any deviations"
            ]
        }
        
        return DocumentElement(
            document_type="pre_registration",
            title="Study Pre-registration",
            content=prereg_content,
            metadata=metadata,
            timestamp=datetime.utcnow()
        )
    
    def generate_citation(self, research_data: Dict[str, Any], format_type: str = "APA") -> str:
        """Generate citations in various formats"""
        
        authors = research_data.get("authors", ["Author"])
        title = research_data.get("title", "Title")
        year = research_data.get("year", datetime.utcnow().year)
        journal = research_data.get("journal", "Journal")
        
        if format_type == "APA":
            citation = f"{', '.join(authors)} ({year}). {title}. {journal}."
        elif format_type == "MLA":
            citation = f"{', '.join(authors)}. \"{title}.\" {journal}, {year}."
        elif format_type == "Chicago":
            citation = f"{', '.join(authors)}. \"{title}.\" {journal} ({year})."
        else:
            citation = f"{', '.join(authors)} ({year}). {title}. {journal}."
        
        return citation
    
    def create_publication_package(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a complete publication package"""
        
        package = {
            "manuscript": None,
            "abstract": None,
            "pre_registration": None,
            "citations": [],
            "recommendations": []
        }
        
        # Generate main manuscript
        if "title" in research_data:
            package["manuscript"] = self.generate_manuscript(
                research_data["title"], research_data
            )
        
        # Generate abstract
        package["abstract"] = self.generate_abstract(research_data)
        
        # Generate pre-registration if study design exists
        if "study_design" in research_data:
            package["pre_registration"] = self.generate_pre_registration(
                research_data["study_design"]
            )
        
        # Generate citations in multiple formats
        citation_formats = ["APA", "MLA", "Chicago"]
        for fmt in citation_formats:
            citation = self.generate_citation(research_data, fmt)
            package["citations"].append({"format": fmt, "citation": citation})
        
        # Add publication recommendations
        package["recommendations"] = [
            "Review manuscript for clarity and accuracy",
            "Ensure all figures and tables are referenced",
            "Check journal submission requirements",
            "Verify citation format matches journal style",
            "Consider pre-print publication for early dissemination"
        ]
        
        return package
    
    def _generate_introduction(self, research_data: Dict[str, Any]) -> str:
        """Generate introduction section"""
        background = research_data.get("background", "")
        objective = research_data.get("objective", "")
        
        intro = f"""
        {background}
        
        The objective of this study is to {objective}.
        """
        return intro.strip()
    
    def _generate_methods_section(self, methods: Dict[str, Any]) -> str:
        """Generate methods section"""
        if not methods:
            return "Methods section to be developed based on study design."
        
        methods_text = "Methods:\n"
        for key, value in methods.items():
            methods_text += f"- {key}: {value}\n"
        
        return methods_text.strip()
    
    def _generate_results_section(self, results: Dict[str, Any]) -> str:
        """Generate results section"""
        if not results:
            return "Results section to be populated with experimental findings."
        
        results_text = "Results:\n"
        for key, value in results.items():
            results_text += f"- {key}: {value}\n"
        
        return results_text.strip()
    
    def _generate_discussion_section(self, research_data: Dict[str, Any]) -> str:
        """Generate discussion section"""
        implications = research_data.get("implications", [])
        
        if not implications:
            return "Discussion section to be developed based on results and implications."
        
        discussion = "Discussion:\n"
        for i, implication in enumerate(implications, 1):
            discussion += f"{i}. {implication}\n"
        
        return discussion.strip()
    
    def _generate_conclusions_section(self, conclusions: List[str]) -> str:
        """Generate conclusions section"""
        if not conclusions:
            return "Conclusions section to be developed based on research findings."
        
        conclusions_text = "Conclusions:\n"
        for i, conclusion in enumerate(conclusions, 1):
            conclusions_text += f"{i}. {conclusion}\n"
        
        return conclusions_text.strip()
    
    def _estimate_word_count(self, research_data: Dict[str, Any]) -> int:
        """Estimate manuscript word count"""
        base_count = 0
        for key in ["abstract", "background", "objective"]:
            if key in research_data:
                base_count += len(research_data[key].split())
        
        # Estimate additional sections
        estimated_total = base_count * 5  # Rough estimate
        return estimated_total
    
    def _generate_document_id(self) -> str:
        """Generate unique document ID"""
        self.document_counter += 1
        return f"doc_{self.document_counter}_{int(datetime.utcnow().timestamp())}"
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and capabilities"""
        return {
            "status": "operational",
            "available_types": self.available_types,
            "total_documents_generated": self.document_counter,
            "capabilities": [
                "Research manuscript generation",
                "Abstract creation",
                "Pre-registration documents",
                "Citation formatting",
                "Publication package creation"
            ]
        }


# Example usage and testing
if __name__ == "__main__":
    # Create document workflow
    workflow = DocumentWorkflow()
    
    # Test with sample research data
    sample_research = {
        "title": "Consciousness Emergence in Complex Systems",
        "authors": ["AI Researcher", "Neuroscientist"],
        "year": 2025,
        "journal": "Nature Neuroscience",
        "abstract": "This study investigates how consciousness emerges from complex neural interactions.",
        "background": "Consciousness remains one of the most challenging problems in neuroscience.",
        "objective": "understand the mechanisms of consciousness emergence",
        "methods_summary": "Computational modeling and theoretical analysis",
        "key_results": "Consciousness emerges through complex system interactions",
        "conclusions": "Emergence theory provides new insights into consciousness",
        "keywords": ["consciousness", "emergence", "complex systems", "neuroscience"],
        "implications": [
            "New understanding of consciousness mechanisms",
            "Applications to artificial intelligence",
            "Insights into brain function"
        ],
        "study_design": {
            "title": "Consciousness Emergence Study",
            "hypothesis": "Consciousness emerges from complex neural interactions",
            "participants": {"type": "computational models", "count": "multiple"},
            "materials": ["Neural network models", "Complexity analysis tools"],
            "procedure": ["Model creation", "Interaction analysis", "Emergence detection"],
            "analysis_plan": {"method": "Complexity metrics", "threshold": "Emergence detection"}
        }
    }
    
    # Generate publication package
    package = workflow.create_publication_package(sample_research)
    
    print("Document Workflow Status:")
    print(workflow.get_workflow_status())
    
    print("\nGenerated Documents:")
    print(f"Manuscript: {'Yes' if package['manuscript'] else 'No'}")
    print(f"Abstract: {'Yes' if package['abstract'] else 'No'}")
    print(f"Pre-registration: {'Yes' if package['pre_registration'] else 'No'}")
    print(f"Citations: {len(package['citations'])}")
    
    print("\nSample Citation (APA):")
    if package['citations']:
        print(package['citations'][0]['citation'])
