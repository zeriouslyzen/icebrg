#!/usr/bin/env python3
"""
Test Truth-Finding Capabilities
Tests suppression detection and information archaeology
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.truth.suppression_detector import SuppressionDetector
from iceburg.truth.information_archaeology import InformationArchaeology
from iceburg.research.insight_generator import InsightGenerator


async def test_truth_finding():
    """Test truth-finding capabilities"""
    print("\n" + "="*70)
    print("TRUTH-FINDING TEST")
    print("="*70 + "\n")
    
    # Create sample documents with suppression indicators
    documents = [
        {
            "id": "doc1",
            "content": "Research on quantum computing was classified for 20 years before public release. The military had working quantum computers in 2000, but this was not disclosed until 2020.",
            "metadata": {
                "creation_date": "2000-01-01",
                "release_date": "2020-01-01",
                "source": "military_research",
                "classification": "top_secret"
            }
        },
        {
            "id": "doc2",
            "content": "Internal military report on AI capabilities that contradicts public narratives. The report states that AI systems achieved human-level performance in 2010, contradicting public claims that this was achieved in 2023.",
            "metadata": {
                "source": "internal_military_report",
                "classification": "restricted",
                "publication_date": "2010-06-15"
            }
        },
        {
            "id": "doc3",
            "content": "Research on free energy devices was suppressed through classification delays and funding misdirection. The technology was developed in 1995 but remains classified.",
            "metadata": {
                "creation_date": "1995-01-01",
                "release_date": "2095-01-01",
                "source": "government_research",
                "classification": "classified"
            }
        }
    ]
    
    # Test suppression detection
    print("Testing Suppression Detection...")
    suppression_detector = SuppressionDetector()
    suppression_result = suppression_detector.detect_suppression(documents)
    
    print(f"\nDocuments Analyzed: {len(documents)}")
    print(f"Suppression Detected: {suppression_result.get('suppression_detected', False)}")
    print(f"Suppression Score: {suppression_result.get('overall_suppression_score', 0.0):.2f}")
    print(f"Findings: {len(suppression_result.get('details', []))}")
    
    if suppression_result.get('details'):
        print("\nSuppression Findings:")
        for i, detail in enumerate(suppression_result.get('details', [])[:3], 1):
            print(f"  {i}. Document {detail.get('doc_id')}:")
            for finding in detail.get('findings', [])[:2]:
                print(f"     - {finding.get('type')}: {finding.get('score', 0):.2f}")
    
    # Test information archaeology
    print("\n\nTesting Information Archaeology...")
    archaeology = InformationArchaeology()
    recovered = await archaeology.recover_knowledge(documents)
    
    print(f"Recovered Information: {len(recovered)} items")
    for i, item in enumerate(recovered[:2], 1):
        print(f"  {i}. From document {item.get('original_doc_id')}")
        print(f"     Content: {item.get('reconstructed_content', '')[:100]}...")
    
    # Test insight generation
    print("\n\nTesting Insight Generation...")
    insight_generator = InsightGenerator()
    insights = insight_generator.generate_insights(
        query="What suppressed knowledge exists about quantum computing and AI capabilities?",
        documents=documents,
        domain="truth_finding"
    )
    
    print(f"Insights Generated: {len(insights.get('insights', []))}")
    print(f"Breakthroughs: {len(insights.get('breakthroughs', []))}")
    print(f"Suppression Detected: {insights.get('suppression_detected', False)}")
    
    if insights.get('insights'):
        print("\nKey Insights:")
        for i, insight in enumerate(insights.get('insights', [])[:3], 1):
            print(f"  {i}. {insight.get('type', 'insight')}: {insight.get('description', '')[:80]}...")
    
    print("\n" + "="*70)
    print("TRUTH-FINDING TEST COMPLETE")
    print("="*70 + "\n")
    
    return {
        "suppression_detected": suppression_result.get('suppression_detected', False),
        "suppression_score": suppression_result.get('overall_suppression_score', 0.0),
        "recovered_items": len(recovered),
        "insights_count": len(insights.get('insights', [])),
        "breakthroughs": len(insights.get('breakthroughs', []))
    }


if __name__ == "__main__":
    asyncio.run(test_truth_finding())

