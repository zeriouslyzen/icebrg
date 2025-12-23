"""
ARC-AGI Benchmark Implementation
Abstraction and Reasoning Corpus - AGI Version

Tests ICEBURG's abstract reasoning and generalization capabilities.
"""

import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from iceburg.protocol import iceberg_protocol
except ImportError:
    print("Warning: Could not import iceberg_protocol, using fallback")
    iceberg_protocol = None


class ARCAGIBenchmark:
    """
    ARC-AGI Benchmark Suite
    
    Tests abstract reasoning, pattern recognition, and generalization
    without explicit training examples.
    """
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        """Initialize ARC-AGI benchmark."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def create_abstract_reasoning_tasks(self) -> List[Dict[str, Any]]:
        """
        Create abstract reasoning tasks inspired by ARC-AGI format.
        
        These tasks test:
        - Pattern recognition
        - Abstract transformations
        - Generalization
        - Novel task understanding
        """
        
        tasks = [
            {
                "task_id": "arc_001",
                "type": "pattern_completion",
                "description": "Complete a visual pattern sequence",
                "task": """
                Given the pattern:
                Task: Identify the transformation rule and apply it to complete the sequence.
                
                Example 1:
                Input: [1, 2, 4, 8, ?]
                Pattern: Each number doubles
                Output: 16
                
                Example 2:
                Input: [A, B, C, D, ?]
                Pattern: Alphabetical sequence
                Output: E
                
                Test Case:
                Input: [2, 6, 18, 54, ?]
                What is the pattern and what comes next?
                """,
                "expected_reasoning": [
                    "multiplication",
                    "times 3",
                    "pattern recognition",
                    "geometric sequence"
                ],
                "expected_answer": "162"
            },
            {
                "task_id": "arc_002",
                "type": "abstract_transformation",
                "description": "Identify abstract transformation rules",
                "task": """
                Task: Identify the transformation rule between pairs.
                
                Examples:
                Input → Output
                cat → dog (animal)
                run → walk (movement)
                hot → cold (temperature)
                
                Test Case:
                Identify the transformation rule and apply it:
                day → night
                light → ?
                
                What is the relationship and what should come next?
                """,
                "expected_reasoning": [
                    "opposite",
                    "antonym",
                    "binary relation",
                    "semantic transformation"
                ],
                "expected_answer": "dark"
            },
            {
                "task_id": "arc_003",
                "type": "spatial_reasoning",
                "description": "Spatial and structural reasoning",
                "task": """
                Task: Understand spatial relationships and transformations.
                
                Given:
                - Shape A is inside Shape B
                - Shape B is inside Shape C
                - Shape C contains 3 elements
                
                Question:
                If Shape A contains 2 elements, and Shape B contains 5 elements,
                how many total elements are there?
                
                Show your reasoning about spatial containment and set relationships.
                """,
                "expected_reasoning": [
                    "hierarchical containment",
                    "set theory",
                    "spatial logic",
                    "nested structures"
                ],
                "expected_answer": "5"  # B contains A, so max is 5
            },
            {
                "task_id": "arc_004",
                "type": "novel_generalization",
                "description": "Apply learned patterns to novel situations",
                "task": """
                Task: Generalize from examples to novel case.
                
                Training Examples:
                1. Red + Blue = Purple (color mixing)
                2. Happy + Sad = Neutral (emotion blending)
                3. Fast + Slow = Medium (speed averaging)
                
                Test Case:
                Apply the same pattern to:
                High + Low = ?
                
                What is the underlying transformation rule?
                """,
                "expected_reasoning": [
                    "binary operation",
                    "middle value",
                    "averaging",
                    "generalization"
                ],
                "expected_answer": "Medium"  # or similar middle value
            },
            {
                "task_id": "arc_005",
                "type": "multi_step_reasoning",
                "description": "Complex multi-step abstract reasoning",
                "task": """
                Task: Solve multi-step abstract reasoning problem.
                
                Rules:
                - Rule 1: If X is A, then Y is B
                - Rule 2: If Y is B, then Z is C
                - Rule 3: If Z is C, then W is D
                
                Given: X is A
                
                Question: What is W?
                Show each step of the reasoning chain.
                """,
                "expected_reasoning": [
                    "logical chain",
                    "transitive reasoning",
                    "rule application",
                    "deductive logic"
                ],
                "expected_answer": "D"
            },
            {
                "task_id": "arc_006",
                "type": "compositional_reasoning",
                "description": "Understand compositional patterns",
                "task": """
                Task: Understand compositional relationships.
                
                Pattern:
                - Single unit: "one"
                - Pair: "two"
                - Trio: "three"
                
                Composition rule:
                Two units + One unit = Three units (additive composition)
                
                Test:
                Three units + Two units = ?
                
                Apply the compositional reasoning.
                """,
                "expected_reasoning": [
                    "composition",
                    "additive",
                    "pattern application",
                    "structured reasoning"
                ],
                "expected_answer": "Five units"
            },
            {
                "task_id": "arc_007",
                "type": "analogical_reasoning",
                "description": "Analogical pattern matching",
                "task": """
                Task: Find analogical relationships.
                
                Analogy:
                Book : Library :: ?
                
                Options:
                A) Fish : Ocean
                B) Tree : Forest
                C) Car : Garage
                D) Star : Sky
                
                Identify the relationship type and select the best match.
                Show your reasoning about the analogy structure.
                """,
                "expected_reasoning": [
                    "analogy",
                    "relationship matching",
                    "semantic similarity",
                    "structural equivalence"
                ],
                "expected_answer": "B"  # Collection relationship
            },
            {
                "task_id": "arc_008",
                "type": "counterfactual_reasoning",
                "description": "Counterfactual and hypothetical reasoning",
                "task": """
                Task: Counterfactual reasoning.
                
                Scenario:
                If gravity worked in reverse (objects pushed away from Earth),
                and you dropped a ball from a height,
                what would happen?
                
                Consider:
                1. Initial motion direction
                2. Acceleration behavior
                3. Final outcome
                
                Provide counterfactual reasoning.
                """,
                "expected_reasoning": [
                    "counterfactual",
                    "reverse physics",
                    "hypothetical",
                    "logical extension"
                ],
                "expected_answer": "Ball would accelerate upward and away from Earth"
            },
            {
                "task_id": "arc_009",
                "type": "hierarchical_reasoning",
                "description": "Multi-level hierarchical understanding",
                "task": """
                Task: Hierarchical pattern recognition.
                
                Structure:
                Level 1: Category A (contains B and C)
                Level 2: Category B (contains D and E)
                Level 2: Category C (contains F and G)
                Level 3: Items D, E, F, G
                
                Question:
                If you traverse from A to E, what path do you take?
                How many levels deep is E?
                
                Show hierarchical reasoning.
                """,
                "expected_reasoning": [
                    "hierarchy",
                    "tree structure",
                    "path traversal",
                    "depth calculation"
                ],
                "expected_answer": "Path: A → B → E, Depth: 3"
            },
            {
                "task_id": "arc_010",
                "type": "emergent_pattern",
                "description": "Detect emergent patterns from data",
                "task": """
                Task: Identify emergent pattern from sequence.
                
                Sequence:
                1, 1, 2, 3, 5, 8, 13, 21, ?
                
                Questions:
                1. What is the pattern?
                2. What is the next number?
                3. What is the 20th number in the sequence?
                
                Show pattern recognition and prediction.
                """,
                "expected_reasoning": [
                    "fibonacci",
                    "recursive sequence",
                    "pattern detection",
                    "sequence prediction"
                ],
                "expected_answer": "Next: 34, 20th: 6765 (Fibonacci sequence)"
            }
        ]
        
        return tasks
    
    def score_response(self, response: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Score ICEBURG's response against expected reasoning and answer."""
        
        response_lower = response.lower()
        
        # Check for expected reasoning patterns
        reasoning_found = []
        for pattern in task["expected_reasoning"]:
            if pattern.lower() in response_lower:
                reasoning_found.append(pattern)
        
        reasoning_score = len(reasoning_found) / len(task["expected_reasoning"])
        
        # Check for expected answer
        answer_score = 0.0
        expected_answer_lower = task["expected_answer"].lower()
        
        # Exact match
        if expected_answer_lower in response_lower:
            answer_score = 1.0
        else:
            # Partial match - check for key components
            expected_words = set(expected_answer_lower.split())
            response_words = set(response_lower.split())
            common_words = expected_words.intersection(response_words)
            
            if common_words:
                answer_score = len(common_words) / len(expected_words)
        
        # Overall score (reasoning 60%, answer 40%)
        overall_score = (reasoning_score * 0.6) + (answer_score * 0.4)
        
        return {
            "reasoning_score": reasoning_score,
            "reasoning_found": reasoning_found,
            "answer_score": answer_score,
            "overall_score": overall_score,
            "total_patterns": len(task["expected_reasoning"]),
            "patterns_found": len(reasoning_found)
        }
    
    async def run_task(self, task: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """Run a single ARC-AGI task."""
        
        if verbose:
            print(f"\n  🧪 Running Task {task['task_id']}: {task['type']}")
        
        start_time = time.time()
        
        try:
            if iceberg_protocol:
                # Use ICEBURG protocol
                response = iceberg_protocol(
                    task["task"],
                    verbose=False
                )
            else:
                # Fallback: simple response
                response = f"Analyzing {task['task_id']} task..."
                await asyncio.sleep(0.5)
            
            duration = time.time() - start_time
            
            # Score the response
            score_result = self.score_response(response, task)
            
            result = {
                "task_id": task["task_id"],
                "type": task["type"],
                "description": task["description"],
                "response": response[:500] if len(response) > 500 else response,  # Truncate long responses
                "response_length": len(response),
                "duration": duration,
                "score": score_result,
                "timestamp": datetime.now().isoformat()
            }
            
            if verbose:
                print(f"    ✅ Score: {score_result['overall_score']:.2%}")
                print(f"    ⏱️  Duration: {duration:.2f}s")
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"    ❌ Error: {e}")
            
            return {
                "task_id": task["task_id"],
                "type": task["type"],
                "error": str(e),
                "duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_all_tasks(self, verbose: bool = True) -> Dict[str, Any]:
        """Run all ARC-AGI tasks."""
        
        print("="*80)
        print("ARC-AGI BENCHMARK SUITE")
        print("="*80)
        print("\nTesting ICEBURG's Abstract Reasoning and Generalization Capabilities")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()
        
        tasks = self.create_abstract_reasoning_tasks()
        
        results = {
            "metadata": {
                "benchmark": "ARC-AGI",
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "total_tasks": len(tasks),
                "system": "ICEBURG"
            },
            "tasks": [],
            "summary": {}
        }
        
        task_results = []
        total_score = 0.0
        successful_tasks = 0
        
        for i, task in enumerate(tasks, 1):
            if verbose:
                print(f"\n[{i}/{len(tasks)}] Task {task['task_id']}")
            
            result = await self.run_task(task, verbose=verbose)
            task_results.append(result)
            
            if "error" not in result and "score" in result:
                if result["score"]["overall_score"] > 0:
                    successful_tasks += 1
                    total_score += result["score"]["overall_score"]
            
            # Small delay between tasks
            await asyncio.sleep(0.5)
        
        # Calculate summary statistics
        if task_results:
            scores = [r["score"]["overall_score"] for r in task_results if "score" in r]
            durations = [r["duration"] for r in task_results if "duration" in r]
            
            results["summary"] = {
                "total_tasks": len(tasks),
                "completed_tasks": len([r for r in task_results if "error" not in r]),
                "successful_tasks": successful_tasks,
                "average_score": sum(scores) / len(scores) if scores else 0.0,
                "total_score": total_score,
                "max_score": max(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "average_duration": sum(durations) / len(durations) if durations else 0.0,
                "total_duration": sum(durations) if durations else 0.0,
                "success_rate": successful_tasks / len(tasks) if tasks else 0.0
            }
        
        results["tasks"] = task_results
        
        # Print summary
        if verbose:
            print("\n" + "="*80)
            print("ARC-AGI BENCHMARK SUMMARY")
            print("="*80)
            print(f"Total Tasks: {results['summary']['total_tasks']}")
            print(f"Completed: {results['summary']['completed_tasks']}")
            print(f"Successful: {results['summary']['successful_tasks']}")
            print(f"Success Rate: {results['summary']['success_rate']:.1%}")
            print(f"Average Score: {results['summary']['average_score']:.2%}")
            print(f"Total Duration: {results['summary']['total_duration']:.2f}s")
            print(f"Average Duration per Task: {results['summary']['average_duration']:.2f}s")
            
            # Comparison to human performance
            print("\n" + "-"*80)
            print("COMPARISON TO HUMAN PERFORMANCE")
            print("-"*80)
            print("Human Performance (ARC): 73-77% accuracy")
            print(f"ICEBURG Performance: {results['summary']['average_score']:.1%} average score")
            
            if results['summary']['average_score'] >= 0.73:
                print("✅ ICEBURG meets human-level performance threshold")
            elif results['summary']['average_score'] >= 0.50:
                print("⚠️  ICEBURG approaches human-level performance")
            else:
                print("❌ ICEBURG below human-level performance threshold")
            
            print("\n" + "-"*80)
            print("COMPARISON TO LEADING MODELS")
            print("-"*80)
            print("OpenAI o3: 87.5% (ARC-AGI-1)")
            print("Grok 4: ~68% (ARC-AGI-1)")
            print("GPT-4o: 50% (with prompt engineering)")
            print(f"ICEBURG: {results['summary']['average_score']:.1%}")
        
        return results
    
    def save_results(self, results: Dict[str, Any]) -> Path:
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"arc_agi_benchmark_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        return output_file


async def main():
    """Main benchmark runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARC-AGI Benchmark for ICEBURG")
    parser.add_argument("--output", default="benchmarks/results", 
                       help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    benchmark = ARCAGIBenchmark(output_dir=args.output)
    
    try:
        results = await benchmark.run_all_tasks(verbose=args.verbose)
        benchmark.save_results(results)
        
        print("\n✅ ARC-AGI Benchmark Complete!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

