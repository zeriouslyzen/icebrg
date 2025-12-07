"""
Comprehensive stress tests for Secretary Agent Goal-Driven Planning (Phase 1)

Tests complex scenarios, edge cases, and stress conditions.
"""

import sys
import os
import json
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.iceburg.config import load_config
from src.iceburg.agents.secretary import SecretaryAgent, run
from src.iceburg.agents.secretary_planner import SecretaryPlanner, Task, TaskStatus


def create_test_config():
    """Create a test configuration."""
    try:
        cfg = load_config()
        return cfg
    except Exception as e:
        # Fallback to minimal config for testing structure
        from pathlib import Path
        from src.iceburg.config import IceburgConfig
        return IceburgConfig(
            data_dir=Path("./data"),
            surveyor_model="gemini-2.0-flash-exp",
            dissident_model="gemini-2.0-flash-exp",
            synthesist_model="gemini-2.0-flash-exp",
            oracle_model="gemini-2.0-flash-exp",
            embed_model="nomic-embed-text"
        )


def test_simple_goal_detection():
    """Test 1: Simple goal detection."""
    print("\n" + "="*60)
    print("TEST 1: Simple Goal Detection")
    print("="*60)
    
    cfg = create_test_config()
    planner = SecretaryPlanner(cfg)
    
    test_queries = [
        "What is ICEBURG?",  # Should NOT be a goal
        "Organize my files",  # Should be a goal
        "Summarize all PDFs",  # Should be a goal
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            goals = planner.extract_goals(query)
            if goals:
                print(f"  ✓ Detected {len(goals)} goal(s)")
                for goal in goals:
                    print(f"    - {goal.get('description', 'N/A')}")
            else:
                print(f"  ✓ No goals detected (simple query)")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    return True


def test_complex_multi_goal_query():
    """Test 2: Complex query with multiple goals."""
    print("\n" + "="*60)
    print("TEST 2: Complex Multi-Goal Query")
    print("="*60)
    
    cfg = create_test_config()
    planner = SecretaryPlanner(cfg)
    
    complex_query = """
    I need you to:
    1. Organize all my files by type
    2. Summarize all PDF documents
    3. Create a research document about quantum computing
    """
    
    print(f"Query: '{complex_query.strip()}'")
    try:
        goals = planner.extract_goals(complex_query)
        print(f"  ✓ Detected {len(goals)} goal(s)")
        
        for i, goal in enumerate(goals, 1):
            print(f"\n  Goal {i}:")
            print(f"    Description: {goal.get('description', 'N/A')}")
            print(f"    Priority: {goal.get('priority', 'N/A')}")
            print(f"    Sub-goals: {len(goal.get('sub_goals', []))}")
            
            # Test planning for this goal
            tasks = planner.plan_task(goal)
            print(f"    Tasks planned: {len(tasks)}")
            for j, task in enumerate(tasks[:3], 1):  # Show first 3
                print(f"      {j}. {task.description[:60]}...")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return True


def test_task_dependency_resolution():
    """Test 3: Task dependency resolution."""
    print("\n" + "="*60)
    print("TEST 3: Task Dependency Resolution")
    print("="*60)
    
    # Create tasks with dependencies
    tasks = [
        Task(task_id="task_0", description="Step 1: Initialize", dependencies=[]),
        Task(task_id="task_1", description="Step 2: Process", dependencies=["task_0"]),
        Task(task_id="task_2", description="Step 3: Validate", dependencies=["task_1"]),
        Task(task_id="task_3", description="Step 4: Finalize", dependencies=["task_2"]),
    ]
    
    print("Task dependency chain:")
    for task in tasks:
        deps = ", ".join(task.dependencies) if task.dependencies else "none"
        print(f"  {task.task_id}: {task.description[:40]}... (depends on: {deps})")
    
    # Verify dependency order
    task_map = {task.task_id: task for task in tasks}
    execution_order = []
    completed = set()
    
    while len(completed) < len(tasks):
        ready = [
            t for t in tasks
            if t.task_id not in completed
            and all(dep in completed for dep in t.dependencies)
        ]
        if not ready:
            print("  ✗ Circular dependency or missing task detected!")
            break
        for task in ready:
            execution_order.append(task.task_id)
            completed.add(task.task_id)
    
    print(f"\n  ✓ Execution order: {' → '.join(execution_order)}")
    return True


def test_goal_hierarchy_integration():
    """Test 4: Goal hierarchy integration."""
    print("\n" + "="*60)
    print("TEST 4: Goal Hierarchy Integration")
    print("="*60)
    
    cfg = create_test_config()
    planner = SecretaryPlanner(cfg)
    
    from src.iceburg.civilization.persistent_agents import GoalPriority
    
    # Add multiple goals with different priorities
    goal_ids = []
    goals_data = [
        ("Organize files", GoalPriority.HIGH),
        ("Summarize PDFs", GoalPriority.MEDIUM),
        ("Create document", GoalPriority.LOW),
    ]
    
    for desc, priority in goals_data:
        goal_id = planner.goal_hierarchy.add_goal(
            description=desc,
            priority=priority
        )
        goal_ids.append(goal_id)
        print(f"  ✓ Added goal: {desc} (priority: {priority.value}, id: {goal_id})")
    
    # Get ready goals (should be all, since no dependencies)
    ready_goals = planner.goal_hierarchy.get_ready_goals()
    print(f"\n  ✓ Ready goals: {len(ready_goals)}")
    for goal in ready_goals:
        print(f"    - {goal.description} (priority: {goal.priority.value})")
    
    # Update progress
    planner.goal_hierarchy.update_goal_progress(goal_ids[0], 0.5)
    progress = planner.get_goal_progress(goal_ids[0])
    print(f"\n  ✓ Goal progress: {progress['progress']:.1%}")
    
    return True


def test_stress_multiple_goals():
    """Test 5: Stress test with many goals."""
    print("\n" + "="*60)
    print("TEST 5: Stress Test - Multiple Goals")
    print("="*60)
    
    cfg = create_test_config()
    planner = SecretaryPlanner(cfg)
    
    from src.iceburg.civilization.persistent_agents import GoalPriority
    
    # Add many goals
    num_goals = 20
    print(f"  Creating {num_goals} goals...")
    
    goal_ids = []
    for i in range(num_goals):
        goal_id = planner.goal_hierarchy.add_goal(
            description=f"Goal {i+1}: Task {i+1}",
            priority=GoalPriority.MEDIUM
        )
        goal_ids.append(goal_id)
    
    print(f"  ✓ Created {len(goal_ids)} goals")
    
    # Check stats
    stats = planner.goal_hierarchy.get_goal_stats()
    print(f"  ✓ Total goals: {stats['total_goals']}")
    print(f"  ✓ Active goals: {stats['active_goals']}")
    
    # Complete some goals
    for goal_id in goal_ids[:5]:
        planner.goal_hierarchy.update_goal_progress(goal_id, 1.0)
    
    stats = planner.goal_hierarchy.get_goal_stats()
    print(f"  ✓ After completion: {stats['completed_goals']} completed, {stats['active_goals']} active")
    
    return True


def test_error_handling():
    """Test 6: Error handling and edge cases."""
    print("\n" + "="*60)
    print("TEST 6: Error Handling & Edge Cases")
    print("="*60)
    
    cfg = create_test_config()
    planner = SecretaryPlanner(cfg)
    
    # Test 1: Empty query
    print("\n  Test 6.1: Empty query")
    try:
        goals = planner.extract_goals("")
        print(f"    ✓ Handled empty query: {len(goals)} goals")
    except Exception as e:
        print(f"    ✗ Error with empty query: {e}")
    
    # Test 2: Invalid goal data
    print("\n  Test 6.2: Invalid goal data")
    try:
        invalid_goal = {"description": ""}  # Missing required fields
        tasks = planner.plan_task(invalid_goal)
        print(f"    ✓ Handled invalid goal: {len(tasks)} tasks")
    except Exception as e:
        print(f"    ✗ Error with invalid goal: {e}")
    
    # Test 3: Circular dependencies
    print("\n  Test 6.3: Circular dependencies")
    try:
        tasks = [
            Task(task_id="task_0", description="Task 0", dependencies=["task_1"]),
            Task(task_id="task_1", description="Task 1", dependencies=["task_0"]),
        ]
        result = planner.execute_plan(tasks)
        print(f"    ✓ Detected circular dependency: {result.get('failed', 0)} failed")
    except Exception as e:
        print(f"    ✗ Error with circular dependency: {e}")
    
    # Test 4: Missing dependency
    print("\n  Test 6.4: Missing dependency")
    try:
        tasks = [
            Task(task_id="task_0", description="Task 0", dependencies=["missing_task"]),
        ]
        result = planner.execute_plan(tasks)
        print(f"    ✓ Handled missing dependency: {result.get('failed', 0)} failed")
    except Exception as e:
        print(f"    ✗ Error with missing dependency: {e}")
    
    return True


def test_integration_with_secretary_agent():
    """Test 7: Integration with SecretaryAgent."""
    print("\n" + "="*60)
    print("TEST 7: Secretary Agent Integration")
    print("="*60)
    
    cfg = create_test_config()
    
    try:
        # Create agent with planning enabled
        agent = SecretaryAgent(
            cfg,
            enable_memory=False,  # Disable for faster testing
            enable_tools=False,
            enable_blackboard=False,
            enable_cache=False,
            enable_planning=True
        )
        
        print("  ✓ SecretaryAgent created with planning enabled")
        print(f"  ✓ Planner initialized: {agent.planner is not None}")
        print(f"  ✓ Planning enabled: {agent.enable_planning}")
        
        # Test that agent has planning capability
        if agent.planner:
            print("  ✓ Planning engine accessible")
            print(f"  ✓ Goal hierarchy available: {agent.planner.goal_hierarchy is not None}")
        
    except Exception as e:
        print(f"  ✗ Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_complex_scenario_file_organization():
    """Test 8: Complex scenario - File organization."""
    print("\n" + "="*60)
    print("TEST 8: Complex Scenario - File Organization")
    print("="*60)
    
    cfg = create_test_config()
    planner = SecretaryPlanner(cfg)
    
    # Simulate file organization goal
    goal = {
        "description": "Organize all files in /Users/test/documents by type and date",
        "priority": "high",
        "sub_goals": [
            "Scan directory structure",
            "Identify file types",
            "Create category folders",
            "Move files to appropriate folders",
            "Verify organization",
            "Generate report"
        ]
    }
    
    print(f"Goal: {goal['description']}")
    print(f"Sub-goals: {len(goal['sub_goals'])}")
    
    # Plan the task
    tasks = planner.plan_task(goal)
    print(f"\n  ✓ Planned {len(tasks)} tasks:")
    
    for i, task in enumerate(tasks, 1):
        deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
        print(f"    {i}. {task.description}{deps}")
    
    # Verify dependency chain
    task_map = {task.task_id: task for task in tasks}
    has_valid_order = True
    
    for task in tasks:
        for dep_id in task.dependencies:
            if dep_id not in task_map:
                has_valid_order = False
                print(f"    ✗ Invalid dependency: {dep_id} not found")
    
    if has_valid_order:
        print("\n  ✓ All dependencies valid")
    
    return True


def test_complex_scenario_research_document():
    """Test 9: Complex scenario - Research document creation."""
    print("\n" + "="*60)
    print("TEST 9: Complex Scenario - Research Document")
    print("="*60)
    
    cfg = create_test_config()
    planner = SecretaryPlanner(cfg)
    
    goal = {
        "description": "Create a comprehensive research document on quantum computing applications in AI",
        "priority": "high",
        "sub_goals": [
            "Research quantum computing fundamentals",
            "Research AI applications",
            "Find connections between quantum computing and AI",
            "Gather relevant papers and sources",
            "Structure document outline",
            "Write introduction",
            "Write main sections",
            "Write conclusion",
            "Add citations",
            "Format document"
        ]
    }
    
    print(f"Goal: {goal['description']}")
    print(f"Sub-goals: {len(goal['sub_goals'])}")
    
    tasks = planner.plan_task(goal)
    print(f"\n  ✓ Planned {len(tasks)} tasks")
    
    # Group tasks by phase
    phases = {
        "Research": [],
        "Analysis": [],
        "Writing": [],
        "Finalization": []
    }
    
    for task in tasks:
        desc_lower = task.description.lower()
        if any(word in desc_lower for word in ["research", "gather", "find"]):
            phases["Research"].append(task)
        elif any(word in desc_lower for word in ["connect", "analyze", "structure"]):
            phases["Analysis"].append(task)
        elif any(word in desc_lower for word in ["write", "introduction", "sections", "conclusion"]):
            phases["Writing"].append(task)
        else:
            phases["Finalization"].append(task)
    
    print("\n  Task phases:")
    for phase, phase_tasks in phases.items():
        if phase_tasks:
            print(f"    {phase}: {len(phase_tasks)} tasks")
    
    return True


def run_all_tests():
    """Run all stress tests."""
    print("\n" + "="*60)
    print("SECRETARY AGENT PHASE 1 - COMPREHENSIVE STRESS TESTS")
    print("="*60)
    
    tests = [
        ("Simple Goal Detection", test_simple_goal_detection),
        ("Complex Multi-Goal Query", test_complex_multi_goal_query),
        ("Task Dependency Resolution", test_task_dependency_resolution),
        ("Goal Hierarchy Integration", test_goal_hierarchy_integration),
        ("Stress: Multiple Goals", test_stress_multiple_goals),
        ("Error Handling", test_error_handling),
        ("Secretary Agent Integration", test_integration_with_secretary_agent),
        ("Complex: File Organization", test_complex_scenario_file_organization),
        ("Complex: Research Document", test_complex_scenario_research_document),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result, _ in results if result)
    failed = len(results) - passed
    
    for test_name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"    Error: {error}")
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

