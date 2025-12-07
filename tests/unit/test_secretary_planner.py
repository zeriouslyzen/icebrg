"""
Unit tests for Secretary Planner (Phase 1: Goal-Driven Autonomy)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.iceburg.agents.secretary_planner import SecretaryPlanner, Task, TaskStatus
from src.iceburg.config import IceburgConfig


@pytest.fixture
def mock_config():
    """Create a mock config."""
    cfg = Mock(spec=IceburgConfig)
    cfg.surveyor_model = "gemini-2.0-flash-exp"
    cfg.primary_model = "gemini-2.0-flash-exp"
    return cfg


@pytest.fixture
def planner(mock_config):
    """Create a planner instance."""
    return SecretaryPlanner(mock_config)


def test_planner_initialization(planner):
    """Test planner initializes correctly."""
    assert planner is not None
    assert planner.goal_hierarchy is not None
    assert planner.task_counter == 0


def test_task_creation():
    """Test Task dataclass creation."""
    task = Task(
        task_id="task_0",
        description="Test task",
        status=TaskStatus.PENDING
    )
    assert task.task_id == "task_0"
    assert task.description == "Test task"
    assert task.status == TaskStatus.PENDING
    assert task.dependencies == []


@patch('src.iceburg.agents.secretary_planner.provider_factory')
def test_extract_goals_simple_query(mock_factory, planner):
    """Test goal extraction for simple query (no goals)."""
    mock_provider = Mock()
    mock_provider.chat_complete.return_value = '{"is_goal": false, "goals": []}'
    mock_factory.return_value = mock_provider
    
    goals = planner.extract_goals("What is ICEBURG?")
    assert goals == []


@patch('src.iceburg.agents.secretary_planner.provider_factory')
def test_extract_goals_with_goal(mock_factory, planner):
    """Test goal extraction for goal-driven query."""
    mock_provider = Mock()
    mock_provider.chat_complete.return_value = '''{
        "is_goal": true,
        "goals": [{
            "description": "Organize files in directory",
            "priority": "medium",
            "sub_goals": ["Scan directory", "Categorize files", "Move files"]
        }]
    }'''
    mock_factory.return_value = mock_provider
    
    goals = planner.extract_goals("Organize my files")
    assert len(goals) == 1
    assert goals[0]["description"] == "Organize files in directory"
    assert len(goals[0]["sub_goals"]) == 3


@patch('src.iceburg.agents.secretary_planner.provider_factory')
def test_plan_task_with_subgoals(mock_factory, planner):
    """Test task planning with provided sub-goals."""
    goal = {
        "description": "Organize files",
        "sub_goals": ["Scan", "Categorize", "Move"]
    }
    
    tasks = planner.plan_task(goal)
    assert len(tasks) == 3
    assert tasks[0].description == "Scan"
    assert tasks[1].description == "Categorize"
    assert tasks[2].description == "Move"
    # Check dependencies
    assert tasks[0].dependencies == []
    assert tasks[1].dependencies == [tasks[0].task_id]
    assert tasks[2].dependencies == [tasks[0].task_id, tasks[1].task_id]


@patch('src.iceburg.agents.secretary_planner.provider_factory')
def test_plan_task_without_subgoals(mock_factory, planner):
    """Test task planning without sub-goals (LLM decomposition)."""
    mock_provider = Mock()
    mock_provider.chat_complete.return_value = '["Step 1: Analyze", "Step 2: Execute", "Step 3: Verify"]'
    mock_factory.return_value = mock_provider
    
    goal = {"description": "Complete task"}
    tasks = planner.plan_task(goal)
    assert len(tasks) >= 1


@patch('src.iceburg.agents.secretary_planner.provider_factory')
def test_execute_task_success(mock_factory, planner):
    """Test successful task execution."""
    mock_provider = Mock()
    mock_provider.chat_complete.return_value = "Task completed successfully"
    mock_factory.return_value = mock_provider
    
    task = Task(task_id="task_0", description="Test task")
    result = planner.execute_task(task)
    
    assert result["success"] is True
    assert task.status == TaskStatus.COMPLETED
    assert task.result is not None


@patch('src.iceburg.agents.secretary_planner.provider_factory')
def test_execute_plan_sequential(mock_factory, planner):
    """Test executing a plan with multiple tasks."""
    mock_provider = Mock()
    mock_provider.chat_complete.return_value = "Step completed"
    mock_factory.return_value = mock_provider
    
    tasks = [
        Task(task_id="task_0", description="Step 1", dependencies=[]),
        Task(task_id="task_1", description="Step 2", dependencies=["task_0"]),
    ]
    
    result = planner.execute_plan(tasks)
    
    assert result["total_tasks"] == 2
    assert result["completed"] == 2
    assert result["failed"] == 0
    assert len(result["results"]) == 2


def test_get_goal_progress(planner):
    """Test getting goal progress."""
    from src.iceburg.civilization.persistent_agents import GoalPriority
    
    goal_id = planner.goal_hierarchy.add_goal(
        description="Test goal",
        priority=GoalPriority.MEDIUM
    )
    
    planner.goal_hierarchy.update_goal_progress(goal_id, 0.5)
    
    progress = planner.get_goal_progress(goal_id)
    assert progress["goal_id"] == goal_id
    assert progress["progress"] == 0.5
    assert progress["status"] == "in_progress"

