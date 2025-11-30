"""
ICEBURG Departments Module

Powerful department system for distributed intelligence and collaborative brainstorming.
"""

from .think_tank import (
    ThinkTankDepartment,
    ThinkTankCoordinator,
    DepartmentType,
    TaskPriority,
    ThinkTankTask,
    DepartmentAgent
)

__all__ = [
    'ThinkTankDepartment',
    'ThinkTankCoordinator', 
    'DepartmentType',
    'TaskPriority',
    'ThinkTankTask',
    'DepartmentAgent'
]
