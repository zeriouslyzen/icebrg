"""
Unity ML-Agents Template for ICEBURG
Generates Unity projects with ML-Agents integration for AI training and simulation.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MLAgentType(Enum):
    """ML Agent types."""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    IMITATION_LEARNING = "imitation_learning"
    CURRICULUM_LEARNING = "curriculum_learning"
    MULTI_AGENT = "multi_agent"
    HIERARCHICAL = "hierarchical"


class EnvironmentType(Enum):
    """Environment types."""
    GRID_WORLD = "grid_world"
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    MIXED = "mixed"


@dataclass
class MLAgentConfig:
    """ML Agent configuration."""
    agent_type: MLAgentType
    behavior_name: str
    observation_space: int
    action_space: int
    max_steps: int = 1000
    reward_signal: str = "extrinsic"
    curriculum: bool = False
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    network_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    env_type: EnvironmentType
    size: Tuple[int, int] = (10, 10)
    obstacles: List[Tuple[int, int]] = field(default_factory=list)
    goals: List[Tuple[int, int]] = field(default_factory=list)
    spawn_points: List[Tuple[int, int]] = field(default_factory=list)
    physics_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnityProject:
    """Unity project structure."""
    project_name: str
    project_path: str
    agents: List[MLAgentConfig] = field(default_factory=list)
    environments: List[EnvironmentConfig] = field(default_factory=list)
    training_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnityMLTemplate:
    """
    Unity ML-Agents template generator.
    
    Features:
    - ML-Agents integration
    - Procedural content generation
    - Training configuration
    - Multi-agent environments
    - Curriculum learning
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Unity ML template.
        
        Args:
            config: Template configuration
        """
        self.config = config or {}
        self.template_dir = self.config.get("template_dir", "templates/unity")
        self.unity_version = self.config.get("unity_version", "2022.3.0f1")
        self.ml_agents_version = self.config.get("ml_agents_version", "2.0.0")
        
        # Ensure template directory exists
        os.makedirs(self.template_dir, exist_ok=True)
    
    async def generate_project(self, 
                             project_name: str,
                             description: str,
                             agent_specs: List[Dict[str, Any]] = None,
                             environment_specs: List[Dict[str, Any]] = None,
                             training_specs: Dict[str, Any] = None) -> UnityProject:
        """
        Generate Unity ML-Agents project.
        
        Args:
            project_name: Project name
            description: Project description
            agent_specs: Agent specifications
            environment_specs: Environment specifications
            training_specs: Training specifications
            
        Returns:
            Generated Unity project
        """
        project_path = os.path.join(self.template_dir, project_name)
        os.makedirs(project_path, exist_ok=True)
        
        # Create project structure
        await self._create_project_structure(project_path)
        
        # Generate agents
        agents = []
        if agent_specs:
            for spec in agent_specs:
                agent = await self._create_ml_agent(spec)
                agents.append(agent)
                await self._generate_agent_script(project_path, agent)
        
        # Generate environments
        environments = []
        if environment_specs:
            for spec in environment_specs:
                env = await self._create_environment(spec)
                environments.append(env)
                await self._generate_environment_script(project_path, env)
        
        # Generate training configuration
        training_config = await self._generate_training_config(training_specs or {})
        await self._write_training_config(project_path, training_config)
        
        # Generate project files
        await self._generate_project_files(project_path, project_name, description)
        
        # Generate ML-Agents configuration
        await self._generate_ml_agents_config(project_path, agents, environments)
        
        project = UnityProject(
            project_name=project_name,
            project_path=project_path,
            agents=agents,
            environments=environments,
            training_config=training_config,
            metadata={
                "description": description,
                "unity_version": self.unity_version,
                "ml_agents_version": self.ml_agents_version,
                "created_time": time.time()
            }
        )
        
        logger.info(f"Generated Unity ML project: {project_name}")
        return project
    
    async def _create_project_structure(self, project_path: str):
        """Create Unity project directory structure."""
        directories = [
            "Assets/Scripts/Agents",
            "Assets/Scripts/Environments",
            "Assets/Scripts/Utils",
            "Assets/Prefabs",
            "Assets/Materials",
            "Assets/Scenes",
            "Assets/ML-Agents",
            "Assets/ML-Agents/Config",
            "Assets/ML-Agents/Models",
            "Assets/ML-Agents/Results",
            "ProjectSettings"
        ]
        
        for directory in directories:
            os.makedirs(os.path.join(project_path, directory), exist_ok=True)
    
    async def _create_ml_agent(self, spec: Dict[str, Any]) -> MLAgentConfig:
        """Create ML agent from specification."""
        agent_type = MLAgentType(spec.get("type", "reinforcement_learning"))
        behavior_name = spec.get("behavior_name", "DefaultBehavior")
        observation_space = spec.get("observation_space", 8)
        action_space = spec.get("action_space", 4)
        max_steps = spec.get("max_steps", 1000)
        reward_signal = spec.get("reward_signal", "extrinsic")
        curriculum = spec.get("curriculum", False)
        
        hyperparameters = spec.get("hyperparameters", {
            "learning_rate": 3e-4,
            "batch_size": 64,
            "buffer_size": 12000,
            "learning_starts": 1000,
            "train_frequency": 4,
            "target_network_frequency": 1000,
            "tau": 1.0,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995
        })
        
        network_settings = spec.get("network_settings", {
            "hidden_units": 128,
            "num_layers": 2,
            "vis_encode_type": "simple",
            "normalize": True
        })
        
        return MLAgentConfig(
            agent_type=agent_type,
            behavior_name=behavior_name,
            observation_space=observation_space,
            action_space=action_space,
            max_steps=max_steps,
            reward_signal=reward_signal,
            curriculum=curriculum,
            hyperparameters=hyperparameters,
            network_settings=network_settings
        )
    
    async def _create_environment(self, spec: Dict[str, Any]) -> EnvironmentConfig:
        """Create environment from specification."""
        env_type = EnvironmentType(spec.get("type", "grid_world"))
        size = tuple(spec.get("size", [10, 10]))
        obstacles = [tuple(obs) for obs in spec.get("obstacles", [])]
        goals = [tuple(goal) for goal in spec.get("goals", [])]
        spawn_points = [tuple(spawn) for spawn in spec.get("spawn_points", [])]
        
        physics_settings = spec.get("physics_settings", {
            "gravity": -9.81,
            "time_scale": 1.0,
            "fixed_timestep": 0.02
        })
        
        return EnvironmentConfig(
            env_type=env_type,
            size=size,
            obstacles=obstacles,
            goals=goals,
            spawn_points=spawn_points,
            physics_settings=physics_settings
        )
    
    async def _generate_agent_script(self, project_path: str, agent: MLAgentConfig):
        """Generate Unity C# script for ML agent."""
        script_content = f"""
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class {agent.behavior_name}Agent : Agent
{{
    [Header("Agent Settings")]
    public float moveSpeed = 5f;
    public float rotationSpeed = 100f;
    
    [Header("Environment")]
    public Transform target;
    public Transform[] obstacles;
    
    private Rigidbody rb;
    private Vector3 startPosition;
    private Quaternion startRotation;
    
    public override void Initialize()
    {{
        rb = GetComponent<Rigidbody>();
        startPosition = transform.position;
        startRotation = transform.rotation;
    }}
    
    public override void OnEpisodeBegin()
    {{
        // Reset agent position and rotation
        transform.position = startPosition;
        transform.rotation = startRotation;
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        
        // Randomize target position
        if (target != null)
        {{
            target.position = GetRandomPosition();
        }}
    }}
    
    public override void CollectObservations(VectorSensor sensor)
    {{
        // Agent position
        sensor.AddObservation(transform.position.x);
        sensor.AddObservation(transform.position.z);
        
        // Agent rotation
        sensor.AddObservation(transform.rotation.y);
        
        // Target position (relative to agent)
        if (target != null)
        {{
            Vector3 targetDirection = (target.position - transform.position).normalized;
            sensor.AddObservation(targetDirection.x);
            sensor.AddObservation(targetDirection.z);
        }}
        else
        {{
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }}
        
        // Distance to target
        if (target != null)
        {{
            float distance = Vector3.Distance(transform.position, target.position);
            sensor.AddObservation(distance);
        }}
        else
        {{
            sensor.AddObservation(0f);
        }}
        
        // Agent velocity
        sensor.AddObservation(rb.velocity.x);
        sensor.AddObservation(rb.velocity.z);
    }}
    
    public override void OnActionReceived(ActionBuffers actions)
    {{
        // Get actions
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        float rotate = actions.ContinuousActions[2];
        
        // Apply movement
        Vector3 movement = new Vector3(moveX, 0, moveZ) * moveSpeed * Time.deltaTime;
        rb.MovePosition(transform.position + movement);
        
        // Apply rotation
        float rotation = rotate * rotationSpeed * Time.deltaTime;
        transform.Rotate(0, rotation, 0);
        
        // Calculate reward
        CalculateReward();
    }}
    
    private void CalculateReward()
    {{
        if (target == null) return;
        
        float distance = Vector3.Distance(transform.position, target.position);
        
        // Reward for getting closer to target
        float distanceReward = 1f / (1f + distance);
        AddReward(distanceReward * 0.1f);
        
        // Reward for reaching target
        if (distance < 1f)
        {{
            AddReward(10f);
            EndEpisode();
        }}
        
        // Penalty for hitting obstacles
        foreach (Transform obstacle in obstacles)
        {{
            if (obstacle != null && Vector3.Distance(transform.position, obstacle.position) < 1f)
            {{
                AddReward(-1f);
            }}
        }}
        
        // Small penalty for time (encourage efficiency)
        AddReward(-0.01f);
    }}
    
    private Vector3 GetRandomPosition()
    {{
        float x = Random.Range(-8f, 8f);
        float z = Random.Range(-8f, 8f);
        return new Vector3(x, 0.5f, z);
    }}
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {{
        var continuousActionsOut = actionsOut.ContinuousActions;
        
        // Manual control for testing
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
        continuousActionsOut[2] = Input.GetAxis("Mouse X");
    }}
}}
"""
        
        script_path = os.path.join(project_path, f"Assets/Scripts/Agents/{agent.behavior_name}Agent.cs")
        with open(script_path, "w") as f:
            f.write(script_content)
    
    async def _generate_environment_script(self, project_path: str, env: EnvironmentConfig):
        """Generate Unity C# script for environment."""
        script_content = f"""
using UnityEngine;
using Unity.MLAgents;

public class {env.env_type.value.title()}Environment : MonoBehaviour
{{
    [Header("Environment Settings")]
    public int width = {env.size[0]};
    public int height = {env.size[1]};
    public GameObject wallPrefab;
    public GameObject targetPrefab;
    public GameObject agentPrefab;
    
    [Header("Physics")]
    public float gravity = {env.physics_settings.get('gravity', -9.81)}f;
    public float timeScale = {env.physics_settings.get('time_scale', 1.0)}f;
    
    private GameObject[,] grid;
    private List<GameObject> agents;
    private List<GameObject> targets;
    
    void Start()
    {{
        Time.timeScale = timeScale;
        Physics.gravity = new Vector3(0, gravity, 0);
        
        CreateEnvironment();
        SpawnAgents();
        SpawnTargets();
    }}
    
    void CreateEnvironment()
    {{
        grid = new GameObject[width, height];
        
        // Create walls around perimeter
        for (int x = 0; x < width; x++)
        {{
            for (int z = 0; z < height; z++)
            {{
                if (x == 0 || x == width - 1 || z == 0 || z == height - 1)
                {{
                    CreateWall(x, z);
                }}
            }}
        }}
        
        // Create obstacles
        foreach (var obstacle in {env.obstacles})
        {{
            CreateWall(obstacle[0], obstacle[1]);
        }}
    }}
    
    void CreateWall(int x, int z)
    {{
        if (wallPrefab != null)
        {{
            Vector3 position = new Vector3(x, 0.5f, z);
            GameObject wall = Instantiate(wallPrefab, position, Quaternion.identity);
            wall.transform.parent = transform;
            grid[x, z] = wall;
        }}
    }}
    
    void SpawnAgents()
    {{
        agents = new List<GameObject>();
        
        foreach (var spawn in {env.spawn_points})
        {{
            if (agentPrefab != null)
            {{
                Vector3 position = new Vector3(spawn[0], 0.5f, spawn[1]);
                GameObject agent = Instantiate(agentPrefab, position, Quaternion.identity);
                agent.transform.parent = transform;
                agents.Add(agent);
            }}
        }}
    }}
    
    void SpawnTargets()
    {{
        targets = new List<GameObject>();
        
        foreach (var goal in {env.goals})
        {{
            if (targetPrefab != null)
            {{
                Vector3 position = new Vector3(goal[0], 0.5f, goal[1]);
                GameObject target = Instantiate(targetPrefab, position, Quaternion.identity);
                target.transform.parent = transform;
                targets.Add(target);
            }}
        }}
    }}
    
    public void ResetEnvironment()
    {{
        // Reset all agents
        foreach (GameObject agent in agents)
        {{
            if (agent != null)
            {{
                var agentComponent = agent.GetComponent<Agent>();
                if (agentComponent != null)
                {{
                    agentComponent.EndEpisode();
                }}
            }}
        }}
        
        // Reset targets
        foreach (GameObject target in targets)
        {{
            if (target != null)
            {{
                // Randomize target position
                Vector3 randomPos = GetRandomPosition();
                target.transform.position = randomPos;
            }}
        }}
    }}
    
    private Vector3 GetRandomPosition()
    {{
        int x, z;
        do
        {{
            x = Random.Range(1, width - 1);
            z = Random.Range(1, height - 1);
        }}
        while (grid[x, z] != null);
        
        return new Vector3(x, 0.5f, z);
    }}
}}
"""
        
        script_path = os.path.join(project_path, f"Assets/Scripts/Environments/{env.env_type.value.title()}Environment.cs")
        with open(script_path, "w") as f:
            f.write(script_content)
    
    async def _generate_training_config(self, training_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training configuration."""
        return {
            "behaviors": {
                "DefaultBehavior": {
                    "trainer_type": "ppo",
                    "hyperparameters": {
                        "learning_rate": 3e-4,
                        "batch_size": 64,
                        "buffer_size": 12000,
                        "learning_starts": 1000,
                        "train_frequency": 4,
                        "target_network_frequency": 1000,
                        "tau": 1.0,
                        "gamma": 0.99,
                        "epsilon_start": 1.0,
                        "epsilon_end": 0.01,
                        "epsilon_decay": 0.995
                    },
                    "network_settings": {
                        "hidden_units": 128,
                        "num_layers": 2,
                        "vis_encode_type": "simple",
                        "normalize": True
                    },
                    "reward_signals": {
                        "extrinsic": {
                            "gamma": 0.99,
                            "strength": 1.0
                        }
                    },
                    "max_steps": 1000000,
                    "time_horizon": 64,
                    "summary_freq": 10000
                }
            },
            "environment_parameters": training_specs.get("environment_parameters", {}),
            "curriculum": training_specs.get("curriculum", {}),
            "checkpoint_settings": {
                "run_id": "iceburg_training",
                "initialize_from": None,
                "resume": False,
                "force": False,
                "train": True,
                "inference": False
            }
        }
    
    async def _write_training_config(self, project_path: str, config: Dict[str, Any]):
        """Write training configuration to file."""
        config_path = os.path.join(project_path, "Assets/ML-Agents/Config/training_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    async def _generate_project_files(self, project_path: str, project_name: str, description: str):
        """Generate Unity project files."""
        # Project settings
        project_settings = {
            "projectName": project_name,
            "description": description,
            "unityVersion": self.unity_version,
            "mlAgentsVersion": self.ml_agents_version,
            "createdTime": time.time()
        }
        
        settings_path = os.path.join(project_path, "ProjectSettings/ProjectSettings.asset")
        with open(settings_path, "w") as f:
            json.dump(project_settings, f, indent=2)
        
        # README
        readme_content = f"""# {project_name}

{description}

## Unity ML-Agents Project

This project was generated by ICEBURG Unity ML-Agents template.

### Requirements

- Unity {self.unity_version} or later
- ML-Agents {self.ml_agents_version} or later
- Python 3.8+ (for training)

### Setup

1. Open the project in Unity
2. Install ML-Agents package
3. Configure training parameters in Assets/ML-Agents/Config/
4. Run training with: `mlagents-learn config/training_config.yaml --run-id=iceburg_training`

### Training

```bash
# Start training
mlagents-learn Assets/ML-Agents/Config/training_config.yaml --run-id=iceburg_training

# Resume training
mlagents-learn Assets/ML-Agents/Config/training_config.yaml --run-id=iceburg_training --resume

# Run inference
mlagents-learn Assets/ML-Agents/Config/training_config.yaml --run-id=iceburg_training --inference
```

### Project Structure

- `Assets/Scripts/Agents/` - ML Agent scripts
- `Assets/Scripts/Environments/` - Environment scripts
- `Assets/ML-Agents/Config/` - Training configurations
- `Assets/ML-Agents/Models/` - Trained models
- `Assets/ML-Agents/Results/` - Training results
"""
        
        readme_path = os.path.join(project_path, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)
    
    async def _generate_ml_agents_config(self, project_path: str, agents: List[MLAgentConfig], environments: List[EnvironmentConfig]):
        """Generate ML-Agents configuration files."""
        # Behavior specifications
        for agent in agents:
            behavior_spec = {
                "behavior_name": agent.behavior_name,
                "team_id": 0,
                "index": 0,
                "vector_observation_space_size": agent.observation_space,
                "vector_action_space_size": [agent.action_space],
                "vector_action_space_type": "continuous",
                "max_steps": agent.max_steps
            }
            
            spec_path = os.path.join(project_path, f"Assets/ML-Agents/Config/{agent.behavior_name}.yaml")
            with open(spec_path, "w") as f:
                yaml.dump(behavior_spec, f, default_flow_style=False)
        
        # Environment configuration
        env_config = {
            "environments": [
                {
                    "name": env.env_type.value,
                    "type": env.env_type.value,
                    "size": list(env.size),
                    "obstacles": env.obstacles,
                    "goals": env.goals,
                    "spawn_points": env.spawn_points,
                    "physics": env.physics_settings
                }
                for env in environments
            ]
        }
        
        env_config_path = os.path.join(project_path, "Assets/ML-Agents/Config/environment_config.yaml")
        with open(env_config_path, "w") as f:
            yaml.dump(env_config, f, default_flow_style=False)
    
    async def generate_training_script(self, project_path: str, training_config: Dict[str, Any]):
        """Generate Python training script."""
        script_content = f"""#!/usr/bin/env python3
\"\"\"
ICEBURG Unity ML-Agents Training Script
Generated automatically by ICEBURG Unity ML template.
\"\"\"

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Train ICEBURG Unity ML-Agents')
    parser.add_argument('--config', default='Assets/ML-Agents/Config/training_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--run-id', default='iceburg_training',
                       help='Training run ID')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--inference', action='store_true',
                       help='Run inference mode')
    parser.add_argument('--env', default='Assets/Scenes/TrainingScene.unity',
                       help='Path to Unity environment')
    
    args = parser.parse_args()
    
    # Build command
    cmd = [
        'mlagents-learn',
        args.config,
        f'--run-id={{args.run_id}}',
        f'--env={{args.env}}'
    ]
    
    if args.resume:
        cmd.append('--resume')
    
    if args.inference:
        cmd.append('--inference')
    
    # Run training
    result = subprocess.run(cmd, cwd=project_path)
    
    if result.returncode == 0:
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
"""
        
        script_path = os.path.join(project_path, "train.py")
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
    
    async def generate_dockerfile(self, project_path: str):
        """Generate Dockerfile for training."""
        dockerfile_content = f"""FROM unityci/editor:ubuntu-{self.unity_version}-base-0.20.0

# Install ML-Agents
RUN pip install mlagents

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Set environment variables
ENV UNITY_LICENSE_FILE=/workspace/unity.ulf
ENV UNITY_EMAIL=your-email@example.com
ENV UNITY_PASSWORD=your-password

# Run training
CMD ["python", "train.py"]
"""
        
        dockerfile_path = os.path.join(project_path, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
    
    async def generate_github_actions(self, project_path: str):
        """Generate GitHub Actions workflow."""
        workflow_content = f"""name: ICEBURG Unity ML Training

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  train:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'
    
    - name: Install ML-Agents
      run: |
        pip install mlagents
        pip install -r requirements.txt
    
    - name: Run training
      run: |
        python train.py --config Assets/ML-Agents/Config/training_config.yaml --run-id=github_actions
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: training-results
        path: Assets/ML-Agents/Results/
"""
        
        workflow_path = os.path.join(project_path, ".github/workflows/train.yml")
        os.makedirs(os.path.dirname(workflow_path), exist_ok=True)
        with open(workflow_path, "w") as f:
            f.write(workflow_content)


# Convenience functions
async def create_unity_ml_template(config: Dict[str, Any] = None) -> UnityMLTemplate:
    """Create Unity ML template."""
    return UnityMLTemplate(config)


async def generate_unity_ml_project(project_name: str,
                                 description: str,
                                 template: UnityMLTemplate = None) -> UnityProject:
    """Generate Unity ML project."""
    if template is None:
        template = await create_unity_ml_template()
    
    # Default agent specifications
    agent_specs = [
        {
            "type": "reinforcement_learning",
            "behavior_name": "DefaultBehavior",
            "observation_space": 8,
            "action_space": 4,
            "max_steps": 1000,
            "curriculum": True
        }
    ]
    
    # Default environment specifications
    environment_specs = [
        {
            "type": "grid_world",
            "size": [10, 10],
            "obstacles": [[5, 5], [3, 7], [7, 3]],
            "goals": [[9, 9]],
            "spawn_points": [[1, 1], [2, 1], [1, 2]]
        }
    ]
    
    # Default training specifications
    training_specs = {
        "max_steps": 1000000,
        "curriculum": {
            "measure": "reward",
            "thresholds": [0.1, 0.3, 0.5, 0.7, 0.9],
            "min_lesson_length": 100,
            "signal_smoothing": True
        }
    }
    
    return await template.generate_project(
        project_name=project_name,
        description=description,
        agent_specs=agent_specs,
        environment_specs=environment_specs,
        training_specs=training_specs
    )
