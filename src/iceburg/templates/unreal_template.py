"""
Unreal Engine Template for ICEBURG
Generates Unreal Engine projects with Blueprint integration and C++ components.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class UnrealProject:
    """Unreal Engine project structure."""
    project_name: str
    project_path: str
    features: List[str] = field(default_factory=list)
    blueprints: List[str] = field(default_factory=list)
    cpp_classes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnrealTemplate:
    """
    Unreal Engine template generator.
    
    Features:
    - Blueprint integration
    - C++ class generation
    - Game modes and actors
    - UI widgets
    - Animation blueprints
    - Material generation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Unreal template.
        
        Args:
            config: Template configuration
        """
        self.config = config or {}
        self.template_dir = self.config.get("template_dir", "templates/unreal")
        self.unreal_version = self.config.get("unreal_version", "5.3")
        
        # Ensure template directory exists
        os.makedirs(self.template_dir, exist_ok=True)
    
    async def generate_project(self, 
                             project_name: str,
                             description: str,
                             features: List[str] = None,
                             output_dir: str = None) -> UnrealProject:
        """
        Generate Unreal Engine project.
        
        Args:
            project_name: Project name
            description: Project description
            features: Project features
            output_dir: Output directory
            
        Returns:
            Generated Unreal project
        """
        if output_dir is None:
            output_dir = os.path.join(self.template_dir, project_name)
        
        project_path = Path(output_dir)
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create project structure
        await self._create_project_structure(project_path)
        
        # Generate project files
        await self._generate_project_files(project_path, project_name)
        
        # Generate C++ classes
        await self._generate_cpp_classes(project_path, project_name, features or [])
        
        # Generate Blueprints
        await self._generate_blueprints(project_path, project_name, features or [])
        
        # Generate content
        await self._generate_content(project_path, project_name, features or [])
        
        # Generate documentation
        await self._generate_documentation(project_path, project_name, description)
        
        project = UnrealProject(
            project_name=project_name,
            project_path=str(project_path),
            features=features or [],
            metadata={
                "description": description,
                "unreal_version": self.unreal_version,
                "created_time": time.time()
            }
        )
        
        logger.info(f"Generated Unreal project: {project_name}")
        return project
    
    async def _create_project_structure(self, project_path: Path):
        """Create Unreal project directory structure."""
        directories = [
            "Source",
            "Source/Public",
            "Source/Private",
            "Content",
            "Content/Blueprints",
            "Content/Materials",
            "Content/Textures",
            "Content/Meshes",
            "Content/Audio",
            "Content/UI",
            "Config",
            "Plugins",
            "Saved",
            "Intermediate",
            "Binaries"
        ]
        
        for directory in directories:
            (project_path / directory).mkdir(parents=True, exist_ok=True)
    
    async def _generate_project_files(self, project_path: Path, project_name: str):
        """Generate Unreal project files."""
        # .uproject file
        uproject_content = {
            "FileVersion": 3,
            "EngineAssociation": "5.3",
            "Category": "",
            "Description": "",
            "Modules": [
                {
                    "Name": project_name,
                    "Type": "Runtime",
                    "LoadingPhase": "Default"
                }
            ],
            "Plugins": [
                {
                    "Name": "EnhancedInput",
                    "Enabled": True
                },
                {
                    "Name": "GameplayTags",
                    "Enabled": True
                }
            ]
        }
        
        with open(project_path / f"{project_name}.uproject", "w") as f:
            json.dump(uproject_content, f, indent=4)
        
        # Build.cs file
        build_cs_content = f"""using UnrealBuildTool;

public class {project_name} : ModuleRules
{{
    public {project_name}(ReadOnlyTargetRules Target) : base(Target)
    {{
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] {{ "Core", "CoreUObject", "Engine", "InputCore" }});

        PrivateDependencyModuleNames.AddRange(new string[] {{ }});
    }}
}}"""
        
        with open(project_path / f"Source/{project_name}.Target.cs", "w") as f:
            f.write(build_cs_content)
        
        # Editor target
        editor_target_content = f"""using UnrealBuildTool;

public class {project_name}Editor : ModuleRules
{{
    public {project_name}Editor(ReadOnlyTargetRules Target) : base(Target)
    {{
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] {{ "Core", "CoreUObject", "Engine", "InputCore" }});

        PrivateDependencyModuleNames.AddRange(new string[] {{ "UnrealEd", "Slate", "SlateCore" }});
    }}
}}"""
        
        with open(project_path / f"Source/{project_name}Editor.Target.cs", "w") as f:
            f.write(editor_target_content)
    
    async def _generate_cpp_classes(self, project_path: Path, project_name: str, features: List[str]):
        """Generate C++ classes."""
        # Game Mode
        game_mode_h = f"""#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "{project_name}GameMode.generated.h"

UCLASS()
class {project_name.upper()}_API A{project_name}GameMode : public AGameModeBase
{{
    GENERATED_BODY()

public:
    A{project_name}GameMode();
}};"""
        
        with open(project_path / f"Source/Public/{project_name}GameMode.h", "w") as f:
            f.write(game_mode_h)
        
        game_mode_cpp = f"""#include "{project_name}GameMode.h"

A{project_name}GameMode::A{project_name}GameMode()
{{
    // Set default pawn class
    DefaultPawnClass = nullptr;
}}"""
        
        with open(project_path / f"Source/Private/{project_name}GameMode.cpp", "w") as f:
            f.write(game_mode_cpp)
        
        # Player Controller
        if "player_controller" in features:
            controller_h = f"""#pragma once

#include "CoreMinimal.h"
#include "GameFramework/PlayerController.h"
#include "{project_name}PlayerController.generated.h"

UCLASS()
class {project_name.upper()}_API A{project_name}PlayerController : public APlayerController
{{
    GENERATED_BODY()

public:
    A{project_name}PlayerController();

protected:
    virtual void BeginPlay() override;
    virtual void SetupInputComponent() override;

    // Input functions
    void MoveForward(float Value);
    void MoveRight(float Value);
    void Turn(float Value);
    void LookUp(float Value);
}};"""
            
            with open(project_path / f"Source/Public/{project_name}PlayerController.h", "w") as f:
                f.write(controller_h)
            
            controller_cpp = f"""#include "{project_name}PlayerController.h"
#include "GameFramework/Character.h"
#include "Engine/World.h"

A{project_name}PlayerController::A{project_name}PlayerController()
{{
    bShowMouseCursor = true;
    DefaultMouseCursor = EMouseCursor::Crosshairs;
}}

void A{project_name}PlayerController::BeginPlay()
{{
    Super::BeginPlay();
}}

void A{project_name}PlayerController::SetupInputComponent()
{{
    Super::SetupInputComponent();
    
    // Bind input actions
    InputComponent->BindAxis("MoveForward", this, &A{project_name}PlayerController::MoveForward);
    InputComponent->BindAxis("MoveRight", this, &A{project_name}PlayerController::MoveRight);
    InputComponent->BindAxis("Turn", this, &A{project_name}PlayerController::Turn);
    InputComponent->BindAxis("LookUp", this, &A{project_name}PlayerController::LookUp);
}}

void A{project_name}PlayerController::MoveForward(float Value)
{{
    if (Value != 0.0f)
    {{
        const FRotator Rotation = GetControlRotation();
        const FRotator YawRotation(0, Rotation.Yaw, 0);
        const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
        GetPawn()->AddMovementInput(Direction, Value);
    }}
}}

void A{project_name}PlayerController::MoveRight(float Value)
{{
    if (Value != 0.0f)
    {{
        const FRotator Rotation = GetControlRotation();
        const FRotator YawRotation(0, Rotation.Yaw, 0);
        const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);
        GetPawn()->AddMovementInput(Direction, Value);
    }}
}}

void A{project_name}PlayerController::Turn(float Value)
{{
    AddYawInput(Value);
}}

void A{project_name}PlayerController::LookUp(float Value)
{{
    AddPitchInput(Value);
}}"""
            
            with open(project_path / f"Source/Private/{project_name}PlayerController.cpp", "w") as f:
                f.write(controller_cpp)
        
        # Character class
        if "character" in features:
            character_h = f"""#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "{project_name}Character.generated.h"

UCLASS()
class {project_name.upper()}_API A{project_name}Character : public ACharacter
{{
    GENERATED_BODY()

public:
    A{project_name}Character();

protected:
    virtual void BeginPlay() override;
    virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

    // Input functions
    void MoveForward(float Value);
    void MoveRight(float Value);
    void Turn(float Value);
    void LookUp(float Value);
}};"""
            
            with open(project_path / f"Source/Public/{project_name}Character.h", "w") as f:
                f.write(character_h)
            
            character_cpp = f"""#include "{project_name}Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "Components/CapsuleComponent.h"

A{project_name}Character::A{project_name}Character()
{{
    PrimaryActorTick.bCanEverTick = true;
    
    // Set size for collision capsule
    GetCapsuleComponent()->InitCapsuleSize(42.f, 96.0f);
    
    // Configure character movement
    GetCharacterMovement()->bOrientRotationToMovement = true;
    GetCharacterMovement()->RotationRate = FRotator(0.0f, 540.0f, 0.0f);
    GetCharacterMovement()->JumpZVelocity = 600.f;
    GetCharacterMovement()->AirControl = 0.35f;
}}

void A{project_name}Character::BeginPlay()
{{
    Super::BeginPlay();
}}

void A{project_name}Character::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{{
    Super::SetupPlayerInputComponent(PlayerInputComponent);
    
    // Bind input actions
    PlayerInputComponent->BindAxis("MoveForward", this, &A{project_name}Character::MoveForward);
    PlayerInputComponent->BindAxis("MoveRight", this, &A{project_name}Character::MoveRight);
    PlayerInputComponent->BindAxis("Turn", this, &A{project_name}Character::Turn);
    PlayerInputComponent->BindAxis("LookUp", this, &A{project_name}Character::LookUp);
}}

void A{project_name}Character::MoveForward(float Value)
{{
    if (Value != 0.0f)
    {{
        const FRotator Rotation = Controller->GetControlRotation();
        const FRotator YawRotation(0, Rotation.Yaw, 0);
        const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
        AddMovementInput(Direction, Value);
    }}
}}

void A{project_name}Character::MoveRight(float Value)
{{
    if (Value != 0.0f)
    {{
        const FRotator Rotation = Controller->GetControlRotation();
        const FRotator YawRotation(0, Rotation.Yaw, 0);
        const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);
        AddMovementInput(Direction, Value);
    }}
}}

void A{project_name}Character::Turn(float Value)
{{
    AddControllerYawInput(Value);
}}

void A{project_name}Character::LookUp(float Value)
{{
    AddControllerPitchInput(Value);
}}"""
            
            with open(project_path / f"Source/Private/{project_name}Character.cpp", "w") as f:
                f.write(character_cpp)
    
    async def _generate_blueprints(self, project_path: Path, project_name: str, features: List[str]):
        """Generate Blueprint files."""
        # Game Mode Blueprint
        game_mode_bp = {
            "Class": "Blueprint",
            "Name": f"BP_{project_name}GameMode",
            "ParentClass": f"{project_name}GameMode",
            "Properties": {
                "DefaultPawnClass": "BP_ThirdPersonCharacter"
            }
        }
        
        with open(project_path / f"Content/Blueprints/BP_{project_name}GameMode.json", "w") as f:
            json.dump(game_mode_bp, f, indent=2)
        
        # Character Blueprint
        if "character" in features:
            character_bp = {
                "Class": "Blueprint",
                "Name": f"BP_{project_name}Character",
                "ParentClass": f"{project_name}Character",
                "Components": {
                    "Mesh": {
                        "Type": "SkeletalMeshComponent",
                        "Mesh": "SK_Mannequin"
                    },
                    "Camera": {
                        "Type": "CameraComponent",
                        "Location": [0, 0, 50]
                    }
                }
            }
            
            with open(project_path / f"Content/Blueprints/BP_{project_name}Character.json", "w") as f:
                json.dump(character_bp, f, indent=2)
        
        # UI Widget Blueprint
        if "ui" in features:
            ui_widget = {
                "Class": "UserWidget",
                "Name": f"WBP_{project_name}HUD",
                "Components": {
                    "Canvas": {
                        "Type": "CanvasPanel",
                        "Children": [
                            {
                                "Type": "TextBlock",
                                "Text": f"Welcome to {project_name}",
                                "FontSize": 24,
                                "Color": [1, 1, 1, 1]
                            }
                        ]
                    }
                }
            }
            
            with open(project_path / f"Content/UI/WBP_{project_name}HUD.json", "w") as f:
                json.dump(ui_widget, f, indent=2)
    
    async def _generate_content(self, project_path: Path, project_name: str, features: List[str]):
        """Generate content files."""
        # Input mapping
        input_mapping = {
            "Mappings": [
                {
                    "Name": "MoveForward",
                    "Key": "W",
                    "Scale": 1.0
                },
                {
                    "Name": "MoveForward",
                    "Key": "S",
                    "Scale": -1.0
                },
                {
                    "Name": "MoveRight",
                    "Key": "D",
                    "Scale": 1.0
                },
                {
                    "Name": "MoveRight",
                    "Key": "A",
                    "Scale": -1.0
                },
                {
                    "Name": "Turn",
                    "Key": "Mouse X",
                    "Scale": 1.0
                },
                {
                    "Name": "LookUp",
                    "Key": "Mouse Y",
                    "Scale": -1.0
                }
            ]
        }
        
        with open(project_path / "Config/DefaultInput.ini", "w") as f:
            f.write("[Input]\n")
            for mapping in input_mapping["Mappings"]:
                f.write(f"ActionMappings=({mapping['Name']},{mapping['Key']},{mapping['Scale']})\n")
        
        # Game settings
        game_settings = {
            "GameName": project_name,
            "GameVersion": "1.0.0",
            "DefaultMap": f"/Game/Maps/{project_name}Map",
            "DefaultGameMode": f"/Game/Blueprints/BP_{project_name}GameMode"
        }
        
        with open(project_path / "Config/DefaultGame.ini", "w") as f:
            f.write("[Game]\n")
            for key, value in game_settings.items():
                f.write(f"{key}={value}\n")
    
    async def _generate_documentation(self, project_path: Path, project_name: str, description: str):
        """Generate project documentation."""
        readme_content = f"""# {project_name}

{description}

## Unreal Engine Project

This project was generated by ICEBURG Unreal Engine template.

### Features

- C++ classes with Blueprint integration
- Input mapping
- Character controller
- UI widgets
- Material system

### Requirements

- Unreal Engine {self.unreal_version}+
- Visual Studio 2022 (Windows) or Xcode (macOS)
- Git

### Building

1. Open the project in Unreal Engine
2. Generate project files
3. Build the project

### Project Structure

- `Source/` - C++ source code
- `Content/` - Game assets and Blueprints
- `Config/` - Project configuration
- `Plugins/` - Custom plugins

### C++ Classes

- `{project_name}GameMode` - Main game mode
- `{project_name}Character` - Player character
- `{project_name}PlayerController` - Player controller

### Blueprints

- `BP_{project_name}GameMode` - Game mode Blueprint
- `BP_{project_name}Character` - Character Blueprint
- `WBP_{project_name}HUD` - UI widget

### Development

1. Open the project in Unreal Engine
2. Create new Blueprints or modify existing ones
3. Add C++ classes as needed
4. Test in the editor

### Deployment

1. Package the project for your target platform
2. Deploy to your distribution platform
"""
        
        with open(project_path / "README.md", "w") as f:
            f.write(readme_content)


# Convenience functions
async def create_unreal_template(config: Dict[str, Any] = None) -> UnrealTemplate:
    """Create Unreal template."""
    return UnrealTemplate(config)


async def generate_unreal_project(project_name: str,
                                description: str,
                                template: UnrealTemplate = None) -> UnrealProject:
    """Generate Unreal project."""
    if template is None:
        template = await create_unreal_template()
    
    return await template.generate_project(
        project_name=project_name,
        description=description,
        features=["character", "player_controller", "ui"]
    )
