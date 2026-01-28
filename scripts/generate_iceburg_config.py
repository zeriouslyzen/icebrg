#!/usr/bin/env python3
"""
Generate ICEBURG server configuration JSON from agent system prompts.
This replaces Grok's configuration structure with ICEBURG-specific information.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any

def extract_agent_prompts() -> Dict[str, Dict[str, str]]:
    """Extract all agent system prompts from Python files"""
    project_root = Path(__file__).parent.parent
    agents_dir = project_root / "src" / "iceburg" / "agents"
    prompts = {}
    
    if not agents_dir.exists():
        print(f"Warning: Agents directory not found: {agents_dir}")
        return prompts
    
    for file in agents_dir.glob("*.py"):
        try:
            content = file.read_text(encoding='utf-8')
            
            # Find all SYSTEM prompt definitions
            system_pattern = re.compile(r'([A-Z_]+_SYSTEM)\s*=\s*\(', re.MULTILINE)
            
            for match in system_pattern.finditer(content):
                name = match.group(1)
                start_pos = match.end()
                
                # Extract the multi-line string from parentheses
                # Handle Python string concatenation: "line1\n" + "line2\n"
                prompt_parts = []
                depth = 1
                i = start_pos
                current_string = ""
                in_string = False
                string_char = None
                escape_next = False
                
                while i < len(content) and depth > 0:
                    char = content[i]
                    
                    if escape_next:
                        current_string += char
                        escape_next = False
                        i += 1
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        current_string += char
                        i += 1
                        continue
                    
                    if not in_string:
                        if char in ('"', "'"):
                            in_string = True
                            string_char = char
                            # Check for triple quotes
                            if i + 2 < len(content) and content[i:i+3] == string_char * 3:
                                string_char = string_char * 3
                                i += 3
                                current_string = ""
                                continue
                            else:
                                i += 1
                                continue
                        elif char == '(':
                            depth += 1
                        elif char == ')':
                            depth -= 1
                            if depth == 0:
                                # Save last string part
                                if current_string:
                                    prompt_parts.append(current_string)
                                break
                    else:
                        # Check for end of string
                        if char == string_char:
                            # Check if it's the end of triple quotes
                            if len(string_char) == 3:
                                if i + 2 < len(content) and content[i:i+3] == string_char:
                                    # End of triple-quoted string
                                    prompt_parts.append(current_string)
                                    current_string = ""
                                    in_string = False
                                    string_char = None
                                    i += 3
                                    continue
                                else:
                                    current_string += char
                                    i += 1
                                    continue
                            else:
                                # End of regular string
                                prompt_parts.append(current_string)
                                current_string = ""
                                in_string = False
                                string_char = None
                                i += 1
                                # Skip whitespace and + operator
                                while i < len(content) and content[i] in (' ', '\t', '\n', '+'):
                                    i += 1
                                continue
                        else:
                            current_string += char
                    
                    i += 1
                
                if prompt_parts:
                    # Join all string parts
                    full_prompt = ''.join(prompt_parts)
                    # Clean up the prompt
                    full_prompt = full_prompt.strip()
                    # Handle escape sequences
                    full_prompt = full_prompt.replace('\\n', '\n').replace('\\t', '\t')
                    # Remove leading/trailing whitespace from each line
                    full_prompt = '\n'.join(line.rstrip() for line in full_prompt.split('\n'))
                    prompts[name] = {
                        "file": str(file.relative_to(project_root)),
                        "prompt": full_prompt
                    }
            
            # Also try simple patterns for non-parenthesized strings
            # Pattern 3: Triple quoted strings without parentheses
            pattern3 = re.compile(
                r'([A-Z_]+_SYSTEM)\s*=\s*"""((?:[^"]|"(?!""))*?)"""',
                re.DOTALL
            )
            for match in pattern3.finditer(content):
                name = match.group(1)
                if name not in prompts:  # Don't overwrite if already found
                    prompt = match.group(2).strip()
                    prompt = prompt.replace('\\n', '\n').replace('\\t', '\t')
                    prompt = '\n'.join(line.rstrip() for line in prompt.split('\n'))
                    prompts[name] = {
                        "file": str(file.relative_to(project_root)),
                        "prompt": prompt
                    }
                    
        except Exception as e:
            print(f"Warning: Error reading {file}: {e}")
            continue
    
    return prompts

def get_agent_metadata() -> Dict[str, Dict[str, Any]]:
    """Get metadata for each ICEBURG agent"""
    return {
        "surveyor": {
            "title": "Surveyor",
            "icon": "🔬",
            "description": "Gnostic research agent for deep analysis",
            "capabilities": ["research", "gnosis", "truth-finding", "uncensored-access"],
            "systemPromptKey": "SURVEYOR_SYSTEM"
        },
        "dissident": {
            "title": "Dissident",
            "icon": "⚡",
            "description": "Challenges consensus, explores alternatives",
            "capabilities": ["challenge", "alternatives", "paradigms", "uncensored-access"],
            "systemPromptKey": "DISSIDENT_SYSTEM"
        },
        "synthesist": {
            "title": "Synthesist",
            "icon": "🧬",
            "description": "Synthesizes information across domains",
            "capabilities": ["synthesis", "multimodal", "fusion", "evidence-integration"],
            "systemPromptKey": "SYNTHESIST_SYSTEM"
        },
        "oracle": {
            "title": "Oracle",
            "icon": "🔮",
            "description": "Meta-analyst formulating principles",
            "capabilities": ["meta-analysis", "principles", "evidence-weighting"],
            "systemPromptKey": "ORACLE_SYSTEM"
        },
        "archaeologist": {
            "title": "Archaeologist",
            "icon": "⛏️",
            "description": "Uncovers buried evidence and historical insights",
            "capabilities": ["research", "history", "suppressed-evidence"],
            "systemPromptKey": "ARCHAEOLOGIST_SYSTEM"
        },
        "scrutineer": {
            "title": "Scrutineer",
            "icon": "🔍",
            "description": "Forensic evidence analyst and validator",
            "capabilities": ["validation", "forensics", "evidence-grading"],
            "systemPromptKey": "SCRUTINEER_SYSTEM"
        },
        "supervisor": {
            "title": "Supervisor",
            "icon": "👁️",
            "description": "Quality control and validation",
            "capabilities": ["quality-control", "validation", "coordination"],
            "systemPromptKey": "SUPERVISOR_SYSTEM"
        },
        "scribe": {
            "title": "Scribe",
            "icon": "📝",
            "description": "Documents and structures knowledge",
            "capabilities": ["documentation", "structuring", "knowledge-synthesis"],
            "systemPromptKey": None  # No explicit system prompt found
        },
        "weaver": {
            "title": "Weaver",
            "icon": "🧵",
            "description": "Generates code and implementations",
            "capabilities": ["code-generation", "implementation", "weaving"],
            "systemPromptKey": None  # No explicit system prompt found
        },
        "ide_agent": {
            "title": "IDE Agent",
            "icon": "💻",
            "description": "Safe command execution and code editing",
            "capabilities": ["command-execution", "code-editing", "ide"],
            "systemPromptKey": "IDE_AGENT_SYSTEM"
        }
    }

def generate_iceburg_config(prompts: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Generate complete ICEBURG server configuration"""
    
    agent_metadata = get_agent_metadata()
    
    # Build agent presets with system prompts
    agent_presets = []
    for agent_id, metadata in agent_metadata.items():
        system_prompt = ""
        if metadata["systemPromptKey"] and metadata["systemPromptKey"] in prompts:
            system_prompt = prompts[metadata["systemPromptKey"]]["prompt"]
        elif agent_id == "scribe":
            system_prompt = "You are ICEBURG Scribe, responsible for documenting and structuring knowledge from Oracle principles. You transform abstract principles into structured, accessible knowledge formats."
        elif agent_id == "weaver":
            system_prompt = "You are ICEBURG Weaver, responsible for generating code and implementations from Oracle principles. You transform abstract principles into concrete, executable code."
        else:
            system_prompt = f"You are ICEBURG {metadata['title']}, {metadata['description']}."
        
        agent_presets.append({
            "agentId": agent_id,
            "title": metadata["title"],
            "systemPrompt": system_prompt,
            "icon": metadata["icon"],
            "description": metadata["description"],
            "capabilities": metadata["capabilities"]
        })
    
    config = {
        "status": "ready",
        "serverConfig": {
            "iceburg_version": "2.0",
            "iceburg_branding": "ICEBURG",
            "iceburg_description": "Truth-Finding AI Civilization",
            
            "models": [
                {
                    "modelId": "llama3.1:8b",
                    "name": "ICEBURG Fast",
                    "description": "Quick responses",
                    "modeDescription": "Fast mode for quick answers",
                    "modelMode": "ICEBURG_MODE_FAST",
                    "agent": "auto",
                    "tags": [],
                    "badgeText": "",
                    "isDefault": True
                },
                {
                    "modelId": "llama3.1:8b",
                    "name": "ICEBURG Surveyor",
                    "description": "Research Expert",
                    "modeDescription": "Deep research and gnostic analysis",
                    "modelMode": "ICEBURG_MODE_SURVEYOR",
                    "agent": "surveyor",
                    "tags": ["research", "gnosis"],
                    "badgeText": ""
                },
                {
                    "modelId": "llama3.1:8b",
                    "name": "ICEBURG Protocol",
                    "description": "Multi-Agent",
                    "modeDescription": "Full protocol with all agents",
                    "modelMode": "ICEBURG_MODE_PROTOCOL",
                    "agent": "protocol",
                    "tags": ["multi-agent"],
                    "badgeText": ""
                }
            ],
            
            "unavailableModels": [],
            "defaultFastModelId": "llama3.1:8b",
            "defaultExpertModelId": "llama3.1:8b",
            "defaultProtocolModelId": "llama3.1:8b",
            
            "agents": {
                "agentPresets": agent_presets
            },
            
            "model_mode_models": {
                "ICEBURG_MODE_FAST": {
                    "modelId": "llama3.1:8b",
                    "agent": "auto",
                    "description": "Quick responses",
                    "useSmallModels": True
                },
                "ICEBURG_MODE_SURVEYOR": {
                    "modelId": "llama3.1:8b",
                    "agent": "surveyor",
                    "description": "Deep research and analysis",
                    "useSmallModels": False
                },
                "ICEBURG_MODE_PROTOCOL": {
                    "modelId": "llama3.1:8b",
                    "agent": "protocol",
                    "description": "Full multi-agent protocol",
                    "useSmallModels": False
                },
                "ICEBURG_MODE_AUTO": {
                    "modelId": "llama3.1:8b",
                    "agent": "auto",
                    "description": "Auto-selects based on query complexity",
                    "useSmallModels": True
                }
            },
            
            "thinking_config": {
                "thinking_auto_open": False,
                "enable_single_thinking_different_summary_ui": True,
                "iceburg_thinking_use_js": False,
                "show_show_thoughts": True,
                "thinking_stream_enabled": True,
                "thinking_glitch_animation": True,
                "thinking_dropdown_enabled": True
            },
            
            "streaming_config": {
                "streaming_markdown_config": {
                    "cutLength": 50,
                    "maxHoldTimeMs": 300,
                    "isEnabled": True,
                    "wordBoundaryChunking": True
                },
                "chunk_delay": 0.02,
                "thinking_stream_poll_interval": 50
            },
            
            "timeline_navigator": {
                "enabled": True,
                "maxResponses": 100,
                "minResponses": 2,
                "minScreenWidth": 768,
                "highlightOnScroll": True,
                "showAgentBadges": True
            },
            
            "response_feedback": {
                "show_like_dropdown": True,
                "show_dislike_dropdown": True,
                "show_research_quality": True,
                "show_accuracy_feedback": True,
                "satisfaction_score": 3,
                "enable_feedback_storage": True
            },
            
            "feature_flags": {
                "enable_memory_toggle": True,
                "enable_text_to_speech": False,
                "enable_code_execution": True,
                "enable_file_sharing": True,
                "enable_mermaid_diagrams": True,
                "enable_sketchpad": False,
                "enable_iceburg_tasks": False,
                "enable_tool_composer": False,
                "enable_voice_mode": False,
                "enable_image_generation": False
            },
            
            "suggestions_config": {
                "enabled": True,
                "maxItems": 7,
                "maxItemsMobile": 3,
                "minChars": 1,
                "maxChars": 75,
                "throttleTimeMs": 250
            },
            
            "typeahead_config": {
                "enabled": True,
                "minChars": 1,
                "maxChars": 40,
                "maxResults": 7,
                "maxResultsMobile": 4,
                "maxWords": 50,
                "throttleTimeMs": 80
            }
        }
    }
    
    return config

def main():
    """Main function to generate and output ICEBURG config"""
    print("Extracting ICEBURG agent system prompts...")
    prompts = extract_agent_prompts()
    
    print(f"Found {len(prompts)} system prompts:")
    for name in sorted(prompts.keys()):
        print(f"  - {name}")
    
    print("\nGenerating ICEBURG configuration...")
    config = generate_iceburg_config(prompts)
    
    # Output as JSON
    json_output = json.dumps(config, indent=2, ensure_ascii=False)
    
    # Save to file
    output_file = Path("frontend/iceburg-config.json")
    output_file.write_text(json_output, encoding='utf-8')
    print(f"\n✅ Configuration saved to: {output_file}")
    
    # Also output to stdout for embedding
    print("\n" + "="*60)
    print("ICEBURG CONFIG JSON (for embedding in HTML):")
    print("="*60)
    print(json_output)
    
    return config

if __name__ == "__main__":
    main()

