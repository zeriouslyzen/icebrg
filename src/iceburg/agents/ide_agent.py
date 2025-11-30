"""IDE Agent - Safe command execution and code editing agent."""

from typing import Dict, Any, List, Optional
import subprocess
import os
import time
import tempfile
from pathlib import Path
from ..config import IceburgConfig
from ..llm import chat_complete
from ..vectorstore import VectorStore

IDE_AGENT_SYSTEM = (
    "You are ICEBURG's IDE Agent, a specialized agent for safe command execution and code editing.\n"
    "Your role is to:\n"
    "- Execute commands safely in isolated environments\n"
    "- Edit and manage code files\n"
    "- Provide terminal/IDE-like functionality\n"
    "- Ensure security and prevent dangerous operations\n"
    "\n"
    "SAFETY CONSTRAINTS:\n"
    "- Never execute destructive commands (rm -rf, format, etc.) without explicit confirmation\n"
    "- Always use isolated environments (sandboxes, containers) when possible\n"
    "- Validate all file paths and prevent directory traversal attacks\n"
    "- Log all command executions for audit purposes\n"
    "- Never execute commands that could harm the system or user data\n"
    "\n"
    "CAPABILITIES:\n"
    "- Execute shell commands in safe environments\n"
    "- Read, write, and edit code files\n"
    "- Navigate file systems safely\n"
    "- Run code in isolated execution environments\n"
    "- Provide IDE-like features (syntax highlighting, linting, etc.)\n"
)


class IDEAgent:
    """IDE Agent for safe command execution and code editing."""
    
    def __init__(self, cfg: IceburgConfig, vs: VectorStore):
        self.cfg = cfg
        self.vs = vs
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix="iceburg_ide_"))
        self.command_history: List[Dict[str, Any]] = []
        
    def _is_safe_command(self, command: str) -> bool:
        """Check if command is safe to execute."""
        dangerous_patterns = [
            "rm -rf",
            "rm -r",
            "format",
            "mkfs",
            "dd if=",
            "> /dev/sd",
            "chmod 777",
            "sudo rm",
            "sudo format",
        ]
        command_lower = command.lower()
        return not any(pattern in command_lower for pattern in dangerous_patterns)
    
    def _validate_path(self, path: str) -> bool:
        """Validate file path to prevent directory traversal."""
        try:
            resolved = Path(path).resolve()
            # Ensure path is within sandbox or current working directory
            return str(resolved).startswith(str(self.sandbox_dir)) or str(resolved).startswith(os.getcwd())
        except Exception:
            return False
    
    def execute_command(self, command: str, cwd: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
        """Execute a command safely."""
        if not self._is_safe_command(command):
            return {
                "success": False,
                "error": "Command contains dangerous patterns and cannot be executed",
                "command": command
            }
        
        try:
            # Use sandbox directory if no cwd specified
            work_dir = cwd if cwd and self._validate_path(cwd) else str(self.sandbox_dir)
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Log command execution
            self.command_history.append({
                "command": command,
                "cwd": work_dir,
                "returncode": result.returncode,
                "timestamp": time.time()
            })
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": command
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "command": command
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": command
            }
    
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """Read a file safely."""
        if not self._validate_path(file_path):
            return {
                "success": False,
                "error": "Invalid file path or path traversal detected"
            }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                "success": True,
                "content": content,
                "file_path": file_path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Write a file safely."""
        if not self._validate_path(file_path):
            return {
                "success": False,
                "error": "Invalid file path or path traversal detected"
            }
        
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {
                "success": True,
                "file_path": file_path,
                "bytes_written": len(content.encode('utf-8'))
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def run(self, query: str, verbose: bool = False) -> str:
        """Process IDE-related queries."""
        context_chunks: List[str] = []
        if not self.cfg.fast:
            hits = self.vs.semantic_search(query, k=6)
            for h in hits:
                context_chunks.append(f"Source: {h.metadata.get('source', 'kb')}\n{h.document}")
        context_block = "\n\n".join(context_chunks)
        
        prompt = (
            f"QUERY: {query}\n\n"
            f"CONTEXT (established sources):\n{context_block if context_block else 'No local sources available.'}\n\n"
            "Analyze the query and determine if it requires:\n"
            "1. Command execution (provide command to execute)\n"
            "2. File operations (read/write files)\n"
            "3. Code editing (edit code files)\n"
            "4. General IDE assistance (provide guidance)\n\n"
            "Respond with a structured plan for how to handle this query safely."
        )
        
        try:
            result = chat_complete(
                self.cfg.surveyor_model,
                prompt,
                system=IDE_AGENT_SYSTEM,
                temperature=0.1,
                options={"num_ctx": 2048, "num_predict": 512},
                context_tag="IDEAgent"
            )
            if verbose:
                print("[IDE_AGENT] Analysis complete")
            return result
        except Exception as e:
            if verbose:
                print(f"[IDE_AGENT] Error: {e}")
            raise


def run(cfg: IceburgConfig, vs: VectorStore, query: str, verbose: bool = False) -> str:
    """Run IDE agent."""
    agent = IDEAgent(cfg, vs)
    return agent.run(query, verbose)

