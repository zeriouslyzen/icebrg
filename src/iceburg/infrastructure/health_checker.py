"""
ICEBURG Infrastructure Health Checker
Implements robust health checks and auto-recovery for critical components
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, Any, Optional
import os

# Optional aiohttp import
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False


class HealthChecker:
    """Comprehensive health checking system for ICEBURG infrastructure"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.health_status = {}
        self.last_checks = {}
        self.retry_counts = {}
        self.max_retries = 3
        
    async def check_ollama_health(self) -> Dict[str, Any]:
        """Check Ollama service health with auto-recovery"""
        check_key = "ollama"
        
        try:
            # Check if Ollama process is running
            result = subprocess.run(
                ["pgrep", "-f", "ollama"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode != 0:
                if self.verbose:
                    print("Ollama process not found, attempting to start...")
                
                # Try to start Ollama
                start_result = await self._start_ollama()
                if not start_result:
                    return {
                        "status": "failed",
                        "error": "Ollama not running and failed to start",
                        "retry_count": self.retry_counts.get(check_key, 0)
                    }
            
            # Test Ollama API endpoint
            if AIOHTTP_AVAILABLE:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    try:
                        host = os.getenv("HOST", "localhost")
                        async with session.get(f"http://{host}:11434/api/tags") as response:
                            if response.status == 200:
                                self.health_status[check_key] = "healthy"
                                self.last_checks[check_key] = datetime.now()
                                self.retry_counts[check_key] = 0
                                
                                if self.verbose:
                                    print("Ollama API is healthy")
                                
                                return {
                                    "status": "healthy",
                                    "response_time": response.headers.get("X-Response-Time", "unknown"),
                                    "last_check": datetime.now().isoformat()
                                }
                            else:
                                raise Exception(f"HTTP {response.status}")
                                
                    except Exception as e:
                        self.retry_counts[check_key] = self.retry_counts.get(check_key, 0) + 1
                        
                        if self.retry_counts[check_key] >= self.max_retries:
                            return {
                                "status": "failed",
                                "error": f"Ollama API check failed after {self.max_retries} retries: {e}",
                                "retry_count": self.retry_counts[check_key]
                            }
                        
                        if self.verbose:
                            print(f"Ollama API check failed, retrying... (attempt {self.retry_counts[check_key]})")
                        
                        # Wait before retry
                        await asyncio.sleep(2 ** self.retry_counts[check_key])
                        return await self.check_ollama_health()
            else:
                # Fallback: use curl to test Ollama API
                try:
                    result = subprocess.run(
                        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"http://{os.getenv('HOST', 'localhost')}:11434/api/tags"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0 and result.stdout.strip() == "200":
                        self.health_status[check_key] = "healthy"
                        self.last_checks[check_key] = datetime.now()
                        self.retry_counts[check_key] = 0
                        
                        if self.verbose:
                            print("Ollama API is healthy (curl fallback)")
                        
                        return {
                            "status": "healthy",
                            "response_time": "unknown",
                            "last_check": datetime.now().isoformat(),
                            "method": "curl_fallback"
                        }
                    else:
                        raise Exception(f"HTTP {result.stdout.strip()}")
                        
                except Exception as e:
                    self.retry_counts[check_key] = self.retry_counts.get(check_key, 0) + 1
                    
                    if self.retry_counts[check_key] >= self.max_retries:
                        return {
                            "status": "failed",
                            "error": f"Ollama API check failed after {self.max_retries} retries: {e}",
                            "retry_count": self.retry_counts[check_key]
                        }
                    
                    if self.verbose:
                        print(f"Ollama API check failed, retrying... (attempt {self.retry_counts[check_key]})")
                    
                    # Wait before retry
                    await asyncio.sleep(2 ** self.retry_counts[check_key])
                    return await self.check_ollama_health()
                    
        except Exception as e:
            self.retry_counts[check_key] = self.retry_counts.get(check_key, 0) + 1
            return {
                "status": "failed",
                "error": f"Ollama health check failed: {e}",
                "retry_count": self.retry_counts[check_key]
            }
    
    async def _start_ollama(self) -> bool:
        """Attempt to start Ollama service"""
        try:
            if self.verbose:
                print("Attempting to start Ollama service...")
            
            # Try to start Ollama in background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            
            # Wait for service to start
            for attempt in range(10):
                await asyncio.sleep(2)
                try:
                    if AIOHTTP_AVAILABLE:
                        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                            host = os.getenv("HOST", "localhost")
                        async with session.get(f"http://{host}:11434/api/tags") as response:
                                if response.status == 200:
                                    if self.verbose:
                                        print("Ollama service started successfully")
                                    return True
                    else:
                        # Use curl fallback
                        result = subprocess.run(
                            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"http://{os.getenv('HOST', 'localhost')}:11434/api/tags"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0 and result.stdout.strip() == "200":
                            if self.verbose:
                                print("Ollama service started successfully (curl fallback)")
                            return True
                except:
                    continue
            
            if self.verbose:
                print("Failed to start Ollama service after multiple attempts")
            return False
            
        except Exception as e:
            if self.verbose:
                print(f"Error starting Ollama: {e}")
            return False
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources (CPU, memory, disk)"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = "healthy"
            warnings = []
            
            if cpu_percent > 90:
                status = "warning"
                warnings.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 90:
                status = "warning"
                warnings.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                status = "warning"
                warnings.append(f"High disk usage: {disk.percent}%")
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "warnings": warnings,
                "last_check": datetime.now().isoformat()
            }
            
        except ImportError:
            return {
                "status": "unknown",
                "error": "psutil not available for system monitoring",
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"System resource check failed: {e}",
                "last_check": datetime.now().isoformat()
            }
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and health"""
        try:
            db_path = os.getenv("ICEBURG_DATA_DIR", "./data")
            sovereign_db = os.path.join(db_path, "sovereign_library.db")
            
            if not os.path.exists(sovereign_db):
                return {
                    "status": "warning",
                    "message": "Database file not found, will be created on first use",
                    "last_check": datetime.now().isoformat()
                }
            
            # Check if database is accessible
            import sqlite3
            conn = sqlite3.connect(sovereign_db, timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            conn.close()
            
            return {
                "status": "healthy",
                "table_count": table_count,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Database health check failed: {e}",
                "last_check": datetime.now().isoformat()
            }
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check on all components"""
        if self.verbose:
            print("Running comprehensive health check...")
        
        start_time = time.time()
        
        # Run all health checks in parallel
        ollama_task = asyncio.create_task(self.check_ollama_health())
        system_task = asyncio.create_task(self.check_system_resources())
        db_task = asyncio.create_task(self.check_database_health())
        
        results = await asyncio.gather(
            ollama_task,
            system_task,
            db_task,
            return_exceptions=True
        )
        
        ollama_result, system_result, db_result = results
        
        # Handle exceptions
        if isinstance(ollama_result, Exception):
            ollama_result = {"status": "error", "error": str(ollama_result)}
        if isinstance(system_result, Exception):
            system_result = {"status": "error", "error": str(system_result)}
        if isinstance(db_result, Exception):
            db_result = {"status": "error", "error": str(db_result)}
        
        # Determine overall health
        overall_status = "healthy"
        critical_failures = []
        
        if ollama_result.get("status") == "failed":
            overall_status = "critical"
            critical_failures.append("Ollama service unavailable")
        elif ollama_result.get("status") == "error":
            overall_status = "warning"
        
        if system_result.get("status") == "error":
            overall_status = "warning"
        
        if db_result.get("status") == "error":
            overall_status = "warning"
        
        duration = time.time() - start_time
        
        health_report = {
            "overall_status": overall_status,
            "critical_failures": critical_failures,
            "components": {
                "ollama": ollama_result,
                "system": system_result,
                "database": db_result
            },
            "check_duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.verbose:
            status_emoji = "✅" if overall_status == "healthy" else "⚠️" if overall_status == "warning" else "❌"
            print(f"{status_emoji} System health: {overall_status}")
            if critical_failures:
                for failure in critical_failures:
                    print(f"  ❌ {failure}")
        
        return health_report
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of current health status"""
        return {
            "health_status": self.health_status,
            "last_checks": {k: v.isoformat() if v else None for k, v in self.last_checks.items()},
            "retry_counts": self.retry_counts
        }


# Global health checker instance
_health_checker = HealthChecker()


async def pre_run_health_check(verbose: bool = False) -> bool:
    """
    Run health check before starting ICEBURG operations
    Returns True if system is ready, False if critical issues found
    """
    global _health_checker
    _health_checker.verbose = verbose
    
    health_report = await _health_checker.comprehensive_health_check()
    
    if health_report["overall_status"] == "critical":
        if verbose:
            for failure in health_report["critical_failures"]:
                print(f"  ❌ {failure}")
        return False
    
    if verbose:
        print("✅ System health check passed")
    
    return True


async def get_system_health() -> Dict[str, Any]:
    """Get current system health status"""
    global _health_checker
    return await _health_checker.comprehensive_health_check()
