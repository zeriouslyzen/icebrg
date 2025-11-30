"""
Integration tests for service layer
"""

import pytest
from src.iceburg.services import ProtocolService, AgentService, StorageService, CacheService
from src.iceburg.interfaces import ProtocolBase, AgentBase, StorageBase
from src.iceburg.config import IceburgConfig


class TestProtocol(ProtocolBase):
    """Test protocol implementation"""
    
    def execute(self, query: str, context=None):
        return {"result": f"Processed: {query}"}


class TestAgent(AgentBase):
    """Test agent implementation"""
    
    def run(self, query: str, context=None):
        return f"Agent processed: {query}"


class TestStorage(StorageBase):
    """Test storage implementation"""
    pass


@pytest.fixture
def config():
    """Create test configuration"""
    return IceburgConfig()


@pytest.fixture
def protocol_service(config):
    """Create protocol service"""
    service = ProtocolService(config)
    service.initialize({})
    service.start()
    return service


@pytest.fixture
def agent_service(config):
    """Create agent service"""
    service = AgentService(config)
    service.initialize({})
    service.start()
    return service


@pytest.fixture
def storage_service(config):
    """Create storage service"""
    service = StorageService(config)
    service.initialize({})
    service.start()
    return service


@pytest.fixture
def cache_service(config):
    """Create cache service"""
    service = CacheService(config)
    service.initialize({})
    service.start()
    return service


def test_protocol_service_initialization(protocol_service):
    """Test protocol service initialization"""
    assert protocol_service.initialized
    assert protocol_service.running
    assert protocol_service.health_check()["healthy"]


def test_protocol_service_registration(protocol_service):
    """Test protocol registration"""
    protocol = TestProtocol("test", ["test"])
    protocol_service.register_protocol("test", protocol)
    assert "test" in protocol_service.get_available_protocols()


def test_protocol_service_execution(protocol_service):
    """Test protocol execution"""
    protocol = TestProtocol("test", ["test"])
    protocol_service.register_protocol("test", protocol)
    result = protocol_service.execute_protocol("test", "test query")
    assert result["result"] == "Processed: test query"


def test_agent_service_initialization(agent_service):
    """Test agent service initialization"""
    assert agent_service.initialized
    assert agent_service.running
    assert agent_service.health_check()["healthy"]


def test_agent_service_registration(agent_service):
    """Test agent registration"""
    agent = TestAgent("test", ["test"])
    agent_service.register_agent("test", agent)
    assert "test" in agent_service.get_available_agents()


def test_agent_service_execution(agent_service):
    """Test agent execution"""
    agent = TestAgent("test", ["test"])
    agent_service.register_agent("test", agent)
    result = agent_service.run_agent("test", "test query")
    assert result == "Agent processed: test query"


def test_storage_service_initialization(storage_service):
    """Test storage service initialization"""
    assert storage_service.initialized
    assert storage_service.running
    assert storage_service.health_check()["healthy"]


def test_storage_service_registration(storage_service):
    """Test storage registration"""
    storage = TestStorage("test")
    storage_service.register_storage("test", storage)
    assert "test" in storage_service.get_available_storages()


def test_cache_service_initialization(cache_service):
    """Test cache service initialization"""
    assert cache_service.initialized
    assert cache_service.running
    assert cache_service.health_check()["healthy"]


def test_cache_service_operations(cache_service):
    """Test cache operations"""
    cache_service.set("key1", "value1", ttl=60)
    assert cache_service.get("key1") == "value1"
    cache_service.delete("key1")
    assert cache_service.get("key1") is None

