from abc import ABC, abstractmethod
from typing import Dict, Any, List
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.state = {}
        self.is_active = False
        self.last_update = datetime.utcnow()
        
    @abstractmethod
    async def initialize(self):
        """Initialize the agent"""
        pass
    
    @abstractmethod
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data"""
        pass
    
    @abstractmethod
    async def make_decision(self) -> Dict[str, Any]:
        """Make agent decision"""
        pass
    
    async def start(self):
        """Start the agent"""
        self.is_active = True
        await self.initialize()
        logger.info(f"Agent {self.agent_id} started")
    
    async def stop(self):
        """Stop the agent"""
        self.is_active = False
        logger.info(f"Agent {self.agent_id} stopped")
    
    def update_state(self, new_state: Dict[str, Any]):
        """Update agent state"""
        self.state.update(new_state)
        self.last_update = datetime.utcnow()
