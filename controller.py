import os
import json
import logging
import threading
import shutil
from abc import ABC, abstractmethod
import psutil
import importlib.util
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgentInterface(ABC):
    """Abstract base class defining the essential methods for agents managed by MASTERMIND."""
    @abstractmethod
    def initialize(self):
        """Prepare the agent for execution."""
        pass

    @abstractmethod
    def execute(self):
        """Core logic of the agent."""
        pass

    @abstractmethod
    def get_data(self):
        """Retrieve execution results or data."""
        pass

    @abstractmethod
    def shutdown(self):
        """Clean up resources post-execution."""
        pass

class MASTERMIND:
    """Core class responsible for managing agent lifecycles within the MASTERMIND framework."""
    def __init__(self):
        self.agents = {}
        self.directories = ["agents", "tools", "executor"]
        self._setup_directories()
        self._load_agents()

    def _setup_directories(self):
        """Ensure required directories exist and are secure."""
        for directory in self.directories:
            os.makedirs(directory, exist_ok=True)
            os.chmod(directory, 0o755)

    def _load_agents(self):
        """Dynamically loads and initializes agents from specified directories."""
        for directory in self.directories:
            for filename in os.listdir(directory):
                if filename.endswith('.py') and directory == "agents":
                    self._load_agent_module(directory, filename)

    def _load_agent_module(self, directory, filename):
        """Load an agent module and initialize its class if it implements AgentInterface."""
        module_path = os.path.join(directory, filename)
        agent_name = filename[:-3]  # Remove '.py'
        spec = importlib.util.spec_from_file_location(agent_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isinstance(attribute, type) and issubclass(attribute, AgentInterface):
                agent_instance = attribute()
                agent_instance.initialize()
                self.agents[agent_name] = (agent_instance, filename)

    def execute_agents(self):
        """Executes all loaded agents concurrently in separate threads."""
        threads = []
        for agent_name, (agent_instance, filename) in self.agents.items():
            thread = threading.Thread(target=self._execute_single_agent, args=(agent_name, agent_instance, filename))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def _execute_single_agent(self, agent_name, agent_instance, filename):
        """Manage the lifecycle of a single agent and potentially save it as a tool."""
        try:
            logging.info(f"Starting execution of agent: {agent_name}")
            agent_instance.execute()
            data = agent_instance.get_data()
            logging.info(f"Agent {agent_name} executed successfully with data: {data}")
            agent_instance.shutdown()
            self._save_agent_as_tool(agent_name, filename)
        except Exception as e:
            logging.error(f"Error during execution of agent {agent_name}: {e}")

    def _save_agent_as_tool(self, agent_name, filename):
        """Save the agent's script to the 'tools' directory after successful execution."""
        source_path = os.path.join("agents", filename)
        destination_path = os.path.join("tools", filename)
        shutil.copy(source_path, destination_path)
        logging.info(f"Agent {agent_name} saved as a tool: {destination_path}")

if __name__ == "__main__":
    mastermind = MASTERMIND()
    mastermind.execute_agents()
    logging.info("All agents have been executed. MASTERMIND controller is shutting down.")
