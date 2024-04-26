import os
import json
import logging
import threading
from typing import Dict, Type, Union, List
from abc import ABC, abstractmethod
import psutil

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Auto-Configuration for config.json
def auto_configure():
    config_path = 'config.json'
    default_agents = ['SimpleCoder.py', 'autonomize.py']
    if not os.path.exists(config_path):
        config = {'agents': default_agents}
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file)
        logging.info('config.json created with default agents: SimpleCoder.py, autonomize.py')
    else:
        logging.info('config.json already exists. Skipping auto-configuration.')

# Calling auto-configuration function during the initialization
auto_configure()

# Agent Interface that all agents will implement
class AgentInterface(ABC):
    @abstractmethod
    def initialize(self):
        """ Initialize the agent """
        pass

    @abstractmethod
    def execute(self):
        """ Execute the agent's main functionality """
        pass

    @abstractmethod
    def get_data(self):
        """ Retrieve data produced by the agent """
        pass

    @abstractmethod
    def shutdown(self):
        """ Clean up resources used by the agent """
        pass

# MASTERMIND Class
class MASTERMIND:
    def __init__(self):
        self.agent_store: Dict[str, AgentInterface] = {}
        self.data_store: Dict[str, Union[str, Dict]] = {}
        self.load_config()
        
    def load_config(self):
        try:
            with open("config.json", "r") as f:
                self.config = json.load(f)
        except Exception as e:
            logging.error(f"Could not load config: {e}")

    def load_agent(self, agent_name: str, agent_class: Type[AgentInterface]):
        """ Load and initialize an agent by name and class type """
        if not self.validate_agent(agent_name):
            logging.error(f"Agent {agent_name} failed the security validation.")
            return
        try:
            agent_instance = agent_class()
            agent_instance.initialize()
            self.agent_store[agent_name] = agent_instance
        except Exception as e:
            logging.error(f"Failed to load agent {agent_name}: {e}")

    def unload_agent(self, agent_name: str):
        """ Unload and shutdown an agent by name """
        try:
            agent_instance = self.agent_store.pop(agent_name)
            agent_instance.shutdown()
        except KeyError:
            logging.error(f"Agent {agent_name} not found.")
        except Exception as e:
            logging.error(f"Failed to unload agent {agent_name}: {e}")

    def execute_agents(self):
        """ Execute all loaded agents in separate threads """
        threads = []
        for agent_name, agent_instance in self.agent_store.items():
            thread = threading.Thread(target=self.execute_single_agent, args=(agent_name, agent_instance,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def execute_single_agent(self, agent_name: str, agent_instance: AgentInterface):
        """ Execute a single agent and collect its data """
        try:
            agent_instance.execute()
            agent_data = agent_instance.get_data()
            self.accumulate_data(agent_name, agent_data)
        except Exception as e:
            logging.error(f"Failed to execute agent {agent_name}: {e}")

    def accumulate_data(self, agent_name: str, data: Union[str, Dict]):
        """ Validate and store data from an agent """
        if not self.validate_data(data):
            logging.error(f"Data from agent {agent_name} failed the validation check.")
            return
        self.data_store[agent_name] = data

    def get_data(self, agent_name: str):
        """ Retrieve stored data for a specific agent """
        return self.data_store.get(agent_name, "Data not found.")

    def validate_agent(self, agent_name: str) -> bool:
        """ Validate if an agent is allowed to be loaded """
        return agent_name in self.config.get("allowed_agents", [])

    def validate_data(self, data: Union[str, Dict]) -> bool:
        """ Placeholder for data validation logic """
        return True

    def monitor_resources(self):
        """ Monitor and log CPU and memory usage """
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        logging.info(f"CPU Usage: {cpu_percent}%")
        logging.info(f"Memory Usage: {memory_info.percent}%")

# Save data store to JSON file
def save_data_store(mastermind_instance: MASTERMIND):
    try:
        with open("data_store.json", "w") as f:
            json.dump(mastermind_instance.data_store, f)
    except Exception as e:
        logging.error(f"Failed to save data store: {e}")

# Example of a simple agent that implements the AgentInterface
class SimpleAgent(AgentInterface):
    def initialize(self):
        self.data = "Initialized"

    def execute(self):
        self.data = "Executed"

    def get_data(self):
        return self.data

    def shutdown(self):
        self.data = "Shutdown"

# Test the MASTERMIND class with a simple agent
if __name__ == "__main__":
    mastermind = MASTERMIND()
    mastermind.load_agent("SimpleAgent", SimpleAgent)
    mastermind.execute_agents()
    save_data_store(mastermind)
    mastermind.monitor_resources()
