# Import necessary modules
from logic import LogicTables
from reasoning import SocraticReasoning

# BDI Classes Start

# Belief Class
class Belief:
    def __init__(self, belief):
        self.belief = belief
        self.logic = LogicTables()  # Initialize Logic class, assuming it handles logical operations
        self.reasoning = SocraticReasoning()  # Initialize SocraticReasoning class

    def process_belief(self):
        # Example method to process a belief using logic and reasoning
        self.logic.evaluate_expression(self.belief, {})  # Hypothetically evaluate expression
        self.reasoning.add_premise(self.belief)  # Adding premise for Socratic reasoning

    def __str__(self):
        return f"Belief: {self.belief}"

# Desire Class
class Desire:
    def __init__(self, goal):
        self.goal = goal

    def __str__(self):
        return f"Desire: {self.goal}"

# Intention Class
class Intention:
    def __init__(self, plan):
        self.plan = plan

    def execute(self):
        # Simulate the execution of a plan
        print(f"Executing plan: {self.plan}")

# Goal Class
class Goal:
    def __init__(self, name, conditions, priority=0):
        self.name = name
        self.conditions = conditions
        self.priority = priority

    def is_fulfilled(self, belief_system):
        # Check if goal conditions are fulfilled based on the current belief system
        return all(belief.process_belief() for belief in belief_system)

    def __str__(self):
        return f"Goal: {self.name}, Priority: {self.priority}"

# Reward Class
class Reward:
    def __init__(self):
        self.total_reward = 0

    def update_reward(self, goal):
        # Update the total reward based on the achievement of a goal
        if goal.is_fulfilled():
            self.total_reward += goal.priority

    def get_reward(self):
        return self.total_reward

# BDI Classes End
