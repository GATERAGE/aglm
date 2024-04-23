class Rule:
    def __init__(self, conditions, conclusions):
        if not isinstance(conditions, set) or not isinstance(conclusions, set):
            raise ValueError("Conditions and conclusions must be sets.")
        self.conditions = conditions
        self.conclusions = conclusions

    def applies(self, beliefs):
        return self.conditions.issubset(beliefs)

class Default:
    def __init__(self, conditions, conclusions):
        if not isinstance(conditions, set) or not isinstance(conclusions, set):
            raise ValueError("Conditions and conclusions must be sets.")
        self.conditions = conditions
        self.conclusions = conclusions

    def applies(self, beliefs):
        return self.conditions.isdisjoint(beliefs)

class DefaultLogic:
    def __init__(self):
        self.rules = []
        self.defaults = []

    def add_rule(self, rule):
        if not isinstance(rule, Rule):
            raise ValueError("Rule must be an instance of Rule.")
        self.rules.append(rule)

    def add_default(self, default):
        if not isinstance(default, Default):
            raise ValueError("Default must be an instance of Default.")
        self.defaults.append(default)

    def evaluate(self, query):
        beliefs = set()
        new_beliefs = set()

        while True:
            new_beliefs.clear()

            for rule in self.rules:
                if rule.applies(beliefs):
                    new_beliefs.update(rule.conclusions)

            for default in self.defaults:
                if default.applies(beliefs):
                    new_beliefs.update(default.conclusions)

            if new_beliefs.issubset(beliefs):
                break

            beliefs.update(new_beliefs)

        return query in beliefs

if __name__ == "__main__":
    # Example usage
    dl = DefaultLogic()
    dl.add_rule(Rule({"A", "B"}, {"C"}))
    dl.add_default(Default({"D"}, {"E"}))
    print("Query 'C' concluded:", dl.evaluate("C"))
