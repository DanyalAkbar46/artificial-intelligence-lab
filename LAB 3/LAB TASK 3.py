class ModelBasedReflexAgent:
    def __init__(self, desired_temperature):
        self.desired_temperature = desired_temperature
       
        self.previous_actions = {}

    def perceive(self, room, current_temperature):
        return room, current_temperature

    def act(self, room, current_temperature):
       
        if current_temperature < self.desired_temperature:
            action = "Turn on heater"
        else:
            action = "Turn off heater"

        
        if room in self.previous_actions and self.previous_actions[room] == action:
            return f"{room}: Current temperature = {current_temperature}°C. No action needed (already {action})."
        else:
            
            self.previous_actions[room] = action
            return f"{room}: Current temperature = {current_temperature}°C. {action}."



rooms = {
    "Living Room": 18,
    "Bedroom": 22,
    "Kitchen": 20,
    "Bathroom": 24
}

desired_temperature = 22
agent = ModelBasedReflexAgent(desired_temperature)


print("First cycle:")
for room, temperature in rooms.items():
    print(agent.act(room, temperature))

print("\nSecond cycle (memory prevents redundant actions):")
for room, temperature in rooms.items():
    print(agent.act(room, temperature))