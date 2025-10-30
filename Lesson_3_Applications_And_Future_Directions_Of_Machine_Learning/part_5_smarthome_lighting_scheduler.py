import numpy as np

class SmartHomeDevice:
    def __init__(self):
        self.lighting_schedule = np.zeros(24)  # assumes 24-hour schedule

    def adjust_lighting(self, hour, brightness):
        self.lighting_schedule[hour] = brightness
        print(f'Lighting adjusted to {brightness} at hour {hour}.')
