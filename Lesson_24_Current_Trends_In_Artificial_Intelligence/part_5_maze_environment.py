class MazeEnv:
    def __init__(self, size):
        self.size = size
        self.state = (0, 0)  # Starting point
        self.goal = (size-1, size-1)  # Goal point

    def step(self, action):
        if action == 'up':
            self.state = (max(0, self.state[0]-1), self.state[1])
        elif action == 'down':
            self.state = (min(self.size-1, self.state[0]+1), self.state[1])
        elif action == 'left':
            self.state = (self.state[0], max(0, self.state[1]-1))
        elif action == 'right':
            self.state = (self.state[0], min(self.size-1, self.state[1]+1))
        
        return self.state

maze = MazeEnv(size=5)

# Simulating agent steps
for action in ['right', 'down', 'down', 'right', 'up']:
    new_state = maze.step(action)
    print(f'Agent moved {action}, new state: {new_state}')
