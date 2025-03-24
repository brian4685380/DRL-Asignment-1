# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
import torch.nn as nn

# Define the Q-network (same as in train.py)
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x

# Load the Q-network checkpoint
def load_q_network(checkpoint_path, input_dim, output_dim, device):
    q_network = QNetwork(input_dim, output_dim).to(device)
    q_network.load_state_dict(torch.load(checkpoint_path, map_location=device))
    q_network.eval()
    return q_network

# Define the agent's action selection function
def get_action(state):
    state = torch.tensor(state, dtype=torch.float32).to(device)
    with torch.no_grad():
        action = torch.argmax(q_network(state)).item()
    return action

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Manually set input and output dimensions based on the environment
input_dim = 16  # Adjust this based on the state representation
output_dim = 6  # Number of actions

# Load the Q-network
q_network = load_q_network('q_network.pt', input_dim, output_dim, device)
