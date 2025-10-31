import torch
import torch.nn as nn
import torch.nn.functional as F

class TicTacToeNet(nn.Module):
    """
    A minimalistic dense neural network for Tic-Tac-Toe.
    
    Input: [batch_size, 9] tensor representing the board
           (1 for 'X', -1 for 'O', 0 for empty)
           
    Outputs:
    1. Policy: [batch_size, 9] tensor of probabilities for each move.
    2. Value:  [batch_size, 1] tensor of the board's evaluation 
               (from -1 [O wins] to +1 [X wins]).
    """
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        
        # The board is 3x3=9. We use a simple dense network.
        input_size = 9
        hidden_size = 256  # Wider but shallower - better for sparse inputs
        policy_output_size = 9 # 9 possible moves
        value_output_size = 1  # 1 evaluation score
        
        # --- Shared "Body" ---
        # Simpler: 2 layers instead of 3, but wider
        self.fc_body1 = nn.Linear(input_size, hidden_size)
        self.fc_body2 = nn.Linear(hidden_size, hidden_size)
        
        # --- 1. Policy "Head" ---
        # Direct path without bottleneck
        self.fc_policy = nn.Linear(hidden_size, policy_output_size)
        
        # --- 2. Value "Head" ---
        # This head decides *how good the position is*
        self.fc_value = nn.Linear(hidden_size, value_output_size)
        
        # Better initialization for ReLU networks
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Use He initialization for ReLU networks"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the network.
        'x' is the [batch_size, 9] board tensor.
        """
        
        # --- Pass through the shared body ---
        # Simpler: just 2 layers
        x = F.relu(self.fc_body1(x))
        x = F.relu(self.fc_body2(x))
        
        # --- Split into the two heads ---
        
        # 1. Policy Head - direct output, no bottleneck
        policy = F.softmax(self.fc_policy(x), dim=1)
        
        # 2. Value Head
        value = torch.tanh(self.fc_value(x))
        
        return policy, value




