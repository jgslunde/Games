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
        hidden_size = 128  # Larger for perfect play (was 32)
        policy_output_size = 9 # 9 possible moves
        value_output_size = 1  # 1 evaluation score
        
        # --- Shared "Body" ---
        # This part learns a general "understanding" of the position
        # Added extra layer for more capacity
        self.fc_body1 = nn.Linear(input_size, hidden_size)
        self.fc_body2 = nn.Linear(hidden_size, hidden_size)
        self.fc_body3 = nn.Linear(hidden_size, hidden_size)
        
        # --- 1. Policy "Head" ---
        # This head decides *which move to play*
        # Add intermediate layer to reduce bottleneck
        self.fc_policy_1 = nn.Linear(hidden_size, 64)
        self.fc_policy_2 = nn.Linear(64, policy_output_size)
        
        # --- 2. Value "Head" ---
        # This head decides *how good the position is*
        self.fc_value = nn.Linear(hidden_size, value_output_size)

    def forward(self, x):
        """
        Forward pass of the network.
        'x' is the [batch_size, 9] board tensor.
        """
        
        # --- Pass through the shared body ---
        # We use ReLU as our activation function
        x = F.relu(self.fc_body1(x))
        x = F.relu(self.fc_body2(x))
        x = F.relu(self.fc_body3(x))
        
        # --- Split into the two heads ---
        
        # 1. Policy Head
        # We output probabilities for each of the 9 moves.
        # Use intermediate layer to avoid bottleneck (128→64→9 instead of 128→9)
        policy_hidden = F.relu(self.fc_policy_1(x))
        policy = F.softmax(self.fc_policy_2(policy_hidden), dim=1)
        
        # 2. Value Head
        # We output a single score from -1 to +1.
        # Tanh is the perfect activation for this, as it squashes
        # any real number into the [-1, 1] range.
        value = torch.tanh(self.fc_value(x))
        
        return policy, value




