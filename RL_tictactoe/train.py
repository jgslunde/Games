import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime

from TTTnet import TicTacToeNet
from selfplay import SelfPlay, TrainingDataset
from evaluate import evaluate_against_random, print_elo_report

class Trainer:
    """
    Handles training of the TicTacToe neural network using self-play data.
    """
    def __init__(
        self,
        network,
        lr=0.001,
        weight_decay=1e-4,
        device='cpu'
    ):
        self.network = network
        self.device = device
        self.network.to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler - reduce LR when learning plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,      # Reduce LR by 50% when plateau detected
            patience=5,      # Wait 5 iterations before reducing (was 3)
            min_lr=1e-6      # Lower minimum to allow finer tuning (was 1e-5)
        )
        
        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Training history
        self.history = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': []
        }
    
    def train_epoch(self, dataloader, policy_weight=1.0, value_weight=1.0):
        """
        Train for one epoch on the given data.
        
        Args:
            dataloader: PyTorch DataLoader with training data
            policy_weight: Weight for policy loss (default 1.0)
            value_weight: Weight for value loss (default 1.0)
        
        Returns:
            avg_total_loss, avg_policy_loss, avg_value_loss
        """
        self.network.train()
        
        total_loss_sum = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        num_batches = 0
        
        for boards, target_policies, target_values in dataloader:
            # Move to device
            boards = boards.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device).unsqueeze(1)  # [batch, 1]
            
            # Forward pass
            pred_policies, pred_values = self.network(boards)
            
            # Compute losses
            # Policy loss: cross-entropy between target and predicted distributions
            policy_loss = self.policy_loss_fn(pred_policies, target_policies)
            
            # Value loss: MSE between target and predicted values
            value_loss = self.value_loss_fn(pred_values, target_values)
            
            # Combined loss
            total_loss = policy_weight * policy_loss + value_weight * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss_sum += total_loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            num_batches += 1
        
        # Calculate averages
        avg_total_loss = total_loss_sum / num_batches
        avg_policy_loss = policy_loss_sum / num_batches
        avg_value_loss = value_loss_sum / num_batches
        
        # Record history
        self.history['total_loss'].append(avg_total_loss)
        self.history['policy_loss'].append(avg_policy_loss)
        self.history['value_loss'].append(avg_value_loss)
        
        return avg_total_loss, avg_policy_loss, avg_value_loss
    
    def step_scheduler(self, metric):
        """
        Step the learning rate scheduler based on a metric (e.g., validation loss).
        
        Args:
            metric: The metric to track (lower is better)
        """
        self.scheduler.step(metric)
    
    def train_on_data(
        self,
        training_data,
        num_epochs=10,
        batch_size=32,
        policy_weight=1.0,
        value_weight=1.0,
        validation_split=0.1,
        verbose=True
    ):
        """
        Train the network on self-play data with train/validation split.
        
        Args:
            training_data: List of (board, policy, value) tuples
            num_epochs: Number of epochs to train
            batch_size: Batch size for training
            policy_weight: Weight for policy loss
            value_weight: Weight for value loss
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress
        """
        # Split into train and validation sets
        n_total = len(training_data)
        n_val = int(n_total * validation_split)
        n_train = n_total - n_val
        
        # Shuffle and split
        import random
        shuffled_data = training_data.copy()
        random.shuffle(shuffled_data)
        
        train_data = shuffled_data[:n_train]
        val_data = shuffled_data[n_train:]
        
        # Create datasets and dataloaders
        train_dataset = TrainingDataset(train_data)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        val_dataset = TrainingDataset(val_data)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        if verbose:
            print(f"\nTraining on {n_train} examples, validating on {n_val} examples, {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            # Training
            avg_train_total, avg_train_policy, avg_train_value = self.train_epoch(
                train_dataloader,
                policy_weight=policy_weight,
                value_weight=value_weight
            )
            
            # Validation
            avg_val_total, avg_val_policy, avg_val_value = self._validate_epoch(
                val_dataloader,
                policy_weight=policy_weight,
                value_weight=value_weight
            )
            
            # Check for improvement
            if avg_val_total < best_val_loss:
                best_val_loss = avg_val_total
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Detect overfitting
            overfitting_indicator = ""
            if avg_val_total > avg_train_total * 1.2:
                overfitting_indicator = " ⚠️ OVERFITTING"
            elif epochs_without_improvement >= 3:
                overfitting_indicator = " ⚠️ VAL NOT IMPROVING"
            
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {avg_train_total:.4f} | "
                      f"Val Loss: {avg_val_total:.4f}"
                      f"{overfitting_indicator}")
        
        # Summary
        if verbose:
            if epochs_without_improvement >= 3:
                print("⚠️  Training may have converged or overfitting detected")
        
        # Return best validation loss for scheduler
        return best_val_loss
    
    def _validate_epoch(self, dataloader, policy_weight=1.0, value_weight=1.0):
        """
        Validate the network on held-out data.
        
        Returns:
            avg_total_loss, avg_policy_loss, avg_value_loss
        """
        self.network.eval()  # Set to evaluation mode
        
        total_loss_sum = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        num_batches = 0
        
        with torch.no_grad():
            for boards, target_policies, target_values in dataloader:
                # Move to device
                boards = boards.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device).unsqueeze(1)
                
                # Forward pass
                pred_policies, pred_values = self.network(boards)
                
                # Compute losses
                policy_loss = self.policy_loss_fn(pred_policies, target_policies)
                value_loss = self.value_loss_fn(pred_values, target_values)
                total_loss = policy_weight * policy_loss + value_weight * value_loss
                
                # Accumulate
                total_loss_sum += total_loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                num_batches += 1
        
        # Calculate averages
        avg_total_loss = total_loss_sum / num_batches
        avg_policy_loss = policy_loss_sum / num_batches
        avg_value_loss = value_loss_sum / num_batches
        
        return avg_total_loss, avg_policy_loss, avg_value_loss
    
    def save_checkpoint(self, filepath, iteration=None, metadata=None):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            iteration: Training iteration number (optional)
            metadata: Additional metadata to save (optional)
        """
        checkpoint = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'iteration': iteration,
            'metadata': metadata
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        
        Returns:
            checkpoint dictionary with metadata
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint


def train_loop(
    num_iterations=10,
    games_per_iteration=100,
    num_mcts_sims=50,
    epochs_per_iteration=10,
    batch_size=32,
    lr=0.001,
    checkpoint_dir='checkpoints',
    device='cpu',
    eval_games=100,
    eval_mcts_sims=50,
    replay_buffer_size=10000,
    random_opponent_fraction=0.3,
    num_workers=None
):
    """
    Main training loop: self-play -> train -> repeat.
    
    Args:
        num_iterations: Number of training iterations
        games_per_iteration: Number of self-play games per iteration
        num_mcts_sims: Number of MCTS simulations per move
        epochs_per_iteration: Number of training epochs per iteration
        batch_size: Training batch size
        lr: Learning rate
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on ('cpu' or 'cuda')
        eval_games: Number of games to play against random player for evaluation
        eval_mcts_sims: Number of MCTS simulations for evaluation games
        replay_buffer_size: Maximum size of experience replay buffer (0 to disable)
        random_opponent_fraction: Fraction of games against random opponent (0.0-1.0)
        num_workers: Number of parallel workers for game generation (None = auto)
    """
    print("=" * 60)
    print("Starting AlphaZero-style Training for Tic-Tac-Toe")
    print("=" * 60)
    
    # Initialize network
    network = TicTacToeNet().to(device)
    
    # Initialize trainer
    trainer = Trainer(network, lr=lr, device=device)
    
    # Initialize self-play
    self_play = SelfPlay(network, num_simulations=num_mcts_sims)
    
    # Track ELO progression
    elo_history = []
    
    # Experience replay buffer (stores training data from multiple iterations)
    replay_buffer = []
    
    # Initial evaluation removed (baseline is consistently ~50% vs random)
    
    # Training loop
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # 1. Generate self-play data
        print(f"\n[1/4] Generating self-play data ({games_per_iteration} games)...")
        training_data, game_stats = self_play.generate_training_data(
            num_games=games_per_iteration,
            verbose=True,
            use_augmentation=True,  # Keep augmentation - it's genuinely helpful
            random_opponent_fraction=random_opponent_fraction,
            num_workers=num_workers
        )
        
        # Add to replay buffer
        replay_buffer.extend(training_data)
        
        # Determine what to train on
        if replay_buffer_size == 0:
            # No replay buffer - train only on current iteration
            training_subset = training_data
            print(f"Training on current iteration: {len(training_subset)} examples")
        else:
            # Use replay buffer with sliding window
            if len(replay_buffer) > replay_buffer_size:
                replay_buffer = replay_buffer[-replay_buffer_size:]
            
            # For early iterations, limit buffer to avoid training too much on random play
            effective_buffer_size = min(len(replay_buffer), max(2000, len(training_data) * 2))
            training_subset = replay_buffer[-effective_buffer_size:]
            
            print(f"Replay buffer: {len(replay_buffer)} total, training on {len(training_subset)} recent examples")
        
        # 2. Train on selected data
        print("\n[2/4] Training network...")
        best_val_loss = trainer.train_on_data(
            training_subset,
            num_epochs=epochs_per_iteration,
            batch_size=batch_size,
            verbose=True
        )
        
        # Step learning rate scheduler based on validation loss
        trainer.step_scheduler(best_val_loss)
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        if current_lr <= 1e-5:
            print("⚠️  Learning rate at minimum - may need higher min_lr or different scheduler")
        
        # 3. Save checkpoint
        print("\n[3/4] Saving checkpoint...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_iter{iteration+1:03d}_{timestamp}.pt"
        )
        
        # 4. Evaluate against random player (ensure parallel evaluation)
        print("\n[4/4] Evaluating against random player...")
        eval_workers = os.cpu_count() if (num_workers is None or num_workers == 0) else num_workers
        eval_results = evaluate_against_random(
            network,
            num_games=eval_games,
            num_mcts_sims=eval_mcts_sims,
            verbose=True,
            num_workers=eval_workers
        )
        
        # Calculate and display ELO
        estimated_elo = print_elo_report(eval_results, opponent_elo=1000)
        elo_history.append({
            'iteration': iteration + 1,
            'elo': estimated_elo,
            'win_rate': eval_results['win_rate'],
            'results': eval_results
        })
        
        # Save checkpoint with evaluation results
        trainer.save_checkpoint(
            checkpoint_path,
            iteration=iteration + 1,
            metadata={
                'game_stats': game_stats,
                'eval_results': eval_results,
                'estimated_elo': estimated_elo
            }
        )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Print ELO progression summary
    if elo_history:
        print("\n" + "=" * 60)
        print("ELO PROGRESSION SUMMARY")
        print("=" * 60)
        for entry in elo_history:
            print(f"Iteration {entry['iteration']:2d}: ELO {entry['elo']:4.0f} "
                  f"(Win rate: {entry['win_rate']*100:5.1f}%)")
        print("=" * 60)
    
    return network, trainer, elo_history


if __name__ == "__main__":
    # Required for multiprocessing on Windows/Mac
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    # Configure training - OPTIMIZED FOR >99% WIN RATE VS RANDOM
    NUM_ITERATIONS = 50          # More iterations for convergence
    GAMES_PER_ITERATION = 400    # More games = better data quality
    NUM_MCTS_SIMS = 400          # MUCH stronger signal (was 200) - more computation but better targets
    EPOCHS_PER_ITERATION = 15    # More epochs to learn from augmented data
    BATCH_SIZE = 64              # Larger batch for stable gradients (was 32)
    LEARNING_RATE = 0.001        # Back to conservative LR - quick learning suggests good gradient signal
    REPLAY_BUFFER_SIZE = 0       # DISABLED - train only on current iteration
    RANDOM_OPPONENT_FRACTION = 1.0  # 100% random - correct approach for beating random
    NUM_WORKERS = None           # None = use all CPU cores
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("\nTraining Configuration:")
    print("  Network: 2-layer (256 hidden units) + direct policy head")
    print("  Replay buffer: DISABLED (training on current iteration only)")
    print("  Data augmentation: ENABLED (8x symmetries)") 
    print(f"  Games/iteration: {GAMES_PER_ITERATION}")
    print(f"  MCTS simulations: {NUM_MCTS_SIMS}")
    print(f"  Epochs per iteration: {EPOCHS_PER_ITERATION}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Initial learning rate: {LEARNING_RATE} (with adaptive scheduler)")
    print("  LR reduction: 0.5x when validation plateaus (patience=5, min_lr=1e-6)")
    print(f"  Random opponent: {RANDOM_OPPONENT_FRACTION*100:.0f}% of games")
    print(f"  Parallel workers: {NUM_WORKERS if NUM_WORKERS else 'auto (all cores)'}")
    print("\nGoal: X >99%, O 80-90% win rate against random player")
    
    # Run training
    network, trainer, elo_history = train_loop(
        num_iterations=NUM_ITERATIONS,
        games_per_iteration=GAMES_PER_ITERATION,
        num_mcts_sims=NUM_MCTS_SIMS,
        epochs_per_iteration=EPOCHS_PER_ITERATION,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        checkpoint_dir='checkpoints',
        device=device,
        eval_games=200,              # More evaluation games for accuracy
        eval_mcts_sims=400,          # MUST match training MCTS strength
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        random_opponent_fraction=RANDOM_OPPONENT_FRACTION,
        num_workers=NUM_WORKERS
    )
    
    # Save final model
    final_path = os.path.join('checkpoints', 'final_model.pt')
    
    # Get final evaluation results if available
    final_metadata = {'final': True}
    if elo_history:
        final_metadata['elo_history'] = elo_history
        final_metadata['final_elo'] = elo_history[-1]['elo']
    
    trainer.save_checkpoint(final_path, iteration=NUM_ITERATIONS, metadata=final_metadata)
    print(f"\nFinal model saved to {final_path}")
