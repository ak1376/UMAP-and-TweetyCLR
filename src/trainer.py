from loss import TripletLoss
import torch
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, validation_loader, optimizer, device, max_steps, margin = 1.0, early_stopping = True, eval_interval = 500, trailing_avg_window=1000, patience = 8) -> None:
        self.model = model
        self.train_loader = train_loader 
        self.validation_loader = validation_loader
        self.optimizer = optimizer 
        self.device = device
        self.trailing_avg_window = trailing_avg_window
        self.patience = patience
        self.max_steps = max_steps
        self.margin = margin
        self.eval_interval = eval_interval
        self.early_stopping = early_stopping

    def validation(self, step, anchor_validation, pos_sample_validation, neg_sample_validation):
        self.model.eval()
        with torch.no_grad():
            anchor_rep, pos_rep, neg_rep = self.model.forward(anchor_validation, pos_sample_validation, neg_sample_validation)
            loss = TripletLoss(anchor_rep, pos_rep, neg_rep, margin = self.margin)

            return loss.item()

    def moving_average(self, values, window):
        """Simple moving average over a list of values"""
        if len(values) < window:
            # Return an empty list or some default value if there are not enough values to compute the moving average
            return []
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma.tolist()

    def train(self):
        raw_loss_list = []
        raw_val_loss_list = []
        train_iter = iter(self.train_loader)
        validation_iter = iter(self.validation_loader)
        step = 0
        steps_since_improvement = 0
        best_val_loss = float('inf')

        while step < self.max_steps:
            
            try:
                anchor_train, pos_sample_train, neg_sample_train, idx_train = next(train_iter)
                anchor_validation, pos_sample_validation, neg_sample_validation, idx_validation = next(validation_iter)
        
            except Exception as e:
                train_iter = iter(self.train_loader)
                validation_iter = iter(self.validation_loader)
                step = 0

                continue

            anchor_train = anchor_train.to(self.device).to(torch.float32)
            pos_sample_train = pos_sample_train.to(self.device).to(torch.float32)
            neg_sample_train = neg_sample_train.to(self.device).to(torch.float32)

            anchor_validation = anchor_validation.to(self.device).to(torch.float32)
            pos_sample_validation = pos_sample_validation.to(self.device).to(torch.float32)
            neg_sample_validation = neg_sample_validation.to(self.device).to(torch.float32)

            self.model = self.model.to(self.device)
            self.model.train()

            anchor_rep, pos_rep, neg_rep = self.model.forward(anchor_train, pos_sample_train, neg_sample_train)

            loss = TripletLoss(anchor_rep, pos_rep, neg_rep, margin = self.margin)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate raw losses
            raw_loss_list.append(loss.item())

            # Perform validation and accumulate metrics
            val_loss = self.validation(step, anchor_validation, pos_sample_validation, neg_sample_validation)
            raw_val_loss_list.append(val_loss)

            if step % self.eval_interval == 0 or step == 0:
                # Apply moving average if there are enough data points
                if len(raw_loss_list) >= self.trailing_avg_window:
                    smoothed_training_loss = self.moving_average(raw_loss_list, self.trailing_avg_window)[-1]
                    smoothed_val_loss = self.moving_average(raw_val_loss_list, self.trailing_avg_window)[-1]
                else:
                    # Use the current values if not enough data points for moving average
                    smoothed_training_loss = raw_loss_list[-1]
                    smoothed_val_loss = val_loss

                print(f'Step [{step}/{self.max_steps}], '
                    f'Smoothed Training Loss: {smoothed_training_loss:.4f}, '
                    f'Smoothed Validation Loss: {smoothed_val_loss:.4f}')

                # Update best_val_loss and steps_since_improvement
                if smoothed_val_loss < best_val_loss:
                    best_val_loss = smoothed_val_loss
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1

                if self.early_stopping and steps_since_improvement >= self.patience:
                    print(f"Early stopping triggered at step {step}. No improvement for {self.patience} evaluation intervals.")
                    break
                
            step +=1








    