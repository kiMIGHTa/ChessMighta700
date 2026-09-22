import torch
import torch.optim as optim
import torch.nn as nn
from trunk_layer import ChessNet
from training_model import train_loader, val_loader


model = ChessNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Adam runs the optimization algorithm with a learning rate of 0.001(1e-3) and updates the model's parameters based on the computed gradients from the loss function


best_val_loss = float('inf')  # initialize best validation loss to infinity
patience = 3  # number of epochs to wait for improvement before stopping
epochs_without_improvement = 0  # counter for epochs without improvement
num_epochs = 20

for epoch in range(num_epochs):
    model.train()  # puts the model in "training mode" (matters once we add dropout/batchnorm later)
    total_loss = 0

    for boards, from_labels, to_labels, promo_labels in train_loader:
        # zero out gradients from the previous step
        optimizer.zero_grad()

        # forward pass — get the three logits from the model
        from_logits, to_logits, promo_logits = model(boards) 

        # compute the loss 
        loss_from = criterion(from_logits, from_labels)
        loss_to = criterion(to_logits, to_labels)   
        loss_promo = criterion(promo_logits, promo_labels)
        loss = loss_from + loss_to + loss_promo

        # backward pass and optimization step
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {avg_loss:.4f}")


    # --- validation pass ---
    model.eval()  # evaluate mode — disables dropout and batchnorm layers 
    val_loss = 0

    with torch.no_grad():
        for boards, from_labels, to_labels, promo_labels in val_loader:
            # forward pass
            from_logits, to_logits, promo_logits = model(boards)

            # compute the loss (exactly the same as in training, but we don't do backward pass or optimizer step)
            loss_from = criterion(from_logits, from_labels)
            loss_to = criterion(to_logits, to_labels)
            loss_promo = criterion(promo_logits, promo_labels)
            loss = loss_from + loss_to + loss_promo

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Val loss: {avg_val_loss:.4f}")


    # check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pt")
        print(f"Validation loss improved. Model saved.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement in validation loss. Epochs without improvement: {epochs_without_improvement}/{patience}")

    # break the training loop if we have reached the patience limit
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break