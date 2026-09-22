import torch.optim as optim
import torch.nn as nn
from trunk_layer import ChessNet
from training_model import train_loader


model = ChessNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # puts the model in "training mode" (matters once we add dropout/batchnorm later)
    total_loss = 0

    for boards, from_labels, to_labels, promo_labels in train_loader:
        # zero out gradients from the previous step
        optimizer.zero_grad()

        # forward pass — get the three logits from the model
        from_logits, to_logits, promo_logits = model(boards) 

        # compute the three losses using criterion, then sum them into one `loss`
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