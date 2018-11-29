import torch

def val_loss(net, val_loader, criterion, device):
  with torch.no_grad():
    for data in val_loader:
      images, labels = data
      images, labels = images.to(device), labels.to(device)
      outputs = net(images)
      loss = criterion(outputs, labels)
  return loss.item()