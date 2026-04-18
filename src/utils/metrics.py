def calculate_accuracy(predictions, targets):
    return (predictions.argmax(dim=-1) == targets).float().mean().item()
