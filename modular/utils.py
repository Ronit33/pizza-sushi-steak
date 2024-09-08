import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple

from PIL import Image

# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    

def pred_and_plot_image(model: nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device = "cpu"):
    # 1. Open Image
    img = Image.open(image_path)

    # 2. Create transformation for image
    if transform is not None:
        image_transforms = transform
    else:
        image_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    # 3. Model on target device
    model.to(device)

    # 4. Turn on evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # 5. transforming and adding batch to image
        transformed_image = image_transforms(img).unsqueeze(dim=0)
        
        # 6. making predictions
        target_image_pred = model(transformed_image.to(device))
    
    # 7. Convert logits to prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 8. Convert prediction probabilities to prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    # create model save path
    assert model_name.endswith(".pth") or model_name.endswith('.pt'), \
        "model_name should end with `.pt` or `.pth`"
        
    model_save_path = target_dir_path / model_name
    
    # save model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)