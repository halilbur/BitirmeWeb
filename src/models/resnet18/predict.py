import torch
from torchvision import transforms
from PIL import Image
from .model import get_resnet18_model

# Flask projesindeki sınıf isimleriyle uyumlu olmalı
CLOTHING_CLASSES = [
    'Ceket', 'Elbise', 'Etek', 'Gömlek', 'Hırka', 'Kazak', 
    'Mont', 'Pantalon', 'Sweatshirt', 'Tshirt', 'Yelek', 'Şort'
]

# Inference için transform
inference_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def predict_image(image_path, weights_path, device='cpu'):
    model = get_resnet18_model(num_classes=len(CLOTHING_CLASSES), weights_path=weights_path, device=device)
    image = Image.open(image_path).convert('RGB')
    input_tensor = inference_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
        predicted_class = CLOTHING_CLASSES[pred_idx.item()]
    return predicted_class, conf.item(), probs.cpu().numpy()
