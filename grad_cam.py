import os
import cv2
import torch
import numpy as np

from PIL import Image

from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def generate_gradcam(
    model,
    image_path,
    save_path="static/heatmaps/heatmap.png"
):

    model.eval()


    pil_image = Image.open(
        image_path
    ).convert("RGB")


    rgb_img = np.array(
        pil_image.resize((380, 380))
    ).astype(np.float32) / 255.0



    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(DEVICE)


    target_layers = [
        model.conv_head
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )


    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]


    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )



    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )


    cv2.imwrite(
        save_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    return save_path