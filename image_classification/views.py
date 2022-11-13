from django.shortcuts import render
from .forms import *
import io
import os
import json
import base64
from torchvision import models
from torchvision import transforms
from PIL import Image
from django.conf import settings
# Create your views here.



model = models.densenet121(pretrained=True)
model.eval()

json_path = os.path.join(settings.STATIC_ROOT, "imagenet_class_index.json")
imagenet_mapping = json.load(open(json_path))

def transform_image(image_bytes):
    
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    class_name, human_label = imagenet_mapping[predicted_idx]
    return human_label




def index(request):
    image_url = None
    predicted_label = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_bytes = image.file.read()

            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_url = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

            try:
                predicted_label = get_prediction(image_bytes)
            except RuntimeError as re:
                print(re)
    
    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'image_url': image_url,
        'predicted_label': predicted_label,
    }
    return render(request, 'image_classification/index.html', context)