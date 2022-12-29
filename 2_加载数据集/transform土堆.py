from PIL import Image
from torchvision import transforms

img_pth = "data/hymenoptera_data/train/ants/6240329_72c01e663e.jpg"
img = Image.open(img_pth)
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
print(img_tensor.shape)