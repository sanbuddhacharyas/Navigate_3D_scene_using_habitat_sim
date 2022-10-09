from glob import glob
from PIL import Image
from tqdm import tqdm

all_depth_image = glob("./steve_file/depth/*.png")
print(len(all_depth_image))
for img_path in tqdm(all_depth_image):
    img = Image.open(img_path)
    img.save(img_path)