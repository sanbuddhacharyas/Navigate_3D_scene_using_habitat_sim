from glob import glob
from shutil import copyfile
import os

all_color_images = sorted(glob("./steve_file/color/*.jpg"))
all_depth_images = sorted(glob("./steve_file/depth/*.png"))

for color, depth in zip(all_color_images, all_depth_images):
    file_name_rgb = color.split('/')[-1].split('.')[0]
    file_name_depth = depth.split('/')[-1].split('.')[0]

    assert file_name_rgb == file_name_depth
    
    dst_color = '/'.join(color.split('/')[:-1]).replace('steve_file', 'san') + "/" + file_name_rgb.zfill(7) + '.jpg'
    dst_depth = '/'.join(depth.split('/')[:-1]).replace('steve_file', 'san') + "/" + file_name_depth.zfill(7) + '.png'

    os.makedirs(os.path.dirname(dst_color), exist_ok=True)
    os.makedirs(os.path.dirname(dst_depth), exist_ok=True)

    print(dst_color, dst_depth)

    copyfile(color, dst_color)
    copyfile(depth, dst_depth)