import numpy as np
import matplotlib.pyplot as plt

from habitat_sim.utils.data import ImageExtractor
import habitat_sim

# For viewing the extractor output
def display_sample(sample):
    img = sample["rgba"]
    depth = sample["depth"]
    # semantic = sample["semantic"]

    arr = [img, depth]
    titles = ["rgba", "depth"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 2, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()


scene_filepath = "/media/predator/2454920E5491E33A/3D_apartment_reconstruction/apartment_0/habitat/mesh_semantic.ply"

extractor = ImageExtractor(
    scene_filepath,
    img_size=(640, 480),
    output=["rgba", "depth"],
)


# Use the list of train outputs instead of the default, which is the full list
# of outputs (test + train)
extractor.set_mode('train')

# Index in to the extractor like a normal python list
# sample = extractor[0]

# Or use slicing
# samples = extractor[1:4]
for sample in extractor:
    display_sample(sample)

# Close the extractor so we can instantiate another one later
# (see close method for detailed explanation)
extractor.close()