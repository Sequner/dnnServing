import io

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def receive_file(image_iterator, id):
    with open("input/" + str(id) + "_in.png", 'wb') as f:
        for chunk in image_iterator:
            f.write(chunk.image)
    return id

def save_mask(mask, img_id):
    image = Image.fromarray(mask.astype(np.uint8))
    image.save("output/" + str(img_id) + "_out.png")

# def convert_ItoB(image_path):
#     image = Image.open(image_path, mode='r')
#     image_bytes = io.BytesIO()
#     image.save(image_bytes, format='PNG')
#     return image_bytes.getvalue()

# def masks2rles(msks, ids, heights, widths):
#     pred_strings = []
#     pred_ids = []
#     pred_classes = []
#     for idx in range(msks.shape[0]):
#         height = heights[idx].item()
#         width = widths[idx].item()
#         msk = cv2.resize(msks[idx], 
#                          dsize=(width, height), 
#                          interpolation=cv2.INTER_NEAREST) # back to original shape
#         rle = [None]*3
#         for midx in [0, 1, 2]:
#             rle[midx] = mask2rle(msk[...,midx])
#         pred_strings.extend(rle)
#         pred_ids.extend([ids[idx]]*len(rle))
#         pred_classes.extend(['large_bowel', 'small_bowel', 'stomach'])
#     return pred_strings, pred_ids, pred_classes
