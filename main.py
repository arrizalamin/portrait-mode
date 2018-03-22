import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import predict_depth
from deeplab_model import DeepLabModel

image_path = 'test_images/1521425591660.jpg'
pretrained_deeplabv3_path = 'models/deeplabv3_pascal_train_aug'
pretrained_monodepth_path = 'models/model_city2kitti/model_city2kitti'

MAX_INPUT_SIZE = 513

orig_img = cv2.imread(image_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
H, W, C = orig_img.shape
resize_ratio = 1.0 * MAX_INPUT_SIZE / max(H, W)
H, W = int(resize_ratio * H), int(resize_ratio * W)
orig_img = cv2.resize(orig_img, (W, H), cv2.INTER_AREA)

disp_pp = predict_depth.predict(pretrained_monodepth_path, orig_img)
disp_pp = cv2.resize(disp_pp.squeeze(), (W, H))
disp_pp = disp_pp / disp_pp.max()

model = DeepLabModel(pretrained_deeplabv3_path)

seg_map = model.run(orig_img)
obj_mask = seg_map > 0

result = orig_img.copy()
mask_viz = np.ones_like(obj_mask, dtype=np.float32)
threshs = [0.8, 0.5, 0.3]
kernels = [5, 9, 11]
fg_masks = [disp_pp < thresh for thresh in threshs]
for i, fg_mask in enumerate(fg_masks):
    kernel_size = kernels[i]
    blurred = cv2.GaussianBlur(orig_img, (kernel_size, kernel_size), 0)
    result[fg_mask] = blurred[fg_mask]
    mask_viz[fg_mask] = 1.0 - ((i + 1) / len(threshs))
result[obj_mask] = orig_img[obj_mask]
merged_mask = np.max([obj_mask.astype(np.float32),
                      mask_viz], axis=0)

output_directory = os.path.dirname(image_path)
output_name = os.path.splitext(os.path.basename(image_path))[0]

plt.imsave(os.path.join(output_directory, "{}_disp.png".format(
    output_name)), disp_pp, cmap='gray')
plt.imsave(os.path.join(output_directory, "{}_fg.png".format(
    output_name)), mask_viz, cmap='gray')
plt.imsave(os.path.join(output_directory, "{}_segmap.png".format(
    output_name)), obj_mask, cmap='gray')
plt.imsave(os.path.join(output_directory, "{}_mask.png".format(
    output_name)), merged_mask, cmap='gray')
plt.imsave(os.path.join(output_directory, "{}_blurred.png".format(
    output_name)), result)
plt.imsave(os.path.join(output_directory, "{}_resized.png".format(
    output_name)), orig_img)
