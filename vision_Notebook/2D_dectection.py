import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO, SAM
model_path = Path("/home/jpacker/BinPickingProject/vision_Notebook/runs/detect/train2/weights/best.pt")
sam_path = "/home/jpacker/sam2.1_b.pt"
yolo_model = YOLO(model_path)
sam_model  = SAM(sam_path)

image_path = "/home/jpacker/sample_images/test_lidar/l515_bin1/rgb_0000.png"
image = cv2.imread(image_path)
h, w = image.shape[:2]

yolo_results = yolo_model(image)
bboxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)

combined_mask   = np.zeros((h, w), dtype=np.uint8)
per_object_masks = []

if len(bboxes) > 0:
    print(f"YOLO detected {len(bboxes)} objects. Passing boxes to SAM...")
    sam_results = sam_model(image, bboxes=bboxes)

    r = sam_results[0]
    if r.masks is not None:
        masks = (r.masks.data.cpu().numpy() * 255).astype(np.uint8)   # (N, H, W)
        per_object_masks = [masks[i] for i in range(masks.shape[0])]
        combined_mask = masks.max(axis=0)                              # (H, W)

    # save for the depth step
    np.save("/home/jpacker/sample_images/test_lidar/l515_bin1/combined_mask_0000.npy", combined_mask)


    # visualize
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(yolo_results[0].plot(img=image), cv2.COLOR_BGR2RGB))
    plt.title("YOLO detections"); plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(combined_mask, cmap='gray')
    plt.title("SAM combined mask"); plt.axis('off')
    plt.show()

for i, conf in enumerate(yolo_results[0].boxes.conf):
    print(f"Detection {i+1}: {conf.item():.4f}")

# --- Save top-3 confidence parts as separate .npy files ---
base = Path("/home/jpacker/sample_images/test_lidar/l515_bin1")
confs_all = yolo_results[0].boxes.conf.cpu().numpy()
bboxes_all = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)

TOP_K = 3
top_idx = np.argsort(confs_all)[::-1][:TOP_K]

top_bboxes = bboxes_all[top_idx]
top_confs  = confs_all[top_idx]

if len(top_bboxes) > 0:
    sam_top = sam_model(image, bboxes=top_bboxes)
    r_top = sam_top[0]
    if r_top.masks is not None:
        top_masks = (r_top.masks.data.cpu().numpy() * 255).astype(np.uint8)  # (K, H, W)
        for i in range(top_masks.shape[0]):
            out_path = f"{base}/part_mask_top{i+1}_conf{top_confs[i]:.3f}.npy"
            np.save(out_path, top_masks[i])
            print(f"Saved {out_path}")
img = np.load(base / "part_mask_top1_conf0.910.npy")
plt.imshow(img, cmap='gray')
plt.title("single depth"); plt.axis('off')
plt.show()