import os
import cv2

def get_centroid(bbox):
    """Calculate the centroid of a bounding box.
    Args:
        bbox: Tuple of (x, y, w, h) representing the bounding box
    Returns:
        Tuple of (cx, cy) representing the centroid coordinates
    """
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    return (cx, cy)

def save_yolo_annotations(frame_idx, tracks, frame_shape, out_dir, class_id=0):
    """Save YOLO-format annotation (.txt) for one frame.
    Args:
        frame_idx: integer frame index (used for filename and index file)
        tracks: iterable of tracked objects (each should expose .rectangle -> (x,y,w,h) or be (x,y,w,h))
        frame_shape: (height, width, channels) or (height, width)
        out_dir: directory where .txt files and index.txt will be written
        class_id: integer class id to use in YOLO file (default 0)
    """
    os.makedirs(out_dir, exist_ok=True)
    img_h, img_w = frame_shape[0], frame_shape[1]

    txt_name = f"{frame_idx:06d}.txt"
    txt_path = os.path.join(out_dir, txt_name)

    with open(txt_path, "w") as f:
        for obj in tracks:
            # skip ambiguous tracks (likely multi-player crops)
            if hasattr(obj, "ambiguous") and obj.ambiguous:
                continue
            # try common ways to get bbox
            if hasattr(obj, "rectangle"):
                x, y, w, h = obj.rectangle
            elif hasattr(obj, "bbox"):
                x, y, w, h = obj.bbox
            elif isinstance(obj, (list, tuple)) and len(obj) == 4:
                x, y, w, h = obj
            else:
                # skip if bbox not available
                continue

            if w <= 0 or h <= 0:
                continue

            # convert to YOLO normalized format: class x_center y_center width height (all in 0..1)
            x_c = (x + w / 2.0) / img_w
            y_c = (y + h / 2.0) / img_h
            w_n = w / img_w
            h_n = h / img_h

            # clamp to [0,1]
            x_c = min(max(x_c, 0.0), 1.0)
            y_c = min(max(y_c, 0.0), 1.0)
            w_n = min(max(w_n, 0.0), 1.0)
            h_n = min(max(h_n, 0.0), 1.0)

            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

    # append mapping (frame_idx -> txt filename) to an index file for convenience
    index_path = os.path.join(out_dir, "index.txt")
    with open(index_path, "a") as idxf:
        idxf.write(f"{frame_idx} {txt_name}\n")