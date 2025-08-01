import torch
import numpy as np
import cv2
from nodes import MAX_RESOLUTION

class FRED_AutoImageTile_from_Mask_v1:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "auto_from_mask_bbox": ("BOOLEAN", { "default": True }),
                "Auto_grid_strategy": (["max_rows_cols", "min_rows_cols", "max_rows_min_cols", "min_rows_max_cols", "min_balanced", "max_balanced"], {"default": "min_balanced"}),
                "confidence": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
                "manual_rows": ("INT", { "default": 3, "min": 1, "max": 256, "step": 1 }),
                "manual_cols": ("INT", { "default": 3, "min": 1, "max": 256, "step": 1 }),
                "overlap": ("FLOAT", { "default": 0, "min": 0, "max": 0.5, "step": 0.01 }),
                "overlap_x": ("INT", { "default": 56, "min": 0, "max": MAX_RESOLUTION//2, "step": 1 }),
                "overlap_y": ("INT", { "default": 56, "min": 0, "max": MAX_RESOLUTION//2, "step": 1 }),
                "preview_bbox_color": ("COLOR", {"default": "#FFC800"},),
            },
            "optional": {
                "MASK": ("MASK",),
                "BBOX": ("BBOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGES", "preview", "tile_width", "tile_height", "overlap_x", "overlap_y", "rows", "columns")
    FUNCTION = "execute"
    CATEGORY = "👑FRED/image/postprocessing"

    def execute(self, image, manual_rows, manual_cols, overlap, overlap_x, overlap_y, preview_bbox_color, Auto_grid_strategy, auto_from_mask_bbox=False, confidence=1.0, MASK=None, BBOX=None):
        # h, w = image.shape[1:3]
        rows = manual_rows
        cols = manual_cols

        if len(image.shape) == 4:
            _, h, w, _ = image.shape
        elif len(image.shape) == 3:
            _, h, w = image.shape
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # --- Auto rows/cols logic ---
        if auto_from_mask_bbox:
            mask_bbox = None
            if MASK is not None:
                mask_np = MASK.cpu().numpy() if hasattr(MASK, 'cpu') else MASK
                mask_np = np.squeeze(mask_np)
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                elif mask_np.ndim == 4:
                    mask_np = mask_np[0, 0]
                ys, xs = np.where(mask_np > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    mask_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            elif BBOX is not None:
                if BBOX is not None:
                    # If BBOX is a torch tensor or numpy array, convert to a flat tuple
                    if hasattr(BBOX, "cpu"):
                        bbox_vals = BBOX.cpu().numpy().flatten()
                    else:
                        bbox_vals = np.array(BBOX).flatten()
                    if bbox_vals.shape[0] == 4:
                        mask_bbox = tuple(bbox_vals)
                    else:
                        mask_bbox = None
            min_grid = 2
            if mask_bbox is not None:
                x, y, bw, bh = mask_bbox
                
                # confidence: 1.0 = original bbox, <1.0 = shrink bbox
                scaling = max(min(confidence, 1.0), 0.1)  # Clamp to [0.1, 1.0] for safety
                cx, cy = x + bw / 2, y + bh / 2
                new_bw = max(int(bw * scaling), 1)
                new_bh = max(int(bh * scaling), 1)
                new_x = max(0, int(cx - new_bw / 2))
                new_y = max(0, int(cy - new_bh / 2))
                new_bw = min(w - new_x, new_bw)
                new_bh = min(h - new_y, new_bh)
                x, y, bw, bh = new_x, new_y, new_bw, new_bh
                def find_all_divisions(total, start, length, min_grid=2):
                    results = []
                    for div in range(256, min_grid - 1, -1):
                        step = total // div
                        if (start % step + length) <= step:
                            results.append(div)
                    return results
                row_divs = find_all_divisions(h, y, bh)
                col_divs = find_all_divisions(w, x, bw)
                selected_rows, selected_cols = None, None

                if row_divs and col_divs:
                    grid_pairs = [(r, c) for r in row_divs for c in col_divs]
                    
                    if Auto_grid_strategy == "max_rows_cols":
                        selected_rows = max(row_divs)
                        selected_cols = max(col_divs)
                    elif Auto_grid_strategy == "min_rows_cols":
                        selected_rows = min(row_divs)
                        selected_cols = min(col_divs)
                    elif Auto_grid_strategy == "max_rows_min_cols":
                        selected_rows = max(row_divs)
                        selected_cols = min(col_divs)
                    elif Auto_grid_strategy == "min_rows_max_cols":
                        selected_rows = min(row_divs)
                        selected_cols = max(col_divs)
                    elif Auto_grid_strategy in ["min_balanced", "max_balanced"]:
                        # Pick grid pair with smallest |rows - cols|
                        sorted_pairs = sorted(
                            grid_pairs,
                            key=lambda pair: (abs(pair[0] - pair[1]), -pair[0]*pair[1] if Auto_grid_strategy == "max_balanced" else pair[0]*pair[1])
                        )
                        selected_rows, selected_cols = sorted_pairs[0]

                    rows, cols = selected_rows, selected_cols
                else:
                    preview_bbox_color = "#FF0000"  # Fallback color for no match
        # --- End auto logic ---

        tile_h = h // rows
        tile_w = w // cols
        h_adj = tile_h * rows
        w_adj = tile_w * cols
        overlap_h = int(tile_h * overlap) + overlap_y
        overlap_w = int(tile_w * overlap) + overlap_x

        # max overlap is half of the tile size
        overlap_h = min(tile_h // 2, overlap_h)
        overlap_w = min(tile_w // 2, overlap_w)

        if rows == 1:
            overlap_h = 0
        if cols == 1:
            overlap_w = 0

        tiles = []
        for i in range(rows):
            for j in range(cols):
                y1 = i * tile_h
                x1 = j * tile_w

                if i > 0:
                    y1 -= overlap_h
                if j > 0:
                    x1 -= overlap_w

                y2 = y1 + tile_h + overlap_h
                x2 = x1 + tile_w + overlap_w

                if y2 > h:
                    y2 = h
                    y1 = y2 - tile_h - overlap_h
                if x2 > w:
                    x2 = w
                    x1 = x2 - tile_w - overlap_w

                tiles.append(image[:, y1:y2, x1:x2, :])
        tiles = torch.cat(tiles, dim=0)

        # 1. Create a white overlay image (uint8)
        overlay_image = torch.full((1, h, w, 3), 255, dtype=torch.uint8, device=image.device)

        # 2. Draw the yellow grid on the overlay (still uint8)
        overlay_image = self.draw_grid_overlay(overlay_image, rows, cols, tile_h, tile_w, overlap_h, overlap_w, preview_bbox_color)

        # 3. Convert overlay to float [0,1]
        overlay_image_float = overlay_image.float() / 255.0

        # 4. Make sure preview is a float tensor [0,1]
        preview = image.clone() if len(image.shape) == 4 else image.clone().unsqueeze(0)
        preview_float = preview.float() / 255.0 if preview.max() > 1.0 else preview.float()

        # 5. Blend preview and overlay (0.6 = 60% overlay, 30% original)
        blend_preview = self.blend_images(preview_float, overlay_image_float, 0.6)

        # 6. Convert to numpy for any further OpenCV drawing (optional, if you want to add more shapes)
        blend_preview_np = (blend_preview[0].cpu().numpy() * 255).astype(np.uint8)
        blend_preview_np = np.ascontiguousarray(blend_preview_np)

        # 7. (Optional) Draw more overlays (e.g., rectangles) with OpenCV here if needed
        # Convertir la couleur HEX en RGB
        preview_color = torch.tensor(self.Hex_to_RGB(preview_bbox_color), dtype=torch.uint8, device=image.device)
        # Draw rectangle for the BBOX
        if auto_from_mask_bbox:
            if mask_bbox is not None:
                original_x_start, original_y_start, original_bw, original_bh = mask_bbox
                # Ensure coordinates are integers
                x_int = int(x)
                y_int = int(y)
                bw_int = int(bw)
                bh_int = int(bh)
                pt1 = (x_int, y_int)
                pt2 = (x_int + bw_int, y_int + bh_int)

                # Print the bbox info
                print(f"Drawing bbox: top-left={pt1}, bottom-right={pt2}, color=({int(preview_color[0])}, {int(preview_color[1])}, {int(preview_color[2])}), thickness=4")

                # Draw the rectangle
                cv2.rectangle(blend_preview_np, pt1, pt2, (int(preview_color[0]), int(preview_color[1]), int(preview_color[2])), 4)
                orig_x = int(original_x_start)
                orig_y = int(original_y_start)
                orig_bw = int(original_bw)
                orig_bh = int(original_bh)
                orig_pt1 = (orig_x, orig_y)
                orig_pt2 = (orig_x + orig_bw, orig_y + orig_bh)

                print(f"Drawing original bbox: top-left={orig_pt1}, bottom-right={orig_pt2}, color={self.Hex_to_RGB('#710193')}, thickness=2")

                cv2.rectangle(blend_preview_np, orig_pt1, orig_pt2, self.Hex_to_RGB("#710193"), 2)

        # 8. Convert back to tensor for output
        preview = torch.from_numpy(blend_preview_np).unsqueeze(0).float() / 255.0

        # 9. Return as ComfyUI expects (tensor, float, [0,1])
        return (tiles, preview, tile_w+overlap_w, tile_h+overlap_h, overlap_w, overlap_h, rows, cols)

    def draw_grid_overlay(self, image, rows, cols, tile_h, tile_w, overlap_h, overlap_w, color_hex):
        preview = image.clone()
        color = torch.tensor(self.Hex_to_RGB(color_hex), dtype=preview.dtype, device=preview.device)
        black = torch.tensor([0, 0, 0], dtype=preview.dtype, device=preview.device)
        B, H, W, C = preview.shape

        # --- First pass: Draw all yellow grid lines ---
        for c in range(1, cols):
            x = c * tile_w - (overlap_w // 2 if c > 0 else 0)
            x_start = max(0, x - overlap_w // 2)
            x_end = min(W, x + (overlap_w + 1) // 2)
            preview[:, :, x_start:x_end, :3] = color

        for r in range(1, rows):
            y = r * tile_h - (overlap_h // 2 if r > 0 else 0)
            y_start = max(0, y - overlap_h // 2)
            y_end = min(H, y + (overlap_h + 1) // 2)
            preview[:, y_start:y_end, :, :3] = color

        # --- Second pass: Draw all black lines in the center of each grid ---
        for c in range(1, cols):
            x = c * tile_w - (overlap_w // 2 if c > 0 else 0)
            x_start = max(0, x - overlap_w // 2)
            x_end = min(W, x + (overlap_w + 1) // 2)
            center_x = (x_start + x_end) // 2
            bx_start = max(0, center_x - 1)  # 2-pixel black line
            bx_end = min(W, center_x + 1)
            preview[:, :, bx_start:bx_end, :3] = black

        for r in range(1, rows):
            y = r * tile_h - (overlap_h // 2 if r > 0 else 0)
            y_start = max(0, y - overlap_h // 2)
            y_end = min(H, y + (overlap_h + 1) // 2)
            center_y = (y_start + y_end) // 2
            by_start = max(0, center_y - 1)  # 2-pixel black line
            by_end = min(H, center_y + 1)
            preview[:, by_start:by_end, :, :3] = black

        return preview

    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float):
        if image1.shape != image2.shape:
            image2 = self.crop_and_resize(image2, image1.shape)

        blended_image = image1 * image2
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        return blended_image

    def Hex_to_RGB(self, inhex: str) -> tuple:
        if not inhex.startswith('#'):
            raise ValueError(f'Invalid Hex Code in {inhex}')
        rval = inhex[1:3]
        gval = inhex[3:5]
        bval = inhex[5:]
        return (int(rval, 16), int(gval, 16), int(bval, 16))

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_AutoImageTile_from_Mask_v1": FRED_AutoImageTile_from_Mask_v1
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_AutoImageTile_from_Mask_v1": "👑 FRED_AutoImageTile_from_Mask_v1"
}
