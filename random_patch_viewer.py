import argparse
import random
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    Dataset = object


class RandomPatchViewer:
    def __init__(self, root, video_path, total_frames, patch_size):
        self.root = root
        self.video_path = video_path
        self.total_frames = total_frames
        self.patch_size = patch_size
        self.history = []
        self.history_idx = -1
        self.current_entry = None
        self.photo_image = None
        self.max_display_width = 1100
        self.max_display_height = 750
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.training_thread = None

        self.root.title("Frame Annotation Viewer")
        self._build_ui()
        self._append_random_frame()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=20)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=1)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.minsize(1000, 950)
        self.root.geometry("1200x950")
        self.root.bind("<d>", self._on_key_next)
        self.root.bind("<D>", self._on_key_next)
        self.root.bind("<a>", self._on_key_previous)
        self.root.bind("<A>", self._on_key_previous)

        content = ttk.Frame(container)
        content.grid(row=0, column=0, sticky="n")
        content.grid_columnconfigure(0, weight=1)

        self.info_label = ttk.Label(content, justify="center", anchor="center")
        self.info_label.grid(row=0, column=0, sticky="ew")

        sample_label_frame = ttk.Frame(content)
        sample_label_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        sample_label_frame.grid_columnconfigure(0, weight=1)
        self.state_label = ttk.Label(
            sample_label_frame,
            justify="center",
            anchor="center",
            font=("Segoe UI", 12, "bold"),
        )
        self.state_label.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.coords_label = ttk.Label(
            sample_label_frame,
            justify="center",
            anchor="center",
            font=("Segoe UI", 12, "bold"),
        )
        self.coords_label.grid(row=1, column=0, sticky="ew")

        self.canvas = tk.Canvas(
            content,
            width=self.max_display_width,
            height=self.max_display_height,
            highlightthickness=0,
            borderwidth=0,
            bg="black",
        )
        self.canvas.grid(row=2, column=0, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)

        button_row = ttk.Frame(content)
        button_row.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        self.random_button = ttk.Button(
            button_row,
            text="Next random frame",
            command=self.load_next_frame,
        )
        self.random_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.train_button = ttk.Button(
            button_row,
            text="Train model",
            command=self.start_training,
        )
        self.train_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.stats_label = ttk.Label(content, justify="center", anchor="center")
        self.stats_label.grid(row=4, column=0, pady=(10, 0), sticky="ew")

    def _append_random_frame(self):
        frame_idx, frame, total_frames = choose_random_frame(self.video_path)
        self.total_frames = total_frames
        entry = {"frame_idx": frame_idx, "frame": frame, "annotations": []}
        if self.history_idx < len(self.history) - 1:
            self.history = self.history[: self.history_idx + 1]
        self.history.append(entry)
        self.history_idx = len(self.history) - 1
        self._set_current_entry(entry)

    def _set_current_entry(self, entry):
        self.current_entry = entry
        self._update_info_label()
        self._display_current_frame()
        self._update_stats_label()

    def _on_key_next(self, event):
        self.load_next_frame()

    def _on_key_previous(self, event):
        self.load_previous_frame()

    def load_next_frame(self):
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            self._set_current_entry(self.history[self.history_idx])
        else:
            self._append_random_frame()

    def load_previous_frame(self):
        if self.history_idx <= 0:
            return
        self.history_idx -= 1
        self._set_current_entry(self.history[self.history_idx])

    def _update_info_label(self):
        if not self.current_entry:
            self.info_label.config(text="–")
            return
        entry = self.current_entry
        info_text = (
            f"Video: {self.video_path}\n"
            f"Frame: {entry['frame_idx'] + 1} / {self.total_frames}\n"
            f"Annotations on this frame: {len(entry['annotations'])}"
        )
        self.info_label.config(text=info_text)
        self._update_annotation_label()

    def _update_annotation_label(self):
        if not self.current_entry or not self.current_entry["annotations"]:
            self.state_label.config(text="Annotations on frame: 0")
            self.coords_label.config(text="Last point: –")
            return
        count = len(self.current_entry["annotations"])
        last = self.current_entry["annotations"][-1]
        self.state_label.config(text=f"Annotations on frame: {count}")
        self.coords_label.config(
            text=f"Last point: ({last['x']:.1f}, {last['y']:.1f})"
        )

    def _update_stats_label(self):
        total_frames = len(self.history)
        annotated_frames = sum(1 for e in self.history if e["annotations"])
        total_points = sum(len(e["annotations"]) for e in self.history)
        self.stats_label.config(
            text=f"Frames visited: {total_frames} | Frames with crosses: {annotated_frames} | Total crosses: {total_points}"
        )

    def _display_current_frame(self):
        if not self.current_entry:
            return
        frame = self.current_entry["frame"]
        h, w = frame.shape[:2]
        scale = min(
            self.max_display_width / max(1, w),
            self.max_display_height / max(1, h),
            1.0,
        )
        disp_w = int(w * scale)
        disp_h = int(h * scale)
        self.scale_x = scale
        self.scale_y = scale
        resized = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        for ann in self.current_entry["annotations"]:
            dx = ann["x"] * scale
            dy = ann["y"] * scale
            half = 10
            draw.line((dx - half, dy, dx + half, dy), fill="red", width=2)
            draw.line((dx, dy - half, dx, dy + half), fill="red", width=2)
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.configure(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)

    def on_canvas_click(self, event):
        if not self.current_entry:
            return
        frame_x = event.x / max(1e-6, self.scale_x)
        frame_y = event.y / max(1e-6, self.scale_y)
        frame = self.current_entry["frame"]
        h, w = frame.shape[:2]
        frame_x = float(np.clip(frame_x, 0.0, w - 1e-6))
        frame_y = float(np.clip(frame_y, 0.0, h - 1e-6))
        self.current_entry["annotations"].append({"x": frame_x, "y": frame_y})
        self._update_annotation_label()
        self._update_stats_label()
        self._display_current_frame()

    def on_canvas_right_click(self, event):
        if not self.current_entry or not self.current_entry["annotations"]:
            return
        frame_x = event.x / max(1e-6, self.scale_x)
        frame_y = event.y / max(1e-6, self.scale_y)
        annotations = self.current_entry["annotations"]
        distances = [
            ((a["x"] - frame_x) ** 2 + (a["y"] - frame_y) ** 2, idx)
            for idx, a in enumerate(annotations)
        ]
        distances.sort()
        _, idx = distances[0]
        annotations.pop(idx)
        self._update_annotation_label()
        self._update_stats_label()
        self._display_current_frame()

    def start_training(self):
        if torch is None:
            print("PyTorch is not available. Install torch to enable training.")
            return
        if self.training_thread and self.training_thread.is_alive():
            print("Training already in progress.")
            return
        prepared = self._prepare_training_data()
        if not prepared:
            return
        self.training_thread = threading.Thread(
            target=self._run_training_worker, args=(prepared,), daemon=True
        )
        self.training_thread.start()

    def _prepare_training_data(self):
        positives = []
        negatives = []
        for entry in self.history:
            frame = entry["frame"]
            annotations = entry["annotations"]
            for ann in annotations:
                patch, coord = self._extract_patch(frame, ann["x"], ann["y"], jitter=True)
                positives.append({"image": patch, "label": 1, "coord": coord})
            negative_count = max(2, len(annotations) + 1)
            negatives.extend(
                self._generate_negative_patches(frame, annotations, negative_count)
            )

        if not positives:
            print("[Train] Need at least one annotated cross before training.")
            return None
        samples = positives + negatives
        random.shuffle(samples)
        total = len(samples)
        test_size = max(1, int(total * 0.2))
        train_samples = samples[test_size:]
        test_samples = samples[:test_size]
        if len(train_samples) < 5 or len(test_samples) < 2:
            print("[Train] Not enough samples after split.")
            return None
        pos_count = sum(1 for s in train_samples if s["label"] == 1)
        neg_count = len(train_samples) - pos_count
        pos_weight = max(1.0, neg_count / max(1, pos_count))
        return {
            "train": train_samples,
            "test": test_samples,
            "pos_weight": pos_weight,
        }

    def _generate_negative_patches(self, frame, annotations, target_count):
        negatives = []
        h, w = frame.shape[:2]
        attempts = 0
        max_attempts = target_count * 20
        min_dist = self.patch_size * 0.75
        while len(negatives) < target_count and attempts < max_attempts:
            attempts += 1
            cx = random.uniform(0, w - 1)
            cy = random.uniform(0, h - 1)
            if annotations:
                too_close = any(
                    (abs(ann["x"] - cx) < min_dist)
                    and (abs(ann["y"] - cy) < min_dist)
                    for ann in annotations
                )
                if too_close:
                    continue
            patch, _ = self._extract_patch(frame, cx, cy, jitter=False)
            negatives.append({"image": patch, "label": 0, "coord": (0.0, 0.0)})
        return negatives

    def _extract_patch(self, frame, cross_x, cross_y, jitter=True):
        h, w = frame.shape[:2]
        jitter_range = self.patch_size * 0.15 if jitter else 0.0
        offset_x = random.uniform(-jitter_range, jitter_range)
        offset_y = random.uniform(-jitter_range, jitter_range)
        center_x = np.clip(cross_x + offset_x, 0.0, w - 1e-6)
        center_y = np.clip(cross_y + offset_y, 0.0, h - 1e-6)
        pad = self.patch_size
        padded = cv2.copyMakeBorder(
            frame, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101
        )
        cx = center_x + pad
        cy = center_y + pad
        half = self.patch_size / 2
        x0 = int(round(cx - half))
        y0 = int(round(cy - half))
        patch = padded[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        rel_x = (cross_x + pad) - x0
        rel_y = (cross_y + pad) - y0
        rel_x = float(np.clip(rel_x, 0.0, self.patch_size - 1e-6))
        rel_y = float(np.clip(rel_y, 0.0, self.patch_size - 1e-6))
        return patch, (rel_x, rel_y)

    def _run_training_worker(self, prepared):
        train_samples = prepared["train"]
        test_samples = prepared["test"]
        pos_weight = prepared["pos_weight"]

        train_dataset = PatchDataset(train_samples, self.patch_size, augment=True)
        test_dataset = PatchDataset(test_samples, self.patch_size, augment=False)
        batch_size = min(64, max(4, len(train_dataset)))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CrossNet().to(device)
        cls_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        reg_criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = min(60, max(10, len(train_dataset) // 4))

        print(
            f"[Train] Starting with {len(train_dataset)} training samples and {len(test_dataset)} test samples (pos_weight={pos_weight:.2f})"
        )
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_items = 0
            for images, labels, coords in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                coords = coords.to(device)

                optimizer.zero_grad()
                logits, coord_raw = model(images)
                cls_loss = cls_criterion(logits, labels)
                mask = (labels > 0.5).squeeze(1)
                if mask.any():
                    coord_pred = torch.sigmoid(coord_raw[mask])
                    reg_loss = reg_criterion(coord_pred, coords[mask])
                else:
                    reg_loss = torch.tensor(0.0, device=device)
                loss = cls_loss + 0.5 * reg_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                total_items += images.size(0)

            avg_loss = total_loss / max(1, total_items)
            if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
                metrics = self._evaluate_model(model, test_loader, device)
                print(
                    f"[Train] Epoch {epoch}/{epochs} loss={avg_loss:.4f} acc={metrics['acc']:.3f} "
                    f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} mae={metrics['mae']:.2f}px"
                )

        metrics = self._evaluate_model(model, test_loader, device)
        print(
            f"[Train] Finished training. Test acc={metrics['acc']:.3f} precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} mae={metrics['mae']:.2f}px"
        )

    def _evaluate_model(self, model, loader, device):
        model.eval()
        total = correct = 0
        tp = fp = fn = 0
        mae = 0.0
        mae_batches = 0
        with torch.no_grad():
            for images, labels, coords in loader:
                images = images.to(device)
                labels = labels.to(device)
                coords = coords.to(device)
                logits, coord_raw = model(images)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                label_mask = labels > 0.5
                pred_mask = preds > 0.5
                tp += torch.logical_and(pred_mask, label_mask).sum().item()
                fp += torch.logical_and(pred_mask, ~label_mask).sum().item()
                fn += torch.logical_and(~pred_mask, label_mask).sum().item()

                positive_mask = label_mask.squeeze(1)
                if positive_mask.any():
                    coord_pred = torch.sigmoid(coord_raw[positive_mask])
                    diff = torch.abs((coord_pred - coords[positive_mask]) * self.patch_size)
                    mae += diff.sum(dim=1).mean().item()
                    mae_batches += 1

        accuracy = correct / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        mae_value = mae / mae_batches if mae_batches else 0.0
        return {
            "acc": accuracy,
            "precision": precision,
            "recall": recall,
            "mae": mae_value,
        }


if torch is not None:

    class PatchDataset(Dataset):
        def __init__(self, samples, patch_size, augment=False):
            self.samples = samples
            self.patch_size = patch_size
            self.augment = augment

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            entry = self.samples[idx]
            image = entry["image"]
            if self.augment:
                image = self._apply_augment(image)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb = image_rgb.astype(np.float32) / 255.0
            tensor = torch.from_numpy(np.transpose(image_rgb, (2, 0, 1)))
            label = torch.tensor([entry["label"]], dtype=torch.float32)
            coord = entry["coord"]
            coord_tensor = torch.tensor(
                [coord[0] / self.patch_size, coord[1] / self.patch_size],
                dtype=torch.float32,
            )
            return tensor, label, coord_tensor

        def _apply_augment(self, image):
            augmented = image.astype(np.float32)
            if random.random() < 0.5:
                factor = 0.8 + 0.4 * random.random()
                augmented *= factor
            if random.random() < 0.3:
                noise = np.random.normal(0, 5, image.shape)
                augmented += noise
            return np.clip(augmented, 0, 255).astype(np.uint8)


    def _conv_bn(in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


    def _depthwise_sep(in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


    class CrossNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                _conv_bn(3, 16, stride=2),
                _depthwise_sep(16, 24, stride=1),
                _depthwise_sep(24, 32, stride=2),
                _depthwise_sep(32, 48, stride=1),
                _depthwise_sep(48, 64, stride=2),
                _depthwise_sep(64, 64, stride=1),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.class_head = nn.Linear(64, 1)
            self.reg_head = nn.Linear(64, 2)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).flatten(1)
            logits = self.class_head(x)
            coords = self.reg_head(x)
            return logits, coords


else:
    PatchDataset = None
    CrossNet = None


def choose_random_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video '{video_path}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video '{video_path}' does not contain any frames")

    frame_idx = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(f"Failed to read frame {frame_idx} from '{video_path}'")

    return frame_idx, frame, total_frames


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video '{video_path}'")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return total


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate frames and train a simple cross detector."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="video.mp4",
        help="Path to the video file to sample from.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=100,
        help="Square patch size in pixels (default: 100).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    total_frames = get_total_frames(args.video)
    viewer = RandomPatchViewer(root, args.video, total_frames, args.patch_size)
    root.mainloop()


if __name__ == "__main__":
    main()

