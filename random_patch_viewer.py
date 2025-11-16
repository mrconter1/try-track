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
    def __init__(self, root, video_path, total_frames, frame_idx, coords, patch, patch_size):
        self.root = root
        self.video_path = video_path
        self.patch_size = patch_size
        self.display_size = 400
        self.photo_image = None
        self.total_frames = total_frames
        self.frame_idx = frame_idx
        self.coords = coords
        self.patch = patch
        self.history = []
        self.history_idx = -1
        self.current_sample = None
        self.training_thread = None

        self.root.title("Random Patch Viewer")
        self._build_ui()
        self._store_sample(frame_idx, total_frames, coords, patch)

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=20)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=1)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.minsize(720, 780)
        self.root.geometry("760x820")
        self.root.bind("<d>", self._on_key_randomize)
        self.root.bind("<D>", self._on_key_randomize)
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
            width=self.display_size,
            height=self.display_size,
            highlightthickness=0,
            borderwidth=0,
            bg="black",
        )
        self.canvas.grid(row=2, column=0, pady=10)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)

        button_row = ttk.Frame(content)
        button_row.grid(row=3, column=0, sticky="ew")
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        self.random_button = ttk.Button(
            button_row,
            text="Randomize patch",
            command=self.load_next_patch,
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
        self._update_info_label()

    def _display_patch(self):
        rgb_patch = cv2.cvtColor(self.patch, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_patch).resize(
            (self.display_size, self.display_size),
            resample=Image.NEAREST,
        )
        if self.current_sample and self.current_sample.get("annotation"):
            ann_x, ann_y = self.current_sample["annotation"]
            patch_h, patch_w = self.patch.shape[:2]
            scale_x = self.display_size / max(1, patch_w)
            scale_y = self.display_size / max(1, patch_h)
            draw = ImageDraw.Draw(image)
            disp_x = ann_x * scale_x
            disp_y = ann_y * scale_y
            half = 8
            draw.line(
                (disp_x - half, disp_y, disp_x + half, disp_y),
                fill="red",
                width=2,
            )
            draw.line(
                (disp_x, disp_y - half, disp_x, disp_y + half),
                fill="red",
                width=2,
            )
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)

    def _update_info_label(self):
        info_text = (
            f"Video: {self.video_path}\n"
            f"Frame: {self.frame_idx + 1} / {self.total_frames}\n"
            f"Top-left pixel: ({self.coords[0]}, {self.coords[1]})"
        )
        self.info_label.config(text=info_text)
        self._update_sample_label()

    def _update_sample_label(self):
        state_color = "#cc0000"
        coords_color = "#cc0000"
        if not self.current_sample:
            state_text = "State: –"
            coord_text = "Patch coords: –"
        else:
            state = self.current_sample.get("state", "No cross")
            annotation = self.current_sample.get("annotation")
            if annotation:
                coord_text = f"Patch coords: ({annotation[0]:.1f}, {annotation[1]:.1f})"
                coords_color = "#003399"
            else:
                coord_text = "Patch coords: –"
            if state == "Has cross" and annotation:
                state_color = "#003399"
                coords_color = "#003399"
            else:
                state_color = "#cc0000"
                coords_color = "#cc0000" if annotation is None else coords_color
            state_text = f"State: {state}"
        self.state_label.config(text=state_text, foreground=state_color)
        self.coords_label.config(text=coord_text, foreground=coords_color)
        self._update_stats_label()

    def _update_stats_label(self):
        no_cross = sum(
            1
            for sample in self.history
            if sample.get("state") == "No cross" or not sample.get("annotation")
        )
        has_cross = sum(
            1
            for sample in self.history
            if sample.get("state") == "Has cross" and sample.get("annotation")
        )
        self.stats_label.config(
            text=f"No cross: {no_cross}    With cross: {has_cross}"
        )

    def start_training(self):
        if torch is None or nn is None or optim is None or DataLoader is None:
            print("PyTorch is not available. Install torch to enable training.")
            return
        if self.training_thread and self.training_thread.is_alive():
            print("Training already in progress.")
            return
        if not self.history:
            print("No samples available for training.")
            return
        self.training_thread = threading.Thread(
            target=self._run_training_worker, daemon=True
        )
        self.training_thread.start()

    def _run_training_worker(self):
        prepared = self._prepare_training_data()
        if not prepared:
            return
        train_samples = prepared["train"]
        test_samples = prepared["test"]
        pos_weight = prepared["pos_weight"]

        train_dataset = PatchDataset(train_samples, self.patch_size, augment=True)
        test_dataset = PatchDataset(test_samples, self.patch_size, augment=False)
        batch_size = min(32, max(4, len(train_dataset)))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CrossNet().to(device)
        cls_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        reg_criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = min(40, max(8, len(train_dataset)))

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

    def _prepare_training_data(self):
        snapshot = []
        for sample in self.history:
            patch = sample.get("patch")
            if patch is None:
                continue
            label = 1 if sample.get("annotation") else 0
            coord = sample.get("annotation") or (0.0, 0.0)
            snapshot.append(
                {
                    "image": patch.copy(),
                    "label": label,
                    "coord": (float(coord[0]), float(coord[1])),
                }
            )

        total = len(snapshot)
        if total < 10:
            print(f"[Train] Need at least 10 samples to train (have {total}).")
            return None
        positives = sum(1 for s in snapshot if s["label"] == 1)
        negatives = total - positives
        if positives == 0 or negatives == 0:
            print("[Train] Need both cross and no-cross samples.")
            return None

        random.shuffle(snapshot)
        test_size = max(1, int(total * 0.2))
        train_samples = snapshot[test_size:]
        test_samples = snapshot[:test_size]
        if len(train_samples) < 5 or len(test_samples) < 2:
            print("[Train] Not enough data after train/test split.")
            return None

        pos_weight = max(1.0, negatives / max(1, positives))
        return {"train": train_samples, "test": test_samples, "pos_weight": pos_weight}

    def _evaluate_model(self, model, loader, device):
        model.eval()
        total = 0
        correct = 0
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
        return {"acc": accuracy, "precision": precision, "recall": recall, "mae": mae_value}

    def load_next_patch(self):
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            sample = self.history[self.history_idx]
            self._apply_sample(sample)
        else:
            self._append_random_sample()

    def _append_random_sample(self):
        frame_idx, frame, total_frames = choose_random_frame(self.video_path)
        coords, patch = choose_random_patch(frame, self.patch_size)
        self._store_sample(frame_idx, total_frames, coords, patch)

    def _on_key_randomize(self, event):
        self.load_next_patch()

    def load_previous_patch(self):
        if self.history_idx <= 0:
            return
        self.history_idx -= 1
        sample = self.history[self.history_idx]
        self._apply_sample(sample)

    def _store_sample(self, frame_idx, total_frames, coords, patch):
        if self.history_idx < len(self.history) - 1:
            self.history = self.history[: self.history_idx + 1]
        sample = {
            "frame_idx": frame_idx,
            "total_frames": total_frames,
            "coords": coords,
            "patch": patch,
            "state": "No cross",
            "annotation": None,
        }
        self.history.append(sample)
        self.history_idx += 1
        self._apply_sample(sample)

    def _apply_sample(self, sample):
        self.current_sample = sample
        self.frame_idx = sample["frame_idx"]
        self.total_frames = sample["total_frames"]
        self.coords = sample["coords"]
        self.patch = sample["patch"]
        self._update_info_label()
        self._display_patch()

    def _on_key_previous(self, event):
        self.load_previous_patch()

    def on_canvas_press(self, event):
        self.canvas.configure(cursor="none")
        self._update_annotation_from_event(event)

    def on_canvas_drag(self, event):
        self._update_annotation_from_event(event)

    def on_canvas_release(self, event):
        self.canvas.configure(cursor="")

    def on_canvas_right_click(self, event):
        self.clear_annotation()

    def _update_annotation_from_event(self, event):
        if not self.current_sample:
            return
        patch_h, patch_w = self.patch.shape[:2]
        if patch_h == 0 or patch_w == 0:
            return
        scale_x = patch_w / self.display_size
        scale_y = patch_h / self.display_size
        patch_x = float(np.clip(event.x * scale_x, 0.0, patch_w - 1e-6))
        patch_y = float(np.clip(event.y * scale_y, 0.0, patch_h - 1e-6))
        self.set_annotation(patch_x, patch_y)

    def set_annotation(self, patch_x, patch_y):
        if not self.current_sample:
            return
        self.current_sample["annotation"] = (float(patch_x), float(patch_y))
        self.current_sample["state"] = "Has cross"
        self._update_sample_label()
        self._display_patch()

    def clear_annotation(self):
        if not self.current_sample:
            return
        if self.current_sample.get("annotation") is None:
            return
        self.current_sample["annotation"] = None
        self.current_sample["state"] = "No cross"
        self._update_sample_label()
        self._display_patch()


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


def choose_random_patch(frame, patch_size):
    if patch_size <= 0:
        raise ValueError("Patch size must be positive")

    height, width = frame.shape[:2]
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Patch size {patch_size} exceeds frame dimensions {width}x{height}"
        )

    max_x = width - patch_size
    max_y = height - patch_size
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    patch = frame[y : y + patch_size, x : x + patch_size].copy()
    return (x, y), patch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display a random 100x100 region from a random frame in a video."
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
    frame_idx, frame, total_frames = choose_random_frame(args.video)
    coords, patch = choose_random_patch(frame, args.patch_size)

    root = tk.Tk()
    viewer = RandomPatchViewer(
        root,
        args.video,
        total_frames,
        frame_idx,
        coords,
        patch,
        args.patch_size,
    )
    root.mainloop()


if __name__ == "__main__":
    main()

