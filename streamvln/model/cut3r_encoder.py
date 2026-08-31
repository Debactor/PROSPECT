import os
import sys
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
cut3r_path = os.path.join(project_root, "CUT3R", "src")
if cut3r_path not in sys.path:
    sys.path.insert(0, cut3r_path)

from dust3r.model import ARCroco3DStereo

DEFAULT_CUT3R_HF_REPO = "blanchon/CUT3R"
DEFAULT_CUT3R_WEIGHTS = "cut3r_512_dpt_4_64.pth"


def resolve_cut3r_weights(weights_path=None):
    """Resolve CUT3R checkpoint path, downloading from HuggingFace if needed."""
    if weights_path and os.path.isfile(weights_path):
        return weights_path

    env_path = os.environ.get("CUT3R_WEIGHTS_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    local_dir = os.environ.get("CUT3R_WEIGHTS_DIR", os.path.join(project_root, "checkpoints", "cut3r"))
    local_path = os.path.join(local_dir, DEFAULT_CUT3R_WEIGHTS)
    if os.path.isfile(local_path):
        return local_path

    os.makedirs(local_dir, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        print(f"Downloading CUT3R weights from {DEFAULT_CUT3R_HF_REPO}...")
        return hf_hub_download(
            repo_id=DEFAULT_CUT3R_HF_REPO,
            filename=DEFAULT_CUT3R_WEIGHTS,
            local_dir=local_dir,
        )
    except Exception as exc:
        raise FileNotFoundError(
            "CUT3R weights not found. Set CUT3R_WEIGHTS_PATH, place weights at "
            f"{local_path}, or install huggingface_hub to auto-download from "
            f"https://huggingface.co/{DEFAULT_CUT3R_HF_REPO}"
        ) from exc


def prepare_cut3r_input(pixel_values):
    """Convert [B*V, C, H, W] images into CUT3R view dicts."""
    pixel_values = nn.functional.interpolate(
        pixel_values, size=(512, 512), mode="bilinear", align_corners=False
    )
    pixel_values = pixel_values.unsqueeze(1).permute(1, 0, 2, 3, 4)
    f_max, batch_size, _, height, width = pixel_values.shape
    device = pixel_values.device

    views = []
    for i in range(f_max):
        frame = pixel_values[i]
        views.append(
            {
                "img": frame,
                "ray_map": torch.full((batch_size, 6, height, width), torch.nan).to(device),
                "true_shape": torch.tensor([height, width], device=device).expand(batch_size, -1),
                "idx": i,
                "instance": [str(j) for j in range(batch_size)],
                "camera_pose": torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                "img_mask": torch.tensor(True, device=device).expand(batch_size),
                "ray_mask": torch.tensor(False, device=device).expand(batch_size),
                "update": torch.tensor(True, device=device).expand(batch_size),
                "reset": torch.tensor(False, device=device).expand(batch_size),
            }
        )
    return views


class SimpleCut3rEncoder(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        resolved = resolve_cut3r_weights(weights_path)
        print(f"Loading CUT3R weights from {resolved}")
        self.cut3r = ARCroco3DStereo.from_pretrained(resolved)
        self.cut3r.eval()
        for param in self.cut3r.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values):
        """Return patch features [B*V, N_patch, D]."""
        views = prepare_cut3r_input(pixel_values)
        shape, feat_ls, pos = self.cut3r._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self.cut3r._init_state(feat[0], pos[0])
        mem = self.cut3r.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()

        patch_features = []
        for i in range(len(views)):
            feat_i = feat[i].to(pixel_values.dtype)
            pos_i = pos[i]

            if self.cut3r.pose_head_flag:
                global_img_feat_i = self.cut3r._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.cut3r.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.cut3r.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None

            new_state_feat, dec = self.cut3r._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
            )

            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.cut3r.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
            patch_features.append(dec[-1][:, 1:].clone())

            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            update_mask = (img_mask & update) if update is not None else img_mask
            update_mask = update_mask[:, None, None].to(pixel_values.dtype)
            state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
            mem = new_mem * update_mask + mem * (1 - update_mask)

            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].to(pixel_values.dtype)
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                mem = init_mem * reset_mask + mem * (1 - reset_mask)

        patch_features = torch.stack(patch_features, dim=0)
        if patch_features.shape[0] == 1:
            patch_features = patch_features.squeeze(0)
        return patch_features
