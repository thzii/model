"""Backbone adapter, projection head and classification head (spec §3.2 C3-C5, C9)."""
import os

import pytest
import torch
from torch import nn

from planora_sf.config import BackboneConfig
from planora_sf.models import DINOHead, LinearClsHead, TinyViT, build_backbone, build_model, strip_dino_checkpoint
from planora_sf.models.backbone import BackboneAdapter, infer_feature_dim
from helpers import make_tiny_config


class _NoDimAttr(nn.Module):
    """A ViT-like module without embed_dim, to exercise dimension probing."""

    def __init__(self):
        super().__init__()
        self.inner = TinyViT(img_size=32, embed_dim=24, depth=2)
        self.blocks = self.inner.blocks

    def forward(self, x):
        return self.inner(x)


def test_feature_dim_is_inferred_not_hardcoded():
    assert infer_feature_dim(TinyViT(embed_dim=40)) == 40
    assert BackboneAdapter(_NoDimAttr(), probe_image_size=32).feat_dim == 24
    with pytest.raises(ValueError):
        infer_feature_dim(_NoDimAttr())


def test_adapter_forward_shape_and_validation():
    adapter = BackboneAdapter(TinyViT(embed_dim=32))
    out = adapter(torch.randn(3, 3, 32, 32))
    assert out.shape == (3, 32) and torch.isfinite(out).all()
    with pytest.raises(ValueError):
        BackboneAdapter(nn.Linear(4, 4))
    wrong = BackboneAdapter(TinyViT(embed_dim=32), feat_dim=64)
    with pytest.raises(ValueError):
        wrong(torch.randn(1, 3, 32, 32))


def test_freeze_trains_only_blocks_from_grad_from_block():
    adapter = build_backbone(BackboneConfig(name="tiny", grad_from_block=2, image_size=32))
    assert adapter.num_blocks == 4 and adapter.trainable_block_indices() == [2, 3]
    vit = adapter.vit
    frozen = [vit.cls_token, vit.pos_embed, *vit.patch_embed.parameters(), *vit.norm.parameters(),
              *vit.blocks[0].parameters(), *vit.blocks[1].parameters()]
    assert not any(p.requires_grad for p in frozen)
    assert all(p.requires_grad for p in vit.blocks[3].parameters())
    adapter.freeze(4, train_final_norm=True)
    assert adapter.trainable_block_indices() == [] and all(p.requires_grad for p in vit.norm.parameters())
    with pytest.raises(ValueError):
        adapter.freeze(5)


def test_dinohead_reference_dimensions_and_trainable_parameters():
    head = DINOHead(768)  # SelEx defaults: 768 -> 2048 -> 2048 -> 256 -> 65536
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert trainable == 23_073_024  # spec U-17 arithmetic: MLP + weight-norm direction, gain frozen
    gain = head.last_layer.parametrizations.weight.original0
    assert not gain.requires_grad and torch.all(gain == 1)
    out = head(torch.randn(2, 768))
    assert out.shape == (2, 65536) and torch.isfinite(out).all()


def test_linear_cls_head_and_model_build(tiny_cfg):
    assert LinearClsHead(32, 5)(torch.randn(4, 32)).shape == (4, 5)
    model = build_model(tiny_cfg, num_known_classes=3)
    out = model(torch.randn(4, 3, 32, 32))
    assert out.cls.shape == (4, 32) and out.z.shape == (4, 128) and out.cls_logits.shape == (4, 3)
    torch.testing.assert_close(out.z.norm(dim=1), torch.ones(4))
    no_cls = make_tiny_config()
    no_cls.cls_head.enabled = False
    assert build_model(no_cls, num_known_classes=3)(torch.randn(1, 3, 32, 32)).cls_logits is None


def test_strip_dino_checkpoint_matches_selex_key_handling():
    weight = torch.ones(1)
    state = {"teacher": {"backbone.blocks.0.attn.qkv.weight": weight, "backbone.norm.bias": weight,
                         "head.mlp.0.weight": weight}}
    assert set(strip_dino_checkpoint(state)) == {"blocks.0.attn.qkv.weight", "norm.bias"}
    assert set(strip_dino_checkpoint({"cls_token": weight})) == {"cls_token"}


@pytest.mark.skipif(os.environ.get("PLANORA_SF_ALLOW_DOWNLOAD") != "1",
                    reason="downloads DINOv1 weights via torch.hub; set PLANORA_SF_ALLOW_DOWNLOAD=1 to run")
def test_real_dinov1_backbone_loads():
    adapter = build_backbone(BackboneConfig(name="dinov1_vitb16"))
    assert adapter.feat_dim == 768 and adapter.num_blocks == 12
    assert adapter.trainable_block_indices() == [9, 10, 11]
