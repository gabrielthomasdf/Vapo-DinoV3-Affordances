"""Saved-artifact identity checks, without model dependencies or experiment runs."""

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("experiment_identity", SCRIPTS / "audit_experiment_identity.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def manifests(tmp_path, wrong_order=False, overlap=False, duplicate=False):
    files = {}
    root = {"training": {}, "validation": {}}
    samples = []
    for scene in ("a", "b", "c"):
        train = ["gripper_cam/frame_0"]
        val = ["gripper_cam/frame_2", "static_cam/frame_2", "gripper_cam/frame_1"]
        data = {"training": {"episode_0": train}, "validation": {"episode_1": val}}
        if overlap:
            data["training"]["episode_1"] = ["gripper_cam/frame_1"]
        if duplicate:
            data["validation"]["episode_1"].append("gripper_cam/frame_1")
        files[scene + "/episodes_split.json"] = data
        for split in ("training", "validation"):
            for episode, paths in data[split].items():
                root[split]["/" + scene + "/" + episode] = paths
        samples.extend(["/" + scene + "/episode_1/gripper_cam/frame_1", "/" + scene + "/episode_1/gripper_cam/frame_2"])
    files["episodes_split.json"] = root
    path = tmp_path / "splits.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, value in files.items():
            raw = json.dumps(value).encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    if wrong_order:
        samples.reverse()
    csv = tmp_path / "images.csv"
    csv.write_text("index,sample\n" + "".join(str(i) + "," + sample + "\n" for i, sample in enumerate(samples)))
    return path, csv


def test_all_scene_manifests_match_loader_order_and_exclude_other_cameras(tmp_path):
    archive, csv = manifests(tmp_path)
    result = audit.audit_splits(archive, csv)
    assert result["training_gripper_samples"] == 3
    assert result["validation_gripper_samples"] == 6
    assert result["matches_evaluated_ids_in_order"]
    assert result["training_validation_overlap"] == 0
    assert all(scene["validation_gripper_samples"] == 2 for scene in result["scenes"])
    assert "frame_1" not in json.dumps(result)


@pytest.mark.parametrize("option", ("wrong_order", "overlap", "duplicate"))
def test_split_discrepancies_are_rejected(tmp_path, option):
    archive, csv = manifests(tmp_path, **{option: True})
    with pytest.raises(audit.ValidationError):
        audit.audit_splits(archive, csv)


def test_checkpoint_roles_disambiguate_policy_names_and_remove_private_roots(tmp_path):
    path = tmp_path / "hashes.txt"
    path.write_text("".join(hashlib.sha256(role.encode()).hexdigest() + "  /private/user/" + suffix + "\n"
                            for role, suffix in audit.CHECKPOINTS.items()))
    result = audit.checkpoint_manifest(path)
    identities = result["artifacts"]
    assert identities["sac_vapo"]["sha256"] != identities["sac_no_affordance_ablation"]["sha256"]
    assert identities["sac_vapo"]["selected_setup_reference"]
    assert not identities["sac_no_affordance_ablation"]["selected_setup_reference"]
    assert "/private/user" not in json.dumps(result)


@pytest.mark.parametrize("text", ("not-a-hash  /private/file", "a" * 64 + "  /private/unknown.ckpt"))
def test_bad_checkpoint_manifest_is_rejected(tmp_path, text):
    path = tmp_path / "hashes.txt"
    path.write_text(text)
    with pytest.raises(audit.ValidationError):
        audit.checkpoint_manifest(path)


def test_recorded_warmup_overrides_default_without_executing_evaluator(tmp_path):
    evaluator = tmp_path / "eval.py"
    evaluator.write_text("raise RuntimeError('never execute')\nparser.add_argument('--warmup', default=20)\nparser.add_argument('--amp', action='store_true')")
    command = tmp_path / "command.txt"
    command.write_text("python eval_affordance_dinov3_vs_baseline.py --warmup 5 --amp\n")
    result = audit.audit_command(command, evaluator, {"comparison": {"amp": True}})
    assert result["warmup_iterations_recorded"] == 5
    assert result["warmup_default_in_source"] == 20
    assert result["arguments_supported_by_source"]
    assert not result["actual_runtime_warmup_independently_attested"]
    with pytest.raises(audit.ValidationError):
        audit.audit_command(command, evaluator, {"comparison": {"amp": False}})


def test_recovered_evaluator_defaults_can_be_read_without_torch():
    values = audit.static_defaults((SCRIPTS / "eval_affordance_dinov3_vs_baseline.py").read_text(encoding="utf-8"))
    assert values["--warmup"] == 20
    assert values["--dino-checkpoint"].endswith("trained_models/best_val_miou.ckpt")
