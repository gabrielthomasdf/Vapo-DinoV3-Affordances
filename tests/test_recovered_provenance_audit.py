"""Check log/config provenance without executing archived experiment code."""

import csv
import importlib.util
import io
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("provenance_audit", SCRIPTS / "audit_recovered_provenance.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_parser_preserves_object_episode_success_and_negative_return():
    text = "\x1b[36mINFO - EVALUATION [0] pan_1, Return: -4.949, Success: False, Steps: 7\x1b[0m"
    assert audit.episode_records(text, "baseline", 8) == [("baseline", 8, 0, "pan_1", 0, -4.949, 7)]


def test_duplicate_status_seed_is_rejected():
    with pytest.raises(audit.ValidationError):
        audit.parse_status("seed\texit_code\n0\t0\n0\t0\n")


def test_argument_defaults_are_read_statically():
    source = "raise RuntimeError('never execute')\nparser.add_argument('--warmup', default=20)"
    assert audit.argument_defaults(source)["--warmup"] == 20


def pack(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, value in files.items():
            data = value.encode() if isinstance(value, str) else value
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def make_archive(tmp_path, bad_seed=False, bad_log=False):
    records, nested, files = [], {}, {}
    prefix = audit.PREFIX
    for model in audit.MODELS:
        files[f"{prefix}/runs/{model}/status.tsv"] = "seed\texit_code\n0\t0\n1\t0\n"
        config = {"test": {"folder_name": "/private/policy", "model_name": "policy"},
                  "env": {"seed": None, "use_egl": False}, "img_size": 64,
                  "static_cam_aff_path": "/private/static.ckpt",
                  "affordance": {"gripper_cam": {"model_path": "/private/gripper.ckpt", "hyperparameters": {
                      "cfg": {"encoder_type": model, "unet_cfg": {"decoder_channels": [1, 2]}}}}}}
        files[f"{prefix}/{model}/config_resolved.yaml"] = yaml.safe_dump(config)
        for seed in (0, 1):
            lines = []
            for idx, obj in enumerate(("a", "b")):
                records.append({"model": model, "seed": seed, "episode_index": idx, "object": obj,
                                "success": seed, "return": 1.25, "steps": 2})
                success = not seed if bad_log and model == "baseline" and seed == 0 else bool(seed)
                lines.append(f"EVALUATION [{idx}] {obj}, Return: 1.250, Success: {success}, Steps: 2")
            lines.append(f"Full evaluation over 2 objs\nSuccess: {2 * seed}/2")
            lines.append("Using VREnv with commit " + "b" * 40 + ".")
            files[f"{prefix}/runs/{model}/seed_{seed}.log"] = "\n".join(lines)
            cfg = json.loads(json.dumps(config))
            cfg["test"]["eval_seed"] = 9 if bad_seed else seed
            nested[f"./{model}_seed{seed}/.hydra/config.yaml"] = yaml.safe_dump(cfg)
            nested[f"./{model}_seed{seed}/.hydra/overrides.yaml"] = yaml.safe_dump([f"test.eval_seed={seed}"])
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    (tmp_path / "ab_results.csv").write_bytes(text.getvalue().encode())
    files[prefix + "/ab_results.csv"] = text.getvalue()
    for name in ("eval_affordance_dinov3_vs_baseline.py", "scripts/eval_tabletop.py", "run_baseline_100.sh",
                 "run_dinov3_100.sh", "parse_ab_results.py"):
        files[name] = "# This source is only read, never run.\n"
    for name in (f"{prefix}/git_commit.txt", f"{prefix}/metadata/git_commit.txt", f"{prefix}/dinov3/git_commit.txt"):
        files[name] = "b" * 40
    files[f"{prefix}/dinov3/conda_environment.yml"] = yaml.safe_dump({"dependencies": ["python=3.11.15", {"pip": ["torch==2.7.1"]}]})
    files["ic_vapo_hydra_20261104.tar.gz"] = pack(nested)
    path = tmp_path / "archive.tar.gz"
    path.write_bytes(pack(files))
    return path


def test_archive_logs_configs_and_csv_match_without_exporting_private_paths(tmp_path, monkeypatch):
    path = make_archive(tmp_path)
    monkeypatch.setattr(audit, "git_source_matches", lambda *_: None)
    report = audit.audit_archive(path, tmp_path)
    assert report["robotic_log_validation"]["rows"] == 8
    assert report["robotic_log_validation"]["all_rows_and_order_match"]
    assert report["hydra_validation"]["baseline"]["matching_requested_seeds"] == 2
    assert report["same_resolved_policy_reference"]
    assert "/private/" not in json.dumps(report)


@pytest.mark.parametrize("kwargs", [{"bad_seed": True}, {"bad_log": True}])
def test_config_and_log_discrepancies_fail_the_audit(tmp_path, kwargs):
    path = make_archive(tmp_path, **kwargs)
    with pytest.raises(audit.ValidationError):
        audit.audit_archive(path, tmp_path)


def make_supplement(tmp_path):
    cfg = {"affordance": {"gripper_cam": {"use": True, "use_distance": True, "densify_reward": True}},
           "env": {"seed": None, "use_egl": False}, "img_size": 64,
           "agent": {"net_cfg": {"hidden_dim": 256, "latent_dim": 16, "n_layers": 4,
                                 "actor_net": "Actor", "critic_net": "Critic"}}}
    train = {"n_epochs": 2, "dataset_name": "simulation", "dataset": {"cam": "gripper", "img_resize": {"gripper": 128},
             "radius": {"gripper": 20}}, "dataloader": {"batch_size": 2, "num_workers": 1},
             "trainer": {"gpus": 1, "precision": 16, "strategy": None},
             "model_cfg": {"dinov3_cfg": {"freeze_backbone": True, "normalize_input": True}}}
    root = "reproduction/dinov3_affordance/gripper_seed0/"
    files = {"baseline/.hydra/config.yaml": yaml.safe_dump(cfg), "vapo/.hydra/config.yaml": yaml.safe_dump(cfg),
             root + ".hydra/config.yaml": yaml.safe_dump(train), root + ".hydra/overrides.yaml": yaml.safe_dump(["n_epochs=2"]),
             root + "train_affordance.log": "train_data 10\nval_data 4\n",
             root + "console.log": 'Epoch 1, global step 10: val_miou reached 0.75 (best 0.75), saving model to "trained_models/best.ckpt" as top 2\n',
             "eval_affordance_dinov3_vs_baseline.py": "# recovered source\n"}
    path = tmp_path / "supplement.tar.gz"
    path.write_bytes(pack(files))
    primary = {"member_sha256": {"eval_affordance_dinov3_vs_baseline.py": hashlib.sha256(b"# recovered source\n").hexdigest()},
               "checkpoint_references": {model: {"policy_run_name": "vapo"} for model in audit.MODELS}}
    return path, primary


def test_supplement_recovers_configs_and_checkpoint_monitor_without_claiming_seed(tmp_path):
    path, primary = make_supplement(tmp_path)
    report = audit.audit_supplement(path, primary)
    assert report["saved_policy_configs"]["vapo"]["use_egl"] is False
    assert report["dino_training"]["train_samples_logged"] == [10]
    assert report["dino_training"]["training_seed_explicitly_recorded"] is False
    assert report["dino_training"]["best_checkpoint_save"] == {"epoch": 1, "global_step": 10, "val_miou": .75, "filename": "best.ckpt"}
    assert report["console_log"]["hex64_tokens_found"] == 0
    assert report["perception_evaluator_matches_primary_package"]


def test_supplement_rejects_a_different_perception_source(tmp_path):
    path, primary = make_supplement(tmp_path)
    primary["member_sha256"]["eval_affordance_dinov3_vs_baseline.py"] = "0" * 64
    with pytest.raises(audit.ValidationError):
        audit.audit_supplement(path, primary)


@pytest.mark.parametrize("mismatch", (False, True))
def test_identity_attachment_checks_all_metric_sources(tmp_path, mismatch):
    hashes = {}
    for name in ("metricas_por_imagem.csv", "resumo.json", "ab_results.csv"):
        (tmp_path / name).write_text(name)
        hashes[name] = hashlib.sha256(name.encode()).hexdigest()
    identity = {"metric_sources_sha256": hashes, "dataset_split_audit": {"matches_evaluated_ids_in_order": True},
                "historical_command_audit": {"evaluator_sha256": "a" * 64, "warmup_iterations_recorded": 5}}
    if mismatch:
        hashes["ab_results.csv"] = "b" * 64
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(identity))
    primary = {"member_sha256": {"eval_affordance_dinov3_vs_baseline.py": "a" * 64}}
    if mismatch:
        with pytest.raises(audit.ValidationError):
            audit.attach_identity_report(primary, path, tmp_path)
    else:
        audit.attach_identity_report(primary, path, tmp_path)
        assert primary["identity_evidence"]["reported_warmup_iterations"] == 5
        assert not primary["identity_evidence"]["command_execution_independently_attested"]
