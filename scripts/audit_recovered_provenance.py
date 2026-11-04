"""Check the private provenance archive against the recovered robot CSV.

Reads archive members in memory; never extracts or executes archived scripts.
Requires PyYAML for safe configuration parsing. Exports only a curated report.
"""

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

import yaml

from analyze_recovered_results import MODELS, ValidationError, analyze_robotics, require, sha256


EPISODE = re.compile(r"EVALUATION \[(\d+)\] ([^,]+), Return: ([+-]?\d+(?:\.\d+)?), Success: (True|False), Steps: (\d+)")
COMMIT = re.compile(r"[0-9a-f]{40}")
PREFIX = "reproduction/ab_evaluation"


def episode_records(text, model, seed):
    return [(model, seed, int(index), obj, int(success == "True"), float(ret), int(steps))
            for index, obj, ret, success, steps in EPISODE.findall(text)]


def csv_records(data):
    return [(r["model"], int(r["seed"]), int(r["episode_index"]), r["object"], int(r["success"]),
             float(r["return"]), int(r["steps"])) for r in csv.DictReader(io.StringIO(data.decode("utf-8-sig")))]


def member_bytes(archive, name):
    member = archive.getmember(name)
    require(member.isfile() and member.size < 64 * 1024 * 1024, "Unexpected archive member type/size")
    return archive.extractfile(member).read()


def read_yaml(archive, name):
    return yaml.safe_load(member_bytes(archive, name))


def flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, child in value.items():
        result.update(flatten(child, prefix + "." + str(key) if prefix else str(key)))
    return result


def argument_defaults(source):
    result = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                for keyword in node.keywords:
                    if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                        result[node.args[0].value] = keyword.value.value
    return result


def parse_status(text):
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    result = {int(row["seed"]): int(row["exit_code"]) for row in rows}
    require(len(result) == len(rows), "Duplicate status seed")
    return result


def git_source_matches(source, commit):
    require(bool(COMMIT.fullmatch(commit)), "Invalid commit identifier")
    try:
        saved = subprocess.check_output(["git", "show", commit + ":scripts/eval_tabletop.py"],
                                        cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return source.replace(b"\r\n", b"\n") == saved.replace(b"\r\n", b"\n")


def audit_archive(archive_path, metrics_dir):
    archive_path, metrics_dir = Path(archive_path), Path(metrics_dir)
    robot = analyze_robotics(metrics_dir / "ab_results.csv")
    source_csv = (metrics_dir / "ab_results.csv").read_bytes()
    expected = csv_records(source_csv)
    seeds = robot["seed_labels"]
    parsed, runtime_commits, log_counts = [], Counter(), {}
    member_hashes, captured_commits, resolved = {}, {}, {}
    with tarfile.open(archive_path, "r:gz") as archive:
        names = [m.name for m in archive.getmembers()]
        require(len(names) == len(set(names)), "Duplicate archive member names")
        require(member_bytes(archive, PREFIX + "/ab_results.csv") == source_csv, "Archived CSV differs from recovered CSV")
        for model in MODELS:
            counts = []
            statuses = parse_status(member_bytes(archive, f"{PREFIX}/runs/{model}/status.tsv").decode())
            require(set(statuses) == set(seeds), "Status seeds do not match CSV seeds")
            require(all(code == 0 for code in statuses.values()), "Recorded run failure")
            for seed in seeds:
                text = member_bytes(archive, f"{PREFIX}/runs/{model}/seed_{seed}.log").decode(errors="replace")
                rows = episode_records(text, model, seed)
                require(bool(rows), "Run log has no episode records")
                final = re.search(r"Full evaluation over \d+ objs\s+Success: (\d+)/(\d+)", text)
                require(final is not None and tuple(map(int, final.groups())) == (sum(r[4] for r in rows), len(rows)),
                        "Log final summary disagrees with episode records")
                parsed.extend(rows)
                counts.append(len(rows))
                runtime_commits.update(re.findall(r"Using VREnv with commit ([0-9a-f]{40})", text))
            log_counts[model] = {"logs": len(counts), "episode_count_distribution": dict(Counter(counts)),
                                 "zero_exit_statuses": len(statuses)}
            resolved[model] = read_yaml(archive, f"{PREFIX}/{model}/config_resolved.yaml")
        require(parsed == expected, "Parsed run logs do not exactly match CSV rows/order")
        for name in ("eval_affordance_dinov3_vs_baseline.py", "scripts/eval_tabletop.py", "run_baseline_100.sh",
                     "run_dinov3_100.sh", "parse_ab_results.py", "ic_vapo_hydra_20261104.tar.gz"):
            member_hashes[name] = hashlib.sha256(member_bytes(archive, name)).hexdigest()
        for name in (f"{PREFIX}/git_commit.txt", f"{PREFIX}/metadata/git_commit.txt", f"{PREFIX}/dinov3/git_commit.txt"):
            commit = member_bytes(archive, name).decode().strip()
            require(bool(COMMIT.fullmatch(commit)), "Invalid captured commit")
            captured_commits[name] = commit
        evaluator = member_bytes(archive, "scripts/eval_tabletop.py")
        evaluation_source_matches = {commit: git_source_matches(evaluator, commit) for commit in sorted(set(captured_commits.values()))}
        perception_source = member_bytes(archive, "eval_affordance_dinov3_vs_baseline.py").decode()
        defaults = argument_defaults(perception_source)
        # Static source locations make the protocol review repeatable without importing Torch.
        ast.parse(evaluator.decode())
        source_locations = {node.name: node.lineno for node in ast.walk(ast.parse(perception_source))
                            if isinstance(node, ast.FunctionDef) and node.name in
                            ("predict_and_time", "warm_up", "model_summary", "vapo_validation_iou", "main")}
        config_report, configs = {}, {}
        with tarfile.open(fileobj=io.BytesIO(member_bytes(archive, "ic_vapo_hydra_20261104.tar.gz")), mode="r:gz") as nested:
            nested_names = [m.name for m in nested.getmembers()]
            require(len(nested_names) == len(set(nested_names)), "Duplicate nested archive names")
            for model in MODELS:
                composed_env_seeds, unique_configs, egl_values = Counter(), set(), set()
                for seed in seeds:
                    root = f"./{model}_seed{seed}/.hydra/"
                    config = read_yaml(nested, root + "config.yaml")
                    overrides = read_yaml(nested, root + "overrides.yaml")
                    require(config["test"]["eval_seed"] == seed, "Hydra evaluation seed mismatch")
                    require(f"test.eval_seed={seed}" in overrides, "Missing seed override")
                    composed_env_seeds[str(config["env"]["seed"])] += 1
                    egl_values.add(config["env"]["use_egl"])
                    if seed == seeds[0]:
                        configs[model] = config
                    flat = flatten(config)
                    flat.pop("test.eval_seed")
                    unique_configs.add(json.dumps(flat, sort_keys=True))
                config_report[model] = {"configs": len(seeds), "matching_requested_seeds": len(seeds),
                                        "composed_env_seed_values": dict(composed_env_seeds),
                                        "composed_egl_values": sorted(egl_values),
                                        "distinct_configs_excluding_test_seed": len(unique_configs)}
        left, right = (flatten(configs[m]) for m in MODELS)
        changed_fields = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
        same_policy = all(resolved["baseline"]["test"][key] == resolved["dinov3"]["test"][key]
                          for key in ("folder_name", "model_name"))
        references = {}
        for model in MODELS:
            cfg = resolved[model]
            references[model] = {
                "policy_name": cfg["test"]["model_name"],
                "policy_run_name": Path(cfg["test"]["folder_name"]).name,
                "gripper_checkpoint_filename": Path(cfg["affordance"]["gripper_cam"]["model_path"]).name,
                "static_checkpoint_filename": Path(cfg["static_cam_aff_path"]).name,
                "encoder_type": cfg["affordance"]["gripper_cam"]["hyperparameters"]["cfg"]["encoder_type"],
                "decoder_channels": cfg["affordance"]["gripper_cam"]["hyperparameters"]["cfg"]["unet_cfg"]["decoder_channels"],
                "policy_img_size": cfg["img_size"],
            }
        env_data = read_yaml(archive, f"{PREFIX}/dinov3/conda_environment.yml")
        packages = {}
        wanted = {"torch", "torchvision", "pytorch-lightning", "segmentation-models-pytorch", "numpy", "pybullet", "hydra-core"}
        for item in env_data["dependencies"]:
            if isinstance(item, str) and item.startswith("python="):
                packages["python"] = item.split("=")[1]
            elif isinstance(item, dict):
                for dep in item.get("pip", []):
                    name, separator, version = dep.partition("==")
                    if separator and name in wanted:
                        packages[name] = version
    return {
        "schema_version": 1, "archive_sha256": sha256(archive_path), "archive_bytes": archive_path.stat().st_size,
        "audit_script_sha256": sha256(Path(__file__)), "member_sha256": member_hashes,
        "robotic_csv_sha256": hashlib.sha256(source_csv).hexdigest(),
        "robotic_log_validation": {"rows": len(parsed), "all_rows_and_order_match": True,
                                    "archived_csv_byte_identical": True, "runs": log_counts},
        "captured_commits": captured_commits, "runtime_commit_mentions": dict(runtime_commits),
        "archived_evaluator_matches_git_versions": evaluation_source_matches,
        "hydra_validation": config_report, "cross_model_changed_composed_fields": changed_fields,
        "same_resolved_policy_reference": same_policy, "checkpoint_references": references,
        "recorded_environment_packages": packages,
        "perception_source_review": {
            "function_lines": source_locations,
            "default_warmup_iterations": defaults.get("--warmup"),
            "actual_warmup_override_recovered": False,
            "protocol": "Batch 1; inference_mode; eval models; optional FP16 autocast; CUDA synchronization before/after forward; perf_counter.",
            "timed_scope": "Encoder, decoder, segmentation head and direction head; excludes preprocessing, H2D transfer, argmax, Hough voting and SAC.",
            "model_order": "baseline then dinov3 for every sample; not randomized",
            "summary_median": "torch.tensor(latencies).median(): lower middle sample in float32",
            "full_perception_command_recovered": False,
        },
        "limitations": ["Checkpoint filenames/references recovered; checkpoint content hashes absent.",
                        "Hydra seed requests and evaluator propagation recovered; all RNG streams and identical physical states not proven.",
                        "Hydra files are composed CLI configs, not the final policy-merged runtime configs.",
                        "Perception saved training Hydra config, dataset split manifest and exact invocation absent.",
                        "No archived code was executed; no model training, inference or robot evaluation rerun."],
    }


def audit_supplement(path, primary_report):
    """Read saved training configs and stream the large console log in bounded chunks."""
    path = Path(path)
    train_root = "reproduction/dinov3_affordance/gripper_seed0/"
    selected_runs = {primary_report["checkpoint_references"][model]["policy_run_name"] for model in MODELS}
    require(len(selected_runs) == 1, "Primary captures reference different policy runs")
    selected_run = selected_runs.pop()
    with tarfile.open(path, "r:gz") as archive:
        names = [m.name for m in archive.getmembers()]
        require(len(names) == len(set(names)), "Duplicate supplementary archive names")
        saved = {name: read_yaml(archive, name + "/.hydra/config.yaml") for name in ("baseline", "vapo")}
        require(selected_run in saved, "Selected policy config is absent from supplementary package")
        training = read_yaml(archive, train_root + ".hydra/config.yaml")
        overrides = read_yaml(archive, train_root + ".hydra/overrides.yaml")
        small_hashes = {name: hashlib.sha256(member_bytes(archive, name)).hexdigest()
                        for name in names if name.endswith((".yaml", ".py"))}
        source_hash = small_hashes["eval_affordance_dinov3_vs_baseline.py"]
        require(source_hash == primary_report["member_sha256"]["eval_affordance_dinov3_vs_baseline.py"],
                "Supplementary perception evaluator differs from the primary package")
        train_log = member_bytes(archive, train_root + "train_affordance.log").decode(errors="replace")
        train_counts = re.findall(r"train_data (\d+)", train_log)
        val_counts = re.findall(r"val_data (\d+)", train_log)
        console = archive.getmember(train_root + "console.log")
        require(console.isfile(), "Training console is not a regular file")
        handle = archive.extractfile(console)
        digest, tail, bytes_read = hashlib.sha256(), b"", 0
        needles = (b"eval_affordance_dinov3_vs_baseline", b"--warmup", b"sha256", b"checksum", b"manifest", b"split.json")
        presence = {key.decode(): False for key in needles}
        hashes, saves = set(), set()
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
            window = tail + chunk
            lower = window.lower()
            for key in needles:
                presence[key.decode()] |= key in lower
            hashes.update(re.findall(rb"\b[a-f0-9]{64}\b", lower))
            if b"val_miou reached" in window:
                saves.update(re.findall(rb'Epoch (\d+), global step (\d+): val_miou reached ([0-9.]+)[^\n\r]{0,100}saving model to "([^\"]+)"', window))
            tail = window[-4096:]
        require(bytes_read == console.size, "Incomplete training console read")
        best = max(saves, key=lambda row: float(row[2])) if saves else None
        policy_summary = {}
        for name, cfg in saved.items():
            gripper = cfg["affordance"]["gripper_cam"]
            net = cfg["agent"]["net_cfg"]
            policy_summary[name] = {
                "use_affordance": gripper["use"], "use_distance": gripper["use_distance"],
                "densify_reward": gripper["densify_reward"], "env_seed": cfg["env"]["seed"],
                "use_egl": cfg["env"]["use_egl"], "policy_img_size": cfg["img_size"],
                "network": {key: net[key] for key in ("hidden_dim", "latent_dim", "n_layers", "actor_net", "critic_net")},
            }
        dino_cfg = training["model_cfg"]["dinov3_cfg"]
        return {
            "archive_sha256": sha256(path), "archive_bytes": path.stat().st_size,
            "member_sha256": small_hashes, "perception_evaluator_matches_primary_package": True,
            "saved_policy_configs": policy_summary,
            "selected_policy_config": selected_run + "/.hydra/config.yaml",
            "selection_basis": "Selected by the common policy-run directory name in both primary resolved captures.",
            "selected_policy_uses_affordance": policy_summary[selected_run]["use_affordance"],
            "dino_training": {
                "n_epochs": training["n_epochs"], "camera": training["dataset"]["cam"],
                "dataset_name": training["dataset_name"], "input_size": training["dataset"]["img_resize"]["gripper"],
                "radius": training["dataset"]["radius"]["gripper"],
                "batch_size": training["dataloader"]["batch_size"], "num_workers": training["dataloader"]["num_workers"],
                "trainer_gpus": training["trainer"]["gpus"], "trainer_precision": training["trainer"]["precision"],
                "trainer_strategy": training["trainer"]["strategy"],
                "freeze_backbone": dino_cfg["freeze_backbone"], "normalize_input": dino_cfg["normalize_input"],
                "train_samples_logged": sorted(set(map(int, train_counts))),
                "validation_samples_logged": sorted(set(map(int, val_counts))),
                "training_seed_explicitly_recorded": any("seed" in str(x).split("=")[0] for x in overrides),
                "best_checkpoint_save": {"epoch": int(best[0]), "global_step": int(best[1]), "val_miou": float(best[2]),
                                         "filename": Path(best[3].decode()).name} if best else None,
            },
            "console_log": {"bytes": bytes_read, "sha256": digest.hexdigest(), "hex64_tokens_found": len(hashes),
                            "searched_keyword_presence": presence},
            "package_scope_limits": ["These two archives do not contain checkpoint binaries or a checkpoint content-hash manifest.",
                          "These two archives do not contain a dataset/split manifest or exact perception command.",
                          "Training seed is not established by the gripper_seed0 folder name.",
                          "The audit inspected saved records and did not execute the researcher's existing experiments again."],
        }


def attach_identity_report(report, path, metrics_dir):
    identity = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for name in ("metricas_por_imagem.csv", "resumo.json", "ab_results.csv"):
        require(identity["metric_sources_sha256"][name] == sha256(Path(metrics_dir) / name),
                "Identity audit is associated with different metric inputs")
    require(identity["dataset_split_audit"]["matches_evaluated_ids_in_order"] is True,
            "Identity audit did not verify the ordered partition")
    require(identity["historical_command_audit"]["evaluator_sha256"] == report["member_sha256"]["eval_affordance_dinov3_vs_baseline.py"],
            "Identity audit references a different evaluator")
    report["identity_evidence"] = {
        "report_sha256": sha256(Path(path)), "checkpoint_manifest_recorded": True,
        "ordered_validation_ids_matched": True,
        "reported_warmup_iterations": identity["historical_command_audit"]["warmup_iterations_recorded"],
        "command_execution_independently_attested": False,
    }
    report["limitations"] = [
        "Checkpoint hashes are recorded in a recovered manifest; weight binaries were not locally rehashed or runtime-attested.",
        "Sample IDs and order match all split manifests; raw image/label contents and dataset release are not content-attested.",
        "The reported historical command is compatible with code and AMP metadata, without a contemporaneous execution trace.",
        "Training seed and complete control of all RNG streams/physical initial states remain undocumented.",
        "Real comparative images were not found in the reported searches; no visual inference examples were fabricated.",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, help="Optional saved-policy/training-config package")
    parser.add_argument("--identity-report", type=Path, help="Optional audited checkpoint/split/command evidence")
    args = parser.parse_args()
    try:
        report = audit_archive(args.archive, args.metrics_dir)
        if args.supplement:
            report["supplement"] = audit_supplement(args.supplement, report)
            report["limitations"] = report["supplement"]["package_scope_limits"]
        if args.identity_report:
            attach_identity_report(report, args.identity_report, args.metrics_dir)
    except (ValidationError, OSError, KeyError, ValueError, tarfile.TarError, yaml.YAMLError):
        parser.exit(2, "Provenance audit failed. Check the private archive schema and input consistency.\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Verified {report['robotic_log_validation']['rows']} CSV rows against all recovered run logs.")


if __name__ == "__main__":
    main()
