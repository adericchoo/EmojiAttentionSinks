"""Put src/ on the path and load config. Imported by the numbered scripts."""
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def load_config():
    with open(os.path.join(ROOT, "config", "models.yaml")) as f:
        return yaml.safe_load(f)


def select_model(cfg):
    # Pick which model to load from an optional command-line argument:
    #   python scripts/00_feasibility.py            -> primary (Pythia)
    #   python scripts/00_feasibility.py replication -> Qwen
    model_key = "primary"
    if len(sys.argv) > 1:
        model_key = sys.argv[1]
    if model_key not in cfg:
        raise SystemExit(f"unknown model key {model_key!r}; use 'primary' or 'replication'")
    entry = cfg[model_key]
    return entry["name"], entry.get("revision")
