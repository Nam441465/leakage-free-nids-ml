"""Unit tests for src/evaluation/plot_figures.py.

Verifies:
- Headless Agg backend & zero leaked open figures
- Exact Table 2 & Table 3 canonical class reordering from alphabetical JSON class_order
  (both per-class F1 heatmaps in Fig 6 and rows+columns of Fig 7b confusion matrix)
- Blanking cells < 0.05% in Fig 7b confusion matrix annotations
- Generation of all 5 publication-grade PNG figures (> 5 KB, valid PNG magic header)
- Graceful skipping when optional stress-test JSONs (LOFO / Gaussian noise) are absent
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.evaluation.plot_figures import (
    CICIDS2017_ORDER,
    LOFO_ORDER,
    NF_UNSW_ORDER,
    format_cm_annotations,
    generate_all_figures,
    main as plot_main,
    plot_fig7b_cm_cicids2017,
    reorder_confusion_matrix,
    reorder_per_class_matrix,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Explicit alphabetical JSON orders matching run_main.py output
ALPHABETICAL_CICIDS = [
    "Benign",
    "Bot",
    "BruteForce",
    "DDoS",
    "DoS",
    "PortScan",
    "Rare",
    "Web",
]

ALPHABETICAL_NF_UNSW = [
    "Benign",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Rare",
    "Reconnaissance",
    "Shellcode",
]


def _stat(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "formatted": f"{mean:.2f} ± {std:.2f}",
        "values": vals,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _create_mock_main_jsons(tables_dir: Path) -> None:
    """Create tiny mock binary & multiclass summary JSONs using alphabetical class_order."""
    # 1. Binary summaries
    cicids_binary = {
        "models": {
            "xgboost": {"f1": _stat([99.69, 99.66, 99.68])},
            "lightgbm": {"f1": _stat([99.68, 99.65, 99.67])},
            "random_forest": {"f1": _stat([99.60, 99.58, 99.59])},
        }
    }
    nf_binary = {
        "models": {
            "xgboost": {"f1": _stat([91.78, 91.83, 91.44])},
            "lightgbm": {"f1": _stat([91.78, 91.77, 91.45])},
            "random_forest": {"f1": _stat([93.15, 93.48, 93.06])},
        }
    }
    _write_json(tables_dir / "cicids2017_binary_summary.json", cicids_binary)
    _write_json(tables_dir / "nf_unsw_binary_summary.json", nf_binary)

    # 2. CICIDS2017 multiclass summary (alphabetical class_order: Web is last, Bot is idx 1)
    # Build a distinct, asymmetric 8x8 row-normalized CM with known off-diagonals
    cm_alpha = np.zeros((8, 8), dtype=np.float64)
    # idx: 0=Benign, 1=Bot, 2=BruteForce, 3=DDoS, 4=DoS, 5=PortScan, 6=Rare, 7=Web
    cm_alpha[0, 0] = 99.80
    cm_alpha[0, 1] = 0.08  # Benign -> Bot (>= 0.05, shown as "0.1")
    cm_alpha[0, 4] = 0.06  # Benign -> DoS (>= 0.05, shown as "0.1")
    cm_alpha[0, 5] = 0.04  # Benign -> PortScan (< 0.05, blank!)
    cm_alpha[0, 6] = 0.01  # Benign -> Rare (< 0.05, blank!)
    cm_alpha[0, 7] = 0.01  # Benign -> Web (< 0.05, blank!)
    cm_alpha[1, 0] = 0.80  # Bot -> Benign
    cm_alpha[1, 1] = 99.20
    cm_alpha[2, 0] = 0.13
    cm_alpha[2, 2] = 99.84
    cm_alpha[2, 7] = 0.03  # BruteForce -> Web (< 0.05, blank!)
    cm_alpha[3, 0] = 0.02  # DDoS -> Benign (< 0.05, blank!)
    cm_alpha[3, 3] = 99.98
    cm_alpha[4, 0] = 0.03
    cm_alpha[4, 4] = 99.97
    cm_alpha[5, 0] = 0.02
    cm_alpha[5, 5] = 99.94
    cm_alpha[5, 4] = 0.04  # PortScan -> DoS (< 0.05, blank!)
    cm_alpha[6, 0] = 18.50  # Rare -> Benign
    cm_alpha[6, 6] = 81.50
    cm_alpha[7, 0] = 1.00  # Web -> Benign
    cm_alpha[7, 5] = 0.09  # Web -> PortScan (>= 0.05, shown as "0.1")
    cm_alpha[7, 7] = 98.91

    cicids_f1_by_class = {
        "Benign": 99.9,
        "Bot": 70.1,
        "BruteForce": 99.5,
        "DDoS": 99.9,
        "DoS": 99.6,
        "PortScan": 99.4,
        "Rare": 86.3,
        "Web": 98.0,
    }

    cicids_multi = {
        "class_order": ALPHABETICAL_CICIDS,
        "models": {
            m_name: {
                "per_class_f1": {
                    cls: _stat([val - offset, val - offset + 0.1, val - offset - 0.1])
                    for cls, val in cicids_f1_by_class.items()
                },
                "confusion_matrix_normalized": cm_alpha.tolist(),
            }
            for m_name, offset in [
                ("xgboost", 0.0),
                ("lightgbm", 2.5),
                ("random_forest", 5.0),
            ]
        },
    }
    _write_json(tables_dir / "cicids2017_multiclass_summary.json", cicids_multi)

    # 3. NF-UNSW multiclass summary (alphabetical: Rare precedes Reconnaissance & Shellcode)
    nf_f1_by_class = {
        "Benign": 99.4,
        "Analysis": 16.0,
        "Backdoor": 17.6,
        "DoS": 45.6,
        "Exploits": 83.0,
        "Fuzzers": 71.7,
        "Generic": 63.6,
        "Rare": 71.6,
        "Reconnaissance": 31.8,
        "Shellcode": 68.1,
    }
    nf_multi = {
        "class_order": ALPHABETICAL_NF_UNSW,
        "models": {
            m_name: {
                "per_class_f1": {
                    cls: _stat([val - offset, val - offset + 0.2, val - offset - 0.2])
                    for cls, val in nf_f1_by_class.items()
                }
            }
            for m_name, offset in [
                ("xgboost", 0.0),
                ("lightgbm", 1.5),
                ("random_forest", 3.0),
            ]
        },
    }
    _write_json(tables_dir / "nf_unsw_multiclass_summary.json", nf_multi)


def _create_mock_stress_jsons(tables_dir: Path) -> None:
    """Create tiny mock LOFO and Gaussian noise summary JSONs."""
    # Intentionally store families out of LOFO_ORDER + include overall summary keys
    lofo_summary = {
        "models": {
            "xgboost": {
                "Web": {
                    "held_out_recall": _stat([3.2, 4.1, 3.8]),
                    "seen_attack_recall": _stat([99.9, 99.9, 99.9]),
                    "benign_fpr": _stat([0.12, 0.11, 0.13]),
                    "test_flows": 428,
                },
                "Bot": {
                    "held_out_recall": _stat([2.1, 1.8, 2.5]),
                    "seen_attack_recall": _stat([99.9, 99.9, 99.9]),
                    "benign_fpr": _stat([0.11, 0.12, 0.12]),
                    "test_flows": 390,
                },
                "DoS": {
                    "held_out_recall": _stat([28.5, 31.2, 29.8]),
                    "seen_attack_recall": _stat([99.8, 99.8, 99.8]),
                    "benign_fpr": _stat([0.10, 0.11, 0.10]),
                    "test_flows": 38749,
                },
                "DDoS": {
                    "held_out_recall": _stat([19.4, 21.0, 20.1]),
                    "seen_attack_recall": _stat([99.9, 99.9, 99.9]),
                    "benign_fpr": _stat([0.12, 0.12, 0.11]),
                    "test_flows": 25603,
                },
                "PortScan": {
                    "held_out_recall": _stat([0.9, 1.2, 1.0]),
                    "seen_attack_recall": _stat([99.9, 99.9, 99.9]),
                    "benign_fpr": _stat([0.11, 0.12, 0.11]),
                    "test_flows": 18139,
                },
                "BruteForce": {
                    "held_out_recall": _stat([5.5, 6.2, 5.9]),
                    "seen_attack_recall": _stat([99.9, 99.9, 99.9]),
                    "benign_fpr": _stat([0.12, 0.12, 0.12]),
                    "test_flows": 1830,
                },
                # Extra summary keys produced by run_leave_one_out.py
                "overall_seen_attack_recall": _stat([99.9, 99.9, 99.9]),
                "overall_benign_fpr": _stat([0.11, 0.12, 0.11]),
            }
        }
    }
    _write_json(tables_dir / "leave_one_out_summary.json", lofo_summary)

    noise_summary = {
        "models": {
            "xgboost": {
                "sigma_0.0": {
                    "f1": _stat([99.69, 99.67, 99.68]),
                    "precision": _stat([99.44, 99.38, 99.41]),
                    "pr_auc": _stat([99.98, 99.98, 99.99]),
                    "recall": _stat([99.95, 99.95, 99.94]),
                    "fpr_at_0.5": _stat([0.12, 0.13, 0.12]),
                },
                "sigma_0.1": {
                    "f1": _stat([5.84, 5.90, 5.78]),
                    "precision": _stat([95.95, 96.10, 95.80]),
                    "pr_auc": _stat([80.93, 81.10, 80.75]),
                    "recall": _stat([3.02, 3.05, 2.98]),
                    "fpr_at_0.5": _stat([0.02, 0.02, 0.02]),
                },
                "sigma_0.25": {
                    "f1": _stat([1.84, 1.90, 1.78]),
                    "precision": _stat([90.0, 90.5, 89.5]),
                    "pr_auc": _stat([65.0, 65.5, 64.5]),
                    "recall": _stat([0.93, 0.96, 0.90]),
                    "fpr_at_0.5": _stat([0.01, 0.01, 0.01]),
                },
                "sigma_0.5": {
                    "f1": _stat([0.67, 0.70, 0.64]),
                    "precision": _stat([82.0, 82.5, 81.5]),
                    "pr_auc": _stat([50.0, 50.5, 49.5]),
                    "recall": _stat([0.34, 0.36, 0.32]),
                    "fpr_at_0.5": _stat([0.01, 0.01, 0.01]),
                },
            },
            "lightgbm": {
                "sigma_0.0": {"f1": _stat([99.68, 99.66, 99.67])},
                "sigma_0.1": {"f1": _stat([8.20, 8.50, 7.90])},
                "sigma_0.25": {"f1": _stat([2.50, 2.70, 2.30])},
                "sigma_0.5": {"f1": _stat([1.10, 1.20, 1.00])},
            },
            "random_forest": {
                "sigma_0.0": {"f1": _stat([99.60, 99.58, 99.59])},
                "sigma_0.1": {"f1": _stat([45.0, 46.2, 43.8])},
                "sigma_0.25": {"f1": _stat([22.0, 23.1, 20.9])},
                "sigma_0.5": {"f1": _stat([10.5, 11.0, 10.0])},
            },
        }
    }
    _write_json(tables_dir / "noise_test_summary.json", noise_summary)


def test_agg_backend_active() -> None:
    """Headless Agg backend must be active to avoid GUI dependency."""
    assert matplotlib.get_backend().lower() == "agg"


def test_table2_and_table3_class_reordering_fig6(tmp_path: Path) -> None:
    """Verify per-class F1 matrix reorders alphabetical JSON keys into exact Table 2 & 3 order."""
    tables_dir = tmp_path / "tables"
    _create_mock_main_jsons(tables_dir)

    with open(tables_dir / "cicids2017_multiclass_summary.json", encoding="utf-8") as f:
        cicids_data = json.load(f)
    with open(tables_dir / "nf_unsw_multiclass_summary.json", encoding="utf-8") as f:
        nf_data = json.load(f)

    # Confirm mock JSONs really are alphabetical (Rare before Reconnaissance; Web last)
    assert cicids_data["class_order"][-1] == "Web"
    assert nf_data["class_order"].index("Rare") < nf_data["class_order"].index("Reconnaissance")

    c_models, c_classes, c_mat = reorder_per_class_matrix(
        cicids_data["models"], CICIDS2017_ORDER
    )
    assert c_models == ["xgboost", "lightgbm", "random_forest"]
    assert c_classes == CICIDS2017_ORDER
    # Verify exact value mapping after reordering (XGBoost row 0):
    # DoS=99.6 (idx 1), Web=98.0 (idx 5), Bot=70.1 (idx 6), Rare=86.3 (idx 7)
    assert np.isclose(c_mat[0, 1], 99.6)
    assert np.isclose(c_mat[0, 5], 98.0)
    assert np.isclose(c_mat[0, 6], 70.1)
    assert np.isclose(c_mat[0, 7], 86.3)

    n_models, n_classes, n_mat = reorder_per_class_matrix(
        nf_data["models"], NF_UNSW_ORDER
    )
    assert n_models == ["xgboost", "lightgbm", "random_forest"]
    assert n_classes == NF_UNSW_ORDER
    # Rare must move AFTER Reconnaissance and Shellcode to final column (idx 9)
    assert n_classes[-3:] == ["Reconnaissance", "Shellcode", "Rare"]
    assert np.isclose(n_mat[0, 7], 31.8)  # Reconnaissance
    assert np.isclose(n_mat[0, 8], 68.1)  # Shellcode
    assert np.isclose(n_mat[0, 9], 71.6)  # Rare


def test_confusion_matrix_reordering_and_threshold_blanking_fig7b(tmp_path: Path) -> None:
    """Verify BOTH rows and columns are permuted to CICIDS2017_ORDER and < 0.05% cells are blank."""
    tables_dir = tmp_path / "tables"
    _create_mock_main_jsons(tables_dir)

    with open(tables_dir / "cicids2017_multiclass_summary.json", encoding="utf-8") as f:
        cicids_data = json.load(f)

    cm_alpha = np.asarray(
        cicids_data["models"]["xgboost"]["confusion_matrix_normalized"], dtype=np.float64
    )
    ordered_classes, cm_reordered = reorder_confusion_matrix(
        cm_alpha, cicids_data["class_order"], CICIDS2017_ORDER
    )
    assert ordered_classes == CICIDS2017_ORDER

    # CICIDS2017_ORDER indices:
    # 0=Benign, 1=DoS, 2=DDoS, 3=PortScan, 4=BruteForce, 5=Web, 6=Bot, 7=Rare
    # Check asymmetric off-diagonals landed in exact reordered (true, pred) coordinates:
    assert np.isclose(cm_reordered[0, 6], 0.08)   # Benign -> Bot
    assert np.isclose(cm_reordered[0, 1], 0.06)   # Benign -> DoS
    assert np.isclose(cm_reordered[0, 3], 0.04)   # Benign -> PortScan
    assert np.isclose(cm_reordered[6, 0], 0.80)   # Bot -> Benign
    assert np.isclose(cm_reordered[6, 6], 99.20)  # Bot -> Bot
    assert np.isclose(cm_reordered[7, 0], 18.50)  # Rare -> Benign
    assert np.isclose(cm_reordered[7, 7], 81.50)  # Rare -> Rare
    assert np.isclose(cm_reordered[5, 3], 0.09)   # Web -> PortScan
    assert np.isclose(cm_reordered[5, 5], 98.91)  # Web -> Web

    # Verify annotation formatting with threshold = 0.05%
    annot = format_cm_annotations(cm_reordered, threshold=0.05)
    assert annot[0, 0] == "99.8"
    assert annot[0, 6] == "0.1"   # 0.08% >= 0.05%
    assert annot[0, 1] == "0.1"   # 0.06% >= 0.05%
    assert annot[0, 3] == ""      # 0.04% < 0.05% -> blank
    assert annot[0, 7] == ""      # 0.01% < 0.05% -> blank
    assert annot[5, 3] == "0.1"   # 0.09% >= 0.05%
    assert annot[7, 0] == "18.5"
    assert annot[7, 7] == "81.5"
    assert annot[2, 0] == ""      # 0.02% < 0.05% -> blank


def test_generate_all_five_figures_valid_pngs(tmp_path: Path) -> None:
    """All 5 figures are generated, valid PNGs (> 5 KB), with zero leaked matplotlib figures."""
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    _create_mock_main_jsons(tables_dir)
    _create_mock_stress_jsons(tables_dir)

    generated = generate_all_figures(tables_dir, figures_dir)
    expected_names = [
        "fig4_binary_f1.png",
        "fig6_multiclass_per_class_f1.png",
        "fig7b_xgboost_cm_cicids2017.png",
        "fig9_leave_one_out_recall.png",
        "fig10_gaussian_noise_f1.png",
    ]

    assert [p.name for p in generated] == expected_names
    for fname in expected_names:
        fpath = figures_dir / fname
        assert fpath.exists(), f"Missing generated figure: {fpath}"
        raw = fpath.read_bytes()
        assert raw.startswith(PNG_MAGIC), f"{fname} does not have valid PNG header"
        assert len(raw) > 5_120, f"{fname} unexpectedly small ({len(raw)} bytes)"

    # Confirm all matplotlib figures were closed cleanly
    assert plt.get_fignums() == []


def test_graceful_skip_when_stress_jsons_absent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """When LOFO and Gaussian noise JSONs do not exist yet, skip Fig 9 & 10 gracefully."""
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    _create_mock_main_jsons(tables_dir)

    with caplog.at_level("INFO", logger="plot_figures"):
        generated = generate_all_figures(tables_dir, figures_dir)

    assert [p.name for p in generated] == [
        "fig4_binary_f1.png",
        "fig6_multiclass_per_class_f1.png",
        "fig7b_xgboost_cm_cicids2017.png",
    ]
    assert not (figures_dir / "fig9_leave_one_out_recall.png").exists()
    assert not (figures_dir / "fig10_gaussian_noise_f1.png").exists()
    assert any("fig9_leave_one_out_recall.png" in rec.message for rec in caplog.records)
    assert any("fig10_gaussian_noise_f1.png" in rec.message for rec in caplog.records)
    assert plt.get_fignums() == []


def test_graceful_skip_when_tables_dir_empty(tmp_path: Path) -> None:
    """Empty tables directory skips all 5 figures without crashing."""
    tables_dir = tmp_path / "empty_tables"
    tables_dir.mkdir()
    figures_dir = tmp_path / "figures"

    generated = generate_all_figures(tables_dir, figures_dir)
    assert generated == []
    assert plt.get_fignums() == []


def test_cm_raw_fallback_when_normalized_absent(tmp_path: Path) -> None:
    """Fig 7b works cleanly when only confusion_matrix_raw is provided."""
    cm_raw = np.eye(8, dtype=np.int64) * 1000
    cm_raw[6, 0] = 200  # Rare -> Benign
    payload = {
        "class_order": ALPHABETICAL_CICIDS,
        "models": {"xgboost": {"confusion_matrix_raw": cm_raw.tolist()}},
    }
    json_path = tmp_path / "cicids_raw_only.json"
    _write_json(json_path, payload)
    out_path = tmp_path / "fig7b_raw.png"

    result = plot_fig7b_cm_cicids2017(json_path, out_path)
    assert result == out_path
    assert out_path.exists()
    raw = out_path.read_bytes()
    assert raw.startswith(PNG_MAGIC)
    assert len(raw) > 5_120
    assert plt.get_fignums() == []


def test_cli_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI --tables-dir and --output-dir generate all 5 PNGs without error."""
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    _create_mock_main_jsons(tables_dir)
    _create_mock_stress_jsons(tables_dir)

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_figures.py",
            "--tables-dir",
            str(tables_dir),
            "--output-dir",
            str(figures_dir),
        ],
    )
    plot_main()

    expected = [
        "fig4_binary_f1.png",
        "fig6_multiclass_per_class_f1.png",
        "fig7b_xgboost_cm_cicids2017.png",
        "fig9_leave_one_out_recall.png",
        "fig10_gaussian_noise_f1.png",
    ]
    for fname in expected:
        fpath = figures_dir / fname
        assert fpath.exists()
        assert fpath.read_bytes().startswith(PNG_MAGIC)
    assert plt.get_fignums() == []

