"""Entry points run offline: synthetic data for train/calibrate, a mock CUB tree for download."""
import json

from helpers import make_fake_cub
from planora_sf.cli import calibrate_snr, download_cub, train


def test_train_cli_dummy_smoke(capsys):
    rc = train.main(["--dataset", "dummy", "--epochs", "1", "--max-steps", "2", "--batch-size", "16",
                     "--device", "cpu", "--num-workers", "0"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["split"]["labelled_images"] == 24
    assert len(summary["history"]) == 1 and summary["history"][0]["steps"] == 2
    assert len(summary["evaluations"]) == 1 and "v2" in summary["evaluations"][0]


def test_calibrate_cli_dummy(tmp_path, capsys):
    out = tmp_path / "snr.json"
    assert calibrate_snr.main(["--dataset", "dummy", "--num-workers", "0", "--output", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "acceptance check" not in stdout  # no FX-A Table 7 reference for synthetic data
    payload = json.loads(out.read_text())
    cutoffs = [payload["cutoffs"][k] for k in ("5", "10", "15", "20", "25")]
    assert cutoffs == sorted(cutoffs)  # higher SNR targets need higher cut-offs
    assert set(map(int, payload["per_class_15db"])) <= {0, 1, 2, 3}  # labelled known classes only (U-10)


def test_download_cli_on_mock_cub_reports_and_flags_counts(tmp_path, capsys):
    make_fake_cub(tmp_path, num_classes=200, train_per_class=1, test_per_class=1, size=8)
    rc = download_cub.main(["--root", str(tmp_path), "--no-download", "--check-files"])
    report = json.loads(capsys.readouterr().out)
    assert report["train_images"] == 200 and report["classes"] == 200
    assert report["split"]["known_classes"] == 100 and report["split"]["labelled_images"] == 50
    assert rc == 1  # counts differ from the real CUB (5994 / 5794), so the CLI flags it
