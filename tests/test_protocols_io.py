"""End-to-end smoke tests for protocols, I/O, and provenance."""

import json

import numpy as np
import pytest

from kawasaki2d import io, protocols, provenance
from kawasaki2d.rng import make_rng


def test_prepare_and_quench_roundtrip(tmp_path):
    rng = make_rng(42)
    prep = protocols.prepare_initial_state(
        16, T_i=3.5, magnetisation_target=0, rng=rng, kernel="nonlocal", n_sweeps=50
    )
    assert prep.magnetisation == 0
    assert prep.kernel == "nonlocal"

    schedule = protocols.log_schedule(t_max=200, n_points=8)
    assert schedule[0] == 0
    rows = protocols.track_quench(prep.lattice, T_f=1.36, schedule=schedule, rng=rng)
    assert len(rows) == len(schedule)
    # M is constant across the whole trajectory
    assert {r["magnetisation"] for r in rows} == {0}

    csv_path = tmp_path / "trajectory.csv"
    io.write_trajectory_csv(csv_path, rows)
    data = io.read_trajectory_csv(csv_path)
    assert np.array_equal(data["sweep"], schedule.astype(float))
    assert len(data["energy"]) == len(schedule)


def test_append_only_guard(tmp_path):
    io.new_run_directory(tmp_path, "run-001")
    with pytest.raises(FileExistsError):
        io.new_run_directory(tmp_path, "run-001")


def test_manifest_build_and_write(tmp_path):
    cfg = {"N": 32, "T_i": 3.5, "T_f": 1.36, "M": 0, "seed": 7}
    man = provenance.Manifest.build(
        run_id="run-test",
        timestamp="2026-06-23T00:00:00Z",
        config=cfg,
        seeds={"base": 7},
    )
    # config hash is deterministic
    assert man.config_hash == provenance.config_hash(cfg)
    out = tmp_path / "data.csv"
    out.write_text("a,b\n1,2\n")
    man.add_output("data", out)
    assert man.outputs["data"] == provenance.file_checksum(out)

    man_path = tmp_path / "manifest.json"
    man.write(man_path)
    loaded = json.loads(man_path.read_text())
    assert loaded["config_hash"] == man.config_hash
    assert loaded["environment"]["rng"]["bit_generator"] == "PCG64"
    assert "python_version" in loaded["environment"]


def test_log_schedule_is_sorted_unique():
    sched = protocols.log_schedule(1000, 20)
    assert np.all(np.diff(sched) > 0)
    assert sched[0] == 0
