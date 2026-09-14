"""Phase-2 scaffold tests: WiWorld-style complex-CIR loader against the synthetic CIR generator.

Real-data validation pending WiWorld-RealData download (docs/data_setup.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from channeldreamer.data import (
    WiWorldColumns,
    WiWorldLayout,
    generate_synthetic_cir,
    load_wiworld,
    resolve_wiworld_columns,
)
from channeldreamer.data.sequences import make_windows
from channeldreamer.data.wiworld import parse_quality


@pytest.fixture(scope="module")
def cir_data():
    return generate_synthetic_cir(n_segments=2, segment_length=30, n_taps=32, seed=3)


def test_synthetic_cir_shape_and_structure(cir_data):
    assert cir_data.cir.shape == (60, 2, 32)
    assert cir_data.cir.dtype == np.complex64
    assert np.isfinite(cir_data.cir).all()
    assert cir_data.good.dtype == bool and cir_data.positions.shape == (60, 2)
    # multipath structure: good samples carry much more energy than flagged (noise-only) ones
    energy = (np.abs(cir_data.cir) ** 2).sum(axis=(1, 2))
    if (~cir_data.good).any():
        assert energy[cir_data.good].mean() > 5 * energy[~cir_data.good].mean()
    # the two bands see the same delays but different phases
    b0, b1 = np.abs(cir_data.cir[0, 0]), np.abs(cir_data.cir[0, 1])
    assert np.corrcoef(b0, b1)[0, 1] > 0.9
    assert not np.allclose(np.angle(cir_data.cir[0, 0]), np.angle(cir_data.cir[0, 1]))


def test_loader_single_file_layout(tmp_path, cir_data):
    root = tmp_path / "data"
    cir_data.write_wiworld_layout(root / "wiworld_synth")
    ds = load_wiworld(root)
    assert ds.cir.shape == (60, 2, 32) and ds.cir.dtype == np.complex64
    np.testing.assert_allclose(ds.cir, cir_data.cir)
    np.testing.assert_array_equal(ds.good, cir_data.good)
    np.testing.assert_array_equal(ds.segment_ids, cir_data.segment_ids)
    assert ds.columns["cir"] == "cir_path" and ds.columns["quality"] == "quality_flag"
    assert ds.columns["segment"] == "route_id" and ds.columns["time"] == "timestamp_s"
    assert set(ds.columns["position"]) == {"pos_x_m", "pos_y_m"}
    assert ds.positions.shape == (60, 2)
    assert ds.power_delay_profile_db().shape == ds.cir.shape
    assert "flagged" in ds.summary()


def test_loader_per_band_columns_and_drop_bad(tmp_path, cir_data):
    root = tmp_path / "data"
    cir_data.write_wiworld_layout(root / "wiworld", per_band_columns=True)
    ds = load_wiworld(root)  # two cir_* columns -> auto per-band
    assert ds.columns["cir_per_band"] == ["cir_3.700ghz", "cir_6.775ghz"]
    np.testing.assert_allclose(ds.cir, cir_data.cir)
    ds2 = load_wiworld(root, drop_bad=True, columns=WiWorldColumns(cir_per_band=("cir_3.700ghz", "cir_6.775ghz")))
    assert ds2.n_samples == int(cir_data.good.sum()) and ds2.good.all()


def test_loader_explicit_columns_and_errors(tmp_path, cir_data):
    root = tmp_path / "data"
    manifest = cir_data.write_wiworld_layout(root / "wiworld")
    with pytest.raises(KeyError, match="header"):
        load_wiworld(root, columns=WiWorldColumns(cir="does_not_exist"))
    with pytest.raises(ValueError, match="bands"):
        load_wiworld(root, layout=WiWorldLayout(bands_hz=(3.7e9,)))
    # a manifest without a CIR column reports the header instead of guessing
    df = pd.read_csv(manifest).drop(columns=["cir_path"])
    df.to_csv(manifest, index=False)
    with pytest.raises(KeyError, match="header"):
        load_wiworld(root)


def test_real_imag_layouts(tmp_path, cir_data):
    root = tmp_path / "data" / "wiworld"
    (root / "cir").mkdir(parents=True)
    rows = []
    for i in range(4):
        arr = np.stack([cir_data.cir[i].real, cir_data.cir[i].imag], axis=-1)  # (bands, taps, 2)
        np.save(root / f"cir/s{i}.npy", arr)
        rows.append({"id": i, "cir_file": f"./cir/s{i}.npy", "qual": 1})
    pd.DataFrame(rows).to_csv(root / "m.csv", index=False)
    ds = load_wiworld(root.parent, layout=WiWorldLayout(complex_layout="last_axis_ri"))
    np.testing.assert_allclose(ds.cir, cir_data.cir[:4])
    with pytest.raises(ValueError, match="complex"):
        load_wiworld(root.parent)  # native layout on a real array must fail loudly


def test_column_resolution_and_quality_parsing():
    r = resolve_wiworld_columns(["sample", "h_band0", "h_band1", "valid", "utc", "run", "lat", "lon"])
    assert r["cir_per_band"] == ["h_band0", "h_band1"] and r["quality"] == "valid"
    assert r["time"] == "utc" and r["segment"] == "run" and r["position"] == ["lat", "lon"]
    np.testing.assert_array_equal(parse_quality([1, 0, "GOOD", "bad", True], (1, True, "good")),
                                  [True, False, True, False, True])


def test_cir_windows_reuse_shared_windowing(cir_data):
    """The shared windowing works on the real-valued PDP view, so Phase-2 models get the same splits."""
    pdp = 10 * np.log10(np.abs(cir_data.cir[:, 0]) ** 2 + 1e-12)  # (N, taps) of band 0
    w = make_windows(pdp, cir_data.segment_ids, history=4, horizon=1)
    assert w.histories.shape[1:] == (4, 32) and len(w) == 2 * (30 - 4)
