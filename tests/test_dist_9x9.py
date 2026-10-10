import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.prepare_dist_9x9 import convert, OUTPUT_NAMES, EXOG_NAMES, TARGET_NAMES
from ts_benchmark.baselines.dag.models.dag_model import DAGModel
from ts_benchmark.data.utils import read_data
from ts_benchmark.utils.data_processing import split_channel


def model_config(series_dim=9, exog_dim=9):
    return SimpleNamespace(seq_len=168, pred_len=24, patch_len=48, stride=48,
                           use_c=True, use_t=True, use_c_exog=True, use_t_exog=True,
                           alpha=.9, beta=.1, series_dim=series_dim, infer_use_future=False,
                           enc_in=series_dim + exog_dim, d_model=16, d_ff=32, n_heads=4,
                           e_layers=1, dropout=0., factor=1, activation='gelu',
                           criterion=torch.nn.L1Loss())


def test_conversion_selects_named_channels_from_shuffled_source(tmp_path):
    names = list(reversed(OUTPUT_NAMES)) + ['TI21.PV']
    data = np.random.default_rng(7).normal(size=(240, len(names)))
    stamps = 1688169611.215 + np.arange(240) * 20
    source = tmp_path / 'source.npz'
    np.savez(source, data_array=data, time_stamp_array=stamps,
             vars_index_dict={name: i for i, name in enumerate(names)})
    output = tmp_path / 'forecasting' / 'DIST_9x9.csv'
    report = convert(source, output)
    restored = read_data(str(output))
    target, exog = split_channel(restored, report['target_channel'])
    assert list(target) == list(TARGET_NAMES)
    assert list(exog) == list(EXOG_NAMES)
    np.testing.assert_allclose(target, data[:, [names.index(n) for n in TARGET_NAMES]])
    np.testing.assert_allclose(exog, data[:, [names.index(n) for n in EXOG_NAMES]])
    assert report['excluded_variables'] == ['TI21.PV']
    assert (restored.index[1] - restored.index[0]).total_seconds() == 20
    assert restored.index[0].microsecond == 215000
    with pytest.raises(FileExistsError):
        convert(source, output)


@pytest.mark.parametrize('series_dim,exog_dim', [(1, 9), (9, 9)])
def test_batched_prediction_matches_independent_samples(series_dim, exog_dim):
    # Flattened patches are ordered by sample, then target channel. A sample's
    # attention gate must be applied to all its channels, never another sample.
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = DAGModel(model_config(series_dim, exog_dim)).eval()
    history = torch.randn(3, 168, series_dim + exog_dim)
    with torch.no_grad():
        batched, _ = model(history, None)
        independent = torch.cat([model(sample[None], None)[0] for sample in history])
    assert batched.shape == (3, 24, series_dim)
    torch.testing.assert_close(batched, independent, rtol=1e-5, atol=1e-6)


def test_nine_target_training_backward_and_unknown_future_inference():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = DAGModel(model_config())
    history = torch.randn(2, 168, 18)
    future_op = torch.randn(2, 24, 9)
    future_pv = torch.randn(2, 24, 9)
    prediction, auxiliary_loss = model(history, future_op)
    loss = torch.nn.functional.l1_loss(prediction, future_pv) + auxiliary_loss
    loss.backward()
    assert prediction.shape == future_pv.shape
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        a, _ = model(history, future_op)
        b, _ = model(history, future_op * 1000)
        c, _ = model(history, None)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a, c, rtol=0, atol=0)


def test_checked_config_selects_all_nine_targets():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'config/dist_9x9_config.json').read_text())
    frame = pd.DataFrame(np.zeros((2, 18)), columns=OUTPUT_NAMES)
    target, exog = split_channel(frame, config['evaluation_config']['strategy_args']['target_channel'])
    assert list(target) == list(TARGET_NAMES) and list(exog) == list(EXOG_NAMES)
    assert config['model_config']['models'][0]['model_hyper_params']['infer_use_future'] == 0
