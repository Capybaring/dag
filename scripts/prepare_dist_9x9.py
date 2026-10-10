"""Convert DIST NPZ into nine OP covariates and nine PV forecasting targets."""
import argparse
import hashlib
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ts_benchmark.data.utils import read_data, load_series_info
from ts_benchmark.utils.data_processing import split_channel

LOOPS = ('FIC05', 'FIC09', 'FIC11', 'LIC02', 'LIC03', 'LIC04', 'LIC05', 'LIC06', 'LIC12')
EXOG_NAMES = tuple(name + '.OP' for name in LOOPS)
TARGET_NAMES = tuple(name + '.PV' for name in LOOPS)
OUTPUT_NAMES = EXOG_NAMES + TARGET_NAMES
TARGET_CHANNEL = list(range(9, 18))


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError('Unsupported metadata type: %s.%s' % (module, name))
        return allowed[(module, name)]


def load_mapping(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open('vars_index_dict.npy') as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError('Unsupported metadata NPY version')
            if shape != () or dtype != np.dtype('O'):
                raise ValueError('Expected scalar variable dictionary')
            mapping = MetadataUnpickler(f).load().item()
    if not isinstance(mapping, dict):
        raise ValueError('Variable mapping must be a dictionary')
    if not all(isinstance(k, str) and isinstance(v, int) for k, v in mapping.items()):
        raise ValueError('Variable mapping requires string names and integer indices')
    return mapping


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def convert(source, output, timezone='Asia/Shanghai', limit=None):
    source, output = Path(source), Path(output)
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('Refusing to overwrite CSV or partial output')
    mapping = load_mapping(source)
    with np.load(source, allow_pickle=False) as archive:
        data, stamps = archive['data_array'], archive['time_stamp_array']
    if data.ndim != 2 or stamps.shape != (len(data),):
        raise ValueError('Data and timestamp dimensions do not match')
    if sorted(mapping.values()) != list(range(data.shape[1])):
        raise ValueError('Variable indices must be a complete permutation')
    missing = set(OUTPUT_NAMES) - set(mapping)
    if missing:
        raise ValueError('Missing selected variables: ' + ', '.join(sorted(missing)))
    if not np.isfinite(data).all() or not np.isfinite(stamps).all():
        raise ValueError('Source contains NaN or infinity')
    if len(stamps) < 3 or not np.allclose(np.diff(stamps), 20, rtol=0, atol=1e-6):
        raise ValueError('DIST requires a regular 20-second time axis')
    source_rows = len(data)
    if limit is not None:
        if limit < 3 or limit > source_rows:
            raise ValueError('--limit must be between 3 and the number of source timestamps')
        data, stamps = data[:limit], stamps[:limit]
    epoch_ms = np.rint(stamps * 1000).astype(np.int64)
    if np.max(np.abs(stamps - epoch_ms / 1000)) > 1e-6:
        raise ValueError('Timestamp precision finer than milliseconds is not supported')
    times = pd.to_datetime(epoch_ms, unit='ms', utc=True).tz_convert(timezone).tz_localize(None)
    dates = times.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3].to_numpy()
    expected = data[:, [mapping[name] for name in OUTPUT_NAMES]]
    n = len(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open('w', encoding='utf-8', newline='') as f:
        for j, name in enumerate(OUTPUT_NAMES):
            pd.DataFrame({'date': dates, 'data': expected[:, j], 'cols': name}).to_csv(
                f, index=False, header=j == 0, float_format='%.17g', lineterminator='\n')

    # Check every cell with a round-trip parser, including all variable time blocks.
    offset = 0
    for chunk in pd.read_csv(partial, chunksize=100000, float_precision='round_trip'):
        if list(chunk.columns) != ['date', 'data', 'cols']:
            raise ValueError('Invalid CSV header')
        pos = 0
        while pos < len(chunk):
            j, local = divmod(offset, n)
            count = min(len(chunk) - pos, n - local)
            part = chunk.iloc[pos:pos + count]
            if not (part.cols == OUTPUT_NAMES[j]).all():
                raise ValueError('Incorrect variable block order')
            if not np.array_equal(part.date.to_numpy(), dates[local:local + count]):
                raise ValueError('Timestamp round trip failed')
            if not np.array_equal(part.data.to_numpy(), expected[local:local + count, j]):
                raise ValueError('Numeric round trip failed')
            offset += count
            pos += count
    if offset != n * 18:
        raise ValueError('Incorrect long-table row count')
    restored = read_data(str(partial))
    if list(restored.columns) != list(OUTPUT_NAMES) or restored.shape != (n, 18):
        raise ValueError('Repository reader returned incorrect columns or shape')
    if not restored.index.equals(pd.DatetimeIndex(times, name='date')):
        raise ValueError('Repository reader returned incorrect timestamp index')
    np.testing.assert_allclose(restored.to_numpy(), expected, rtol=1e-14, atol=1e-12)
    target, exog = split_channel(restored, TARGET_CHANNEL)
    if list(target.columns) != list(TARGET_NAMES) or list(exog.columns) != list(EXOG_NAMES):
        raise ValueError('Target/exogenous split failed')
    partial.rename(output)
    meta = output.parent / 'FORECAST_META.csv'
    if not meta.exists():
        # An existing metadata index is preserved; DataSource registers unknown CSVs.
        pd.DataFrame([load_series_info(str(output))]).to_csv(meta, index=False)
    report = dict(
        source_rows=source_rows, timestamps=n, selected_variable_count=18,
        source_sha256=sha256(source), output_sha256=sha256(output), output_bytes=output.stat().st_size,
        long_shape=[n * 18, 3], restored_shape=[n, 18], target_channel=TARGET_CHANNEL,
        target_variables=list(TARGET_NAMES), auxiliary_variables=list(EXOG_NAMES),
        excluded_variables=[name for name, i in sorted(mapping.items(), key=lambda item: item[1]) if name not in OUTPUT_NAMES],
        variable_mapping=[dict(source_index=mapping[name], output_index=j, variable=name,
                               role='target' if j >= 9 else 'auxiliary_input') for j, name in enumerate(OUTPUT_NAMES)],
        sample_interval_seconds=20, timezone_assumption=timezone,
        first_date=dates[0], last_date=dates[-1], pre_normalization=False, imputation=False,
        resampling=False, limit=limit, infer_use_future=0,
        checks=dict(exact_numeric_round_trip=True, full_repository_reader=True,
                    all_time_blocks_equal=True, nine_targets_nine_auxiliaries=True),
        original_reader_max_absolute_error=float(np.max(np.abs(restored.to_numpy() - expected))))
    output.with_suffix('.validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'dataset/forecasting/DIST_9x9.csv')
    p.add_argument('--timezone', default='Asia/Shanghai')
    p.add_argument('--limit', type=int, help='Optional chronological prefix for a small training smoke run')
    args = p.parse_args()
    report = convert(args.input, args.output, args.timezone, args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
