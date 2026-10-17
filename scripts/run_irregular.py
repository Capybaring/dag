"""Train DAG PatchDelay on long-table IMTS using the t-PatchGNN protocol."""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ts_benchmark.data.irregular import (DIST_TARGETS, DIST_EXOG, load_long_table,
    TimeRecords, split_records, normalization_from_seen, IrregularCollator)
from ts_benchmark.baselines.dag.models.IrregularPatchDelayModel import IrregularPatchDelayModel
from ts_benchmark.evaluation.irregular import masked_mse, evaluate_irregular


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--targets', nargs='+', default=list(DIST_TARGETS))
    parser.add_argument('--exog', nargs='+', default=list(DIST_EXOG))
    parser.add_argument('--history', type=float, default=14400, help='Historical seconds')
    parser.add_argument('--forecast', type=float, default=7200, help='Forecast seconds')
    parser.add_argument('--patch-seconds', type=float, default=960)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--d-ff', type=int, default=128)
    parser.add_argument('--te-dim', type=int, default=10)
    parser.add_argument('--delay-context', type=int, default=4)
    parser.add_argument('--use-delay', type=int, choices=[0, 1], default=1)
    parser.add_argument('--keep-rate', type=float, default=1., help='Historical timepoint retention; labels and future X unchanged')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-records', type=int, default=None, help='Smoke tests only: restrict records before splitting')
    parser.add_argument('--save-path', default='result/DIST_irregular/PatchDelay')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error('epochs, patience, batch-size and threads must be positive')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(args.threads)
    output = Path(args.save_path)
    output.mkdir(parents=True, exist_ok=True)
    print('Loading irregular observations:', args.data_path, flush=True)
    table = load_long_table(args.data_path, args.targets, args.exog)
    records = TimeRecords(table, args.history, args.forecast, len(args.targets))
    full_count = len(records)
    if args.max_records is not None:
        if args.max_records < 5:
            parser.error('max-records must be at least five')
        records.records = records.records[:args.max_records]
    splits = split_records(records)
    norm = normalization_from_seen(records, splits['train']+splits['val'])
    loaders = {}
    for name, indices in splits.items():
        collate = IrregularCollator(args.history, args.forecast, args.patch_seconds,
                                    len(args.targets), norm, args.keep_rate, args.seed)
        loaders[name] = DataLoader(Subset(records, indices), batch_size=args.batch_size,
                                   shuffle=(name == 'train'), num_workers=0,
                                   drop_last=False, collate_fn=collate,
                                   generator=torch.Generator().manual_seed(args.seed))
    config = SimpleNamespace(series_dim=len(args.targets), enc_in=len(table.names),
        d_model=args.d_model, d_ff=args.d_ff, te_dim=args.te_dim, dropout=0.,
        patch_len=1, stride=1, use_delay=bool(args.use_delay), delay_context=args.delay_context)
    device = torch.device(args.device)
    model = IrregularPatchDelayModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    record_ids = {name: [records.records[i][0] for i in ids] for name, ids in splits.items()}
    manifest = {'args': vars(args), 'model_config': vars(config), 'normalization': norm,
        'protocol': 'tpatchgnn-record-60-20-20-seed42-seen-minmax-macro-mse',
        'future_exogenous': 'known future X; original missing X remains masked',
        'normalization_scope': 'train+validation (official seen_data)',
        'record_ids': record_ids, 'records_before_smoke_limit': full_count,
        'skipped_empty_records': records.skipped, 'duplicates_merged': table.duplicate_rows,
        'table_shape': list(table.values.shape), 'variable_order_y_then_x': list(table.names),
        'incomplete_tail_seconds': float(table.time[-1] % records.duration),
        'smoke_test': args.max_records is not None}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Records:', {k: len(v) for k, v in splits.items()}, 'shape:', table.values.shape,
          'duplicate cells merged:', table.duplicate_rows, flush=True)
    best, best_epoch, best_state, stale, history = float('inf'), 0, None, 0, []
    for epoch in range(args.epochs):
        model.train();start = time.perf_counter();last_loss = None
        for index, batch in enumerate(loaders['train']):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            prediction, auxiliary = model(batch)
            loss = masked_mse(prediction, batch['data_to_predict'], batch['mask_predicted_data'])
            if loss is None:
                continue
            total = loss+auxiliary
            if not torch.isfinite(total):
                raise ValueError('Nonfinite training loss')
            total.backward();optimizer.step();last_loss = float(total.detach())
            if index == 0 or (index+1) % 50 == 0:
                print(f'Epoch {epoch+1} batch {index+1}/{len(loaders["train"])} MSE={last_loss:.6f}', flush=True)
        if last_loss is None:
            raise ValueError('No observed training targets')
        # A subsampling experiment must validate on the same retained history
        # each epoch; only the training collator resamples across epochs.
        loaders['val'].collate_fn.rng = np.random.RandomState(args.seed)
        val = evaluate_irregular(model, loaders['val'], device)
        history.append({'epoch': epoch+1, 'last_train_batch_mse': last_loss, 'val': val})
        print(f'Epoch {epoch+1} validation MSE={val["mse"]:.6f} MAE={val["mae"]:.6f} seconds={time.perf_counter()-start:.1f}', flush=True)
        if val['mse'] < best:
            best, best_epoch, stale = val['mse'], epoch+1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'state_dict': best_state, 'manifest': manifest, 'best_epoch': best_epoch,
                        'best_validation_mse': best}, output/'best_model.pt')
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    # Make optional subsampling reproducible for final best-checkpoint evaluation.
    loaders['test'].collate_fn.rng = np.random.RandomState(args.seed)
    test = evaluate_irregular(model, loaders['test'], device)
    result = {'best_epoch': best_epoch, 'best_validation_mse': best, 'test': test,
              'training_history': history, 'smoke_test': args.max_records is not None}
    (output/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('Test:', json.dumps(test), 'best epoch:', best_epoch, flush=True)
    return result


if __name__ == '__main__':
    main()
