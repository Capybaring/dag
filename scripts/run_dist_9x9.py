"""Launch the checked DIST 9-target/9-OP benchmark configuration."""
import argparse
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-name', default='DIST_9x9.csv')
    p.add_argument('--config', type=Path, default=ROOT / 'config/dist_9x9_config.json')
    p.add_argument('--num-epochs', type=int)
    p.add_argument('--num-rollings', type=int)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not (ROOT / 'dataset/forecasting' / args.data_name).is_file():
        p.error('Dataset is missing; run scripts/prepare_dist_9x9.py first.')
    config = json.loads(args.config.read_text())
    model = config['model_config']['models'][0]
    params = copy.deepcopy(model['model_hyper_params'])
    strategy = config['evaluation_config']['strategy_args']
    if args.num_epochs is not None:
        if args.num_epochs < 1:
            p.error('--num-epochs must be positive')
        params['num_epochs'] = args.num_epochs
    if args.num_rollings is not None:
        if args.num_rollings < 1:
            p.error('--num-rollings must be positive')
        strategy['num_rollings'] = args.num_rollings
    # run_benchmark appends CLI models. Clear the list in a temporary config to
    # prevent running the same model twice while preserving the editable source.
    config['model_config']['models'] = []
    with tempfile.TemporaryDirectory(prefix='dist-9x9-') as folder:
        effective = Path(folder) / 'config.json'
        effective.write_text(json.dumps(config))
        command = [sys.executable, str(ROOT / 'scripts/run_benchmark.py'),
                   '--config-path', str(effective), '--data-name-list', args.data_name,
                   '--model-name', model['model_name'], '--model-hyper-params', json.dumps(params),
                   '--eval-backend', 'sequential', '--num-workers', '1',
                   '--timeout', '60000', '--save-path', 'DIST_9x9/DAG']
        print('Targets: 9 PV channels; auxiliary inputs: 9 OP channels.', flush=True)
        print('20-second steps: history 56 minutes; forecast 8 minutes.', flush=True)
        if args.dry_run:
            print('Effective configuration:', json.dumps(config, ensure_ascii=False))
            print(shlex.join(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
