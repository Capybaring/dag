"""Compare this implementation with a user-supplied official checkout.

Usage: python tests/check_tpatchgnn_parity.py --reference /path/to/t-PatchGNN
Only named pure functions are loaded; no upstream dataset downloads/imports.
"""
import argparse
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ts_benchmark.data.irregular import IrregularCollator
from ts_benchmark.evaluation.irregular import error_sums, variable_mean, evaluate_irregular


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != set(names):
        raise ValueError('Missing expected official reference functions')
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    ns={'torch':torch,'np':np,'pad_sequence':pad_sequence,'get_device':lambda t:t.device}
    utils=SimpleNamespace(get_next_batch=lambda it:next(it))
    ns['utils']=utils
    load_functions(args.reference/'lib/utils.py',
        ['normalize_masked_data','normalize_masked_tp','split_and_patch_batch'],ns)
    for name in ['normalize_masked_data','normalize_masked_tp','split_and_patch_batch']:
        setattr(utils,name,ns[name])
    load_functions(args.reference/'lib/physionet.py',['patch_variable_time_collate_fn'],ns)
    load_functions(args.reference/'lib/evaluation.py',['compute_error','evaluation'],ns)

    examples=[(0,np.array([0.,1.,3.,4.,5.]),np.array([[0,2],[3,0],[4,5],[6,7],[8,9]],np.float32),
               np.array([[1,1],[1,0],[1,1],[1,1],[1,1]],bool)),
              (1,np.array([.5,2.,4.5]),np.array([[1,0],[0,4],[5,6]],np.float32),
               np.array([[1,0],[0,1],[1,1]],bool))]
    norm={'data_min':[0.,0.],'data_max':[10.,10.],'time_max':6.}
    own=IrregularCollator(4,2,2,1,norm)(examples)
    official_records=[(rid,torch.tensor(t,dtype=torch.float32),torch.tensor(v),torch.tensor(m,dtype=torch.float32)) for rid,t,v,m in examples]
    ref=ns['patch_variable_time_collate_fn'](official_records,
        SimpleNamespace(history=4,patch_size=2,stride=2,npatch=2,dataset='physionet'),
        data_min=torch.tensor(norm['data_min']),data_max=torch.tensor(norm['data_max']),time_max=torch.tensor(6.))
    for name in ['observed_data','observed_tp','observed_mask','tp_to_predict']:
        torch.testing.assert_close(own[name].float(),ref[name].float(),rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(own['data_to_predict'],ref['data_to_predict'][...,:1],rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(own['mask_predicted_data'].float(),ref['mask_predicted_data'][...,:1])

    pred=torch.tensor([[[.2,.6],[.3,.5],[.9,.4]],[[.6,.7],[.1,.2],[.3,.5]]])
    truth=torch.tensor([[[0.,.3],[.4,.2],[.3,.1]],[[.7,.2],[0.,.8],[.4,.9]]])
    mask=torch.tensor([[[1,1],[1,0],[0,1]],[[1,1],[0,1],[1,0]]],dtype=torch.float32)
    for kind in ['mse','mae','mape']:
        sums,counts=error_sums(pred,truth,mask,kind)
        rs,rc=ns['compute_error'](truth,pred,mask,kind.upper(),'sum')
        torch.testing.assert_close(sums,rs,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(counts.float(),rc)
        torch.testing.assert_close(variable_mean(sums,counts),ns['compute_error'](truth,pred,mask,kind.upper(),'mean'))

    class Echo(torch.nn.Module):
        def forward(self,b):return b['prediction'],torch.tensor(0.)
    batches=[{'prediction':pred[:1],'data_to_predict':truth[:1],'mask_predicted_data':mask[:1],
              'tp_to_predict':pred[:1],'observed_data':pred[:1],'observed_tp':pred[:1],'observed_mask':mask[:1]},
             {'prediction':pred[1:],'data_to_predict':truth[1:],'mask_predicted_data':mask[1:],
              'tp_to_predict':pred[1:],'observed_data':pred[1:],'observed_tp':pred[1:],'observed_mask':mask[1:]}]
    official_model=SimpleNamespace(forecasting=lambda queries,*unused:queries)
    official=ns['evaluation'](official_model,iter(batches),2)
    own_metrics=evaluate_irregular(Echo(),batches,'cpu')
    for name in ['mse','mae','rmse','mape','loss']:
        np.testing.assert_allclose(own_metrics[name],official[name],rtol=1e-6,atol=1e-7)
    print('PASS: official time patch packing, min-max/time normalization, target padding, macro training loss and global evaluation metrics')


if __name__=='__main__':main()
