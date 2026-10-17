from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest

import numpy as np
import pandas as pd
import torch

from ts_benchmark.data.irregular import (load_long_table, ObservationTable, TimeRecords,
    split_records, normalization_from_seen, pack_patches, IrregularCollator)
from ts_benchmark.evaluation.irregular import error_sums, variable_mean, masked_mse, evaluate_irregular
from ts_benchmark.baselines.dag.models.IrregularPatchDelayModel import IrregularPatchDelayModel
from ts_benchmark.baselines.dag.models.PatchDelayModel import PatchDelayModel

torch.set_num_threads(2)


def config():
    return SimpleNamespace(series_dim=1, enc_in=2, d_model=8, d_ff=16, te_dim=3,
        patch_len=2, stride=2, dropout=0., use_delay=True, delay_context=2)


def examples():
    a = (0, np.array([0., 1., 3., 4., 5.]),
         np.array([[0, 2], [3, 0], [4, 5], [6, 7], [8, 9]], np.float32),
         np.array([[1, 1], [1, 0], [1, 1], [1, 1], [1, 1]], bool))
    b = (1, np.array([.5, 2., 4.5]), np.array([[1, 0], [0, 4], [5, 6]], np.float32),
         np.array([[1, 0], [0, 1], [1, 1]], bool))
    return [a, b]


class IrregularTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_long_schema_duplicates_roles_and_observed_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.csv'
            pd.DataFrame({'date':[2,0,1,1,0], 'data':[8,0,2,3,4],
                          'cols':['Y','Y','X','X','X']}).to_csv(path,index=False)
            table = load_long_table(path, ['Y'], ['X'])
            self.assertEqual(table.names, ('Y','X'))
            self.assertEqual(table.duplicate_rows, 1)
            np.testing.assert_array_equal(table.time, [0,1,2])
            np.testing.assert_array_equal(table.values, [[0,4],[0,2.5],[8,0]])
            self.assertTrue(table.mask[0,0])
            self.assertFalse(table.mask[1,0])
            native = pd.read_csv(path).rename(columns={'date':'time','data':'value','cols':'variable'})
            native.to_csv(path,index=False)
            other = load_long_table(path,['Y'],['X'])
            np.testing.assert_array_equal(table.mask, other.mask)
            np.testing.assert_array_equal(table.values, other.values)

    def test_datetime_conversion_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.csv'
            f=pd.DataFrame({'date':['2023-07-01 00:00:11.215','2023-07-01 00:00:31.215']*2,
                            'data':[1,2,3,4], 'cols':['Y','Y','X','X']})
            f.to_csv(path,index=False)
            table=load_long_table(path,['Y'],['X'])
            np.testing.assert_array_equal(table.time,[0,20])
            f['data']=f['data'].astype(float)
            f.loc[0,'data']=np.inf;f.to_csv(path,index=False)
            with self.assertRaises(ValueError):load_long_table(path,['Y'],['X'])

    def test_disjoint_record_split_and_seen_normalization(self):
        from sklearn.model_selection import train_test_split
        time=np.arange(121,dtype=np.float64)
        values=np.stack([time,time+2],axis=1).astype(np.float32)
        records=TimeRecords(ObservationTable(time,values,np.ones_like(values,bool),('Y','X'),0),4,2,1)
        splits=split_records(records)
        seen,test=train_test_split(np.arange(20),train_size=.8,random_state=42,shuffle=True)
        train,val=train_test_split(seen,train_size=.75,random_state=42,shuffle=False)
        self.assertEqual(splits,{'train':train.tolist(),'val':val.tolist(),'test':test.tolist()})
        self.assertEqual(len(set(sum(splits.values(),[]))),len(records))
        norm=normalization_from_seen(records,splits['train']+splits['val'])
        expected=np.concatenate([records[i][2] for i in list(train)+list(val)])
        np.testing.assert_array_equal(norm['data_min'],expected.min(0))
        np.testing.assert_array_equal(norm['data_max'],expected.max(0))

    def test_elapsed_time_patches_keep_empty_slots(self):
        batch=examples()
        data,times,mask=pack_patches(batch,0,4,2,range(2))
        self.assertEqual(data.shape,(2,2,2,2))
        self.assertEqual(times[0,1,0,0],3)
        self.assertFalse(mask[1,0,:,1].any())
        # 3 seconds falls in physical patch 1, regardless of event-list length.
        self.assertEqual(data[0,1,0,0],4)
        self.assertTrue(mask[0,0,0,0])  # a real zero is observed

    def test_macro_metrics_not_pooled_and_ignore_hidden_nan(self):
        pred=torch.tensor([[[1.,3.],[1.,float('nan')],[1.,float('nan')]]])
        target=torch.zeros_like(pred)
        mask=torch.tensor([[[1,1],[1,0],[1,0]]],dtype=torch.bool)
        self.assertAlmostEqual(float(masked_mse(pred,target,mask)),5.,places=6)
        self.assertNotAlmostEqual(float(masked_mse(pred,target,mask)),3.)
        self.assertIsNone(masked_mse(pred,target,torch.zeros_like(mask)))

    def test_packed_forward_gradient_padding_and_no_label_leakage(self):
        norm={'data_min':[0,0],'data_max':[10,10],'time_max':6}
        batch=IrregularCollator(4,2,2,1,norm)(examples())
        model=IrregularPatchDelayModel(config()).eval()
        pred,_=model(batch)
        self.assertEqual(tuple(pred.shape),(2,2,1))
        changed={k:v.clone() for k,v in batch.items()}
        changed['data_to_predict'].fill_(12345)
        for prefix in ['observed','future_exog']:
            value='observed_data' if prefix=='observed' else 'future_exog'
            mask='observed_mask' if prefix=='observed' else 'future_exog_mask'
            changed[value][~changed[mask]]=float('nan')
        torch.testing.assert_close(pred,model(changed)[0],rtol=0,atol=0)
        loss=masked_mse(pred,batch['data_to_predict'],batch['mask_predicted_data'])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0)

    def test_history_subsampling_preserves_future_labels_and_x(self):
        norm={'data_min':[0,0],'data_max':[10,10],'time_max':6}
        full=IrregularCollator(4,2,2,1,norm)(examples())
        dropped=IrregularCollator(4,2,2,1,norm,keep_rate=.5)(examples())
        for k in ['future_exog','future_exog_mask','data_to_predict','mask_predicted_data','tp_to_predict']:
            torch.testing.assert_close(full[k],dropped[k])
        self.assertLess(int(dropped['observed_mask'].sum()),int(full['observed_mask'].sum()))

    def test_query_mapping_is_per_record_and_empty_inputs_are_finite(self):
        model=IrregularPatchDelayModel(config()).eval()
        tokens=torch.arange(2*3*4*8).reshape(2,3,4,8)
        indices=torch.tensor([[0,2,3],[3,1,0]])
        selected=model._select_query_patches(tokens,indices)
        for b in range(2):
            torch.testing.assert_close(selected[b],tokens[b,:,indices[b]])
        norm={'data_min':[0,0],'data_max':[10,10],'time_max':6}
        batch=IrregularCollator(4,2,2,1,norm)(examples())
        batch['observed_mask'].zero_();batch['future_exog_mask'].zero_()
        batch['observed_data'].fill_(float('nan'))
        batch['future_exog'].fill_(float('nan'))
        self.assertTrue(torch.isfinite(model(batch)[0]).all())

    def test_global_evaluation_independent_of_batch_partition(self):
        class Echo(torch.nn.Module):
            def forward(self,b):return b['prediction'],torch.tensor(0.)
        pred=torch.tensor([[[1.,3.],[1.,0.]],[[2.,4.],[2.,0.]],[[5.,9.],[5.,0.]]])
        mask=torch.tensor([[[1,1],[1,0]]]*3,dtype=torch.bool)
        all_batch={'prediction':pred,'data_to_predict':torch.ones_like(pred),'mask_predicted_data':mask}
        split=[{k:v[:1] for k,v in all_batch.items()},{k:v[1:] for k,v in all_batch.items()}]
        one=evaluate_irregular(Echo(),[all_batch],'cpu')
        two=evaluate_irregular(Echo(),split,'cpu')
        self.assertEqual(one,two)

    def test_regular_prediction_matches_pre_integration_code(self):
        root=Path(__file__).resolve().parents[1]
        base='36b08de77967bee1145c324b146ff031778bdcad'
        try:
            exo=subprocess.check_output(['git','show',base+':ts_benchmark/baselines/dag/models/PatchExoModel.py'],cwd=root,text=True)
            delay=subprocess.check_output(['git','show',base+':ts_benchmark/baselines/dag/models/PatchDelayModel.py'],cwd=root,text=True)
        except subprocess.CalledProcessError:
            self.skipTest('Original reference commit unavailable in this checkout')
        module=ModuleType('_regular_reference_exo');exec(exo,module.__dict__)
        sys.modules[module.__name__]=module
        delay=delay.replace('from ts_benchmark.baselines.dag.models.PatchExoModel import PatchExoModel',
                            'from _regular_reference_exo import PatchExoModel')
        namespace={};exec(delay,namespace)
        for dim in [1,9]:
            settings=config();settings.series_dim=dim;settings.enc_in=2*dim
            old=namespace['PatchDelayModel'](settings).eval()
            new=PatchDelayModel(settings).eval();new.load_state_dict(old.state_dict())
            x=torch.randn(2,6,2*dim);xf=torch.randn(2,5,dim)
            torch.testing.assert_close(old(x,xf)[0],new(x,xf)[0],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
