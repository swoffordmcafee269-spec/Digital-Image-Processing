import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from canon_enhance.common import PROJECT_ROOT,safe_output
from canon_enhance.data import pack_raw,unpack_raw
from canon_enhance.model import RawToRGB
from canon_enhance.metrics import ssim_torch
from canon_enhance.prepare import split_groups
from canon_enhance.inference import enhance
from canon_enhance.ema import ModelEMA
from canon_enhance.batch import enhance_directory


CAL={"matrix":[[1,0,0,0],[0,.5,.5,0],[0,0,0,1]],"bias":[0,0,0]}


class CoreTests(unittest.TestCase):
    def test_bayer_pack_preserves_phase_and_dynamic_range(self):
        raw=(np.arange(80*64).reshape(64,80)+512).astype(np.uint16)
        packed=pack_raw(raw,512,100)
        np.testing.assert_allclose(unpack_raw(packed),(raw.astype(np.float32)-512)/100)
        self.assertGreater(packed.max(),1)

    def test_ssim_matches_skimage(self):
        rng=np.random.default_rng(12)
        for a,b in [(rng.random((48,64,3),dtype=np.float32),rng.random((48,64,3),dtype=np.float32)),
                    (np.full((48,64,3),.3,dtype=np.float32),np.full((48,64,3),.32,dtype=np.float32))]:
            expected=structural_similarity(a,b,channel_axis=2,data_range=1.0)
            actual=ssim_torch(torch.from_numpy(a.transpose(2,0,1)).unsqueeze(0),torch.from_numpy(b.transpose(2,0,1)).unsqueeze(0)).item()
            self.assertAlmostEqual(float(expected),actual,delta=5e-5)

    def test_no_spatial_normalization_and_correct_color(self):
        model=RawToRGB(CAL,base_channels=8,levels=2).eval()
        self.assertFalse(any(isinstance(m,(torch.nn.InstanceNorm2d,torch.nn.BatchNorm2d)) for m in model.modules()))
        x=torch.zeros(1,4,32,40);x[:,0]=.5
        with torch.no_grad():rgb=model(x)
        self.assertEqual(tuple(rgb.shape),(1,3,64,80))
        self.assertGreater(rgb[:,0].mean().item(),.7)
        self.assertLess(rgb[:,1:].abs().max().item(),1e-6)

    def test_groups_do_not_cross_split(self):
        groups=[list(range(i,i+10)) for i in range(0,1000,10)]
        split=split_groups(groups,1000,20261021)
        self.assertEqual([split.count(s) for s in ['train','val','test']],[800,100,100])
        for group in groups:self.assertEqual(len({split[i] for i in group}),1)

    def test_inference_works_without_gt(self):
        tmp_root=safe_output(PROJECT_ROOT/'artifacts'/'test_tmp');tmp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            path=Path(tmp);raw=np.full((64,80),512,dtype='<u2');raw[::2,::2]=562;raw.tofile(path/'input.raw')
            model=RawToRGB(CAL,base_channels=8,levels=2)
            torch.save({'model':model.state_dict(),'calibration':CAL,'model_config':{'base_channels':8,'levels':2},
                        'input_spec':{'height':64,'width':80,'black_level':512,'scale':100}},path/'model.pt')
            rgb=enhance(path/'input.raw',path/'model.pt',device='cpu')
            self.assertEqual(rgb.shape,(64,80,3));self.assertEqual(rgb.dtype,np.uint8)
            self.assertGreater(float(rgb[...,0].mean()),180);self.assertEqual(int(rgb[...,1:].max()),0)

    def test_output_cannot_point_to_source_dataset(self):
        with self.assertRaises(ValueError):safe_output('D:/BaiduNetdiskDownload/BIED/overwrite.png')

    def test_adaptive_proxy_handles_large_raw_values(self):
        model=RawToRGB(CAL,base_channels=8,levels=2,adaptive_proxy=True).eval()
        x=torch.full((1,4,32,40),200.0)
        with torch.no_grad():rgb=model(x)
        self.assertTrue(torch.isfinite(rgb).all())
        self.assertGreaterEqual(rgb.min().item(),0)
        self.assertLessEqual(rgb.max().item(),1)

    def test_ema_does_not_modify_calibration_and_can_restore_live_weights(self):
        model=RawToRGB(CAL,base_channels=8,levels=2)
        ema=ModelEMA(model,decay=.5)
        with torch.no_grad():model.head.bias.fill_(2)
        ema.update(model)
        backup=ema.apply(model)
        torch.testing.assert_close(model.head.bias,torch.ones_like(model.head.bias))
        torch.testing.assert_close(model.camera_matrix,torch.tensor(CAL['matrix'],dtype=torch.float32))
        model.load_state_dict(backup)
        torch.testing.assert_close(model.head.bias,torch.full_like(model.head.bias,2))

    def test_batch_without_gt_resumes_and_rejects_changed_checkpoint(self):
        tmp_root=safe_output(PROJECT_ROOT/'artifacts'/'test_tmp');tmp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            path=Path(tmp);source=path/'raw';source.mkdir()
            np.full((32,40),540,dtype='<u2').tofile(source/'1.raw')
            model=RawToRGB(CAL,base_channels=8,levels=2)
            checkpoint={'model':model.state_dict(),'calibration':CAL,'model_config':{'base_channels':8,'levels':2},
                        'input_spec':{'height':32,'width':40,'black_level':512,'scale':100}}
            torch.save(checkpoint,path/'model.pt')
            first=enhance_directory(source,path/'model.pt',path/'out',device='cpu')
            second=enhance_directory(source,path/'model.pt',path/'out',device='cpu',resume=True)
            self.assertEqual(first['written'],1);self.assertEqual(second['skipped'],1)
            checkpoint['changed']=True;torch.save(checkpoint,path/'model.pt')
            with self.assertRaises(ValueError):
                enhance_directory(source,path/'model.pt',path/'out',device='cpu',resume=True)


if __name__=='__main__':unittest.main()
