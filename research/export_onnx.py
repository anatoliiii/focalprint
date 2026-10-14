"""Export an FpNet checkpoint to the ONNX the daemon loads
(crates/focalprint-net): input `frame` f32[1,1,80,64], outputs `desc`
f32[1,dim,40,32] (L2-normalised) and `rel` f32[1,1,40,32].

    python export_onnx.py model.pt fpnet.onnx"""
import argparse
import torch
from fpnet import H, W, load

p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('out')
a = p.parse_args()

net = load(a.model).eval()
x = torch.randn(1, 1, H, W)
torch.onnx.export(net, x, a.out, input_names=['frame'], output_names=['desc', 'rel'],
                  opset_version=17, dynamo=False)
print(f'{a.model} {net.config} -> {a.out}')
