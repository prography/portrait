import torch
from torch import onnx
import onnx
from onnx_caffe2.backend import Caffe2Backend

from model.nst import TransferNet

def convert_to_pb(pytorch_modelroot):
    model_name = pytorch_modelroot.replace('.pth', '')
    onnx_outpath = model_name + '_onnx.onnx.pb'
    pb_init_outpath = model_name + '_init.pb'
    pb_pred_outpath = model_name + '_pred.pb'

    # loading pytorch model
    net = TransferNet()
    net.load_state_dict(torch.load(pytorch_modelroot, map_location=lambda storage, loc: storage))
    net.train(False)  # fix model's parameters
    print(net)

    # convert to onnx.pb file
    dummy_input = torch.randn(1, 3, 300, 300, requires_grad=True) # create dummy input for setting graph structure
    onnx.export(net, dummy_input, onnx_outpath, verbose=True, export_params=True) # export to ONNX model
    print("[*] Convert to ONNX completed!")

    onnx_model = onnx.load(onnx_outpath)
    onnx.checker.check_model(onnx_model)
    init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model.grath)

    with open(pb_init_outpath, 'wb') as f:
        f.write(init_net.SerializeToString())
    print("[*] Conver to pb initialize net completed!")

    with open(pb_pred_outpath, 'wb') as f:
        f.write(predict_net.SerializeToString())
    print("[*] Conver to pb prediction net completed!")

if __name__ == '__main__':
    convert_to_pb('weights/style_water.pth')