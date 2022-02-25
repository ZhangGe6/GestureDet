import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(1, 3, 256, 192)
    out = []

    with ncnn.Net() as net:
         net.load_param("./pose_torchscipt.ncnn.param")
         net.load_model("./pose_torchscipt.ncnn.bin")
         outcount = len(net.output_names())

         with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())

            for i in range(outcount):
                _, outi = ex.extract("out" + str(i))
                out.append(torch.from_numpy(np.array(outi)).unsqueeze(0))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)
