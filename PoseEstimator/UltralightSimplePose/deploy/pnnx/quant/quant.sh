# https://github.com/Tencent/ncnn/wiki/quantized-int8-inference
model_name=pose_torchscipt.ncnn
quant_flag=_int8

/home/zg/wdir/zg/moyu/GestureDet/LightGestureDet/tools/deploy_ncnn/ncnn2table ../$model_name.param ../$model_name.bin ./imagelist.txt $model_name.table mean=[103.53,116.54,122.4] norm=[0.0039,0.0039,0.0039] shape=[256,192,3] pixel=BGR thread=8 method=kl
/home/zg/wdir/zg/moyu/GestureDet/LightGestureDet/tools/deploy_ncnn/ncnn2int8 ../$model_name.param ../$model_name.bin $model_name$quant_flag.param $model_name$quant_flag.bin $model_name.table