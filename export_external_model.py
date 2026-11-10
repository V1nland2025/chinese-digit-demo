"""导出公开数据训练权重并比较模型在同一输入上的输出。"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.mobile_deps'))
import numpy as np
from PIL import Image
import onnx,onnxruntime as ort,torch
from experiment import DATA,LeNetLike

HERE=Path(__file__).resolve().parent
weight=ROOT/'results'/'chinese_mnist_grouped'/'model.pt';dest=HERE/'model-external.onnx'
if dest.exists():raise FileExistsError(f'为避免覆盖已有文件：{dest}')
model=LeNetLike().eval();model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True)['state_dict'])
torch.onnx.export(model,torch.zeros(1,1,32,32),str(dest),input_names=['image'],output_names=['logits'],opset_version=17,dynamo=False)
onnx.checker.check_model(onnx.load(str(dest)))
opt=ort.SessionOptions();opt.intra_op_num_threads=1
sess=ort.InferenceSession(str(dest),sess_options=opt,providers=['CPUExecutionProvider'])
rows=json.loads((ROOT/'results'/'full_data_200'/'training_records.json').read_text(encoding='utf8'))
diff=[];maxerr=0.
for row in rows:
    with Image.open(DATA/'images'/f"image_{row['index']}.png") as im:
        x=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)[None,None]/255
    with torch.no_grad():a=model(torch.from_numpy(x)).numpy()
    b=sess.run(None,{'image':x})[0];maxerr=max(maxerr,float(np.max(np.abs(a-b))))
    if a.argmax()!=b.argmax():diff.append(row['index'])
report={'samples':len(rows),'max_logit_error':maxerr,'different_predictions':diff,
        'weight_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'onnx_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
        'note':'使用课程图片核对模型转换，不作为泛化性能测试。'}
(HERE/'公开数据模型转换核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False,indent=2))
if len(rows)!=200 or diff or maxerr>1e-3:raise SystemExit('模型转换核验失败')
