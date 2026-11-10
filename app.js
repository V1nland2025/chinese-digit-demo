'use strict';
const $ = id => document.getElementById(id);
const names = ['一','二','三','四','五','六','七','八','九','十'];
const source = $('source'), scan = $('scan'), input = $('input'), video = $('video');
let stream = null, session = null, hasImage = false, imageVersion = 0;
const baseFrame=document.createElement('canvas'); baseFrame.width=baseFrame.height=512;
const status = text => $('status').textContent = text;
function stopCamera() { if (stream) stream.getTracks().forEach(t => t.stop()); stream = null; video.srcObject = null; $('capture').disabled = true; $('cameraHint').style.display = 'grid'; }
window.addEventListener('pagehide', stopCamera);
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) { status('无法打开摄像头：请使用支持摄像头的浏览器，并通过可信 HTTPS 访问。也可以选择照片。'); return; }
  try {
    stopCamera();
    stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});
    video.srcObject = stream; await video.play(); $('cameraHint').style.display = 'none'; $('capture').disabled = false; status('摄像头已开启，请将完整字符放入方形画面。');
  } catch (e) { stopCamera(); status('摄像头无法使用：请检查权限、HTTPS 和是否被其他应用占用。' + e.message); }
}
// 视频和预览均为居中 cover：从原始帧中心截取与可见方形对应的区域。
function cropSquare(image, width, height) {
  if (!width || !height) throw new Error('图像尚未准备好');
  const side = Math.min(width, height), x = (width-side)/2, y = (height-side)/2;
  const ctx = source.getContext('2d'); ctx.clearRect(0,0,512,512); ctx.drawImage(image,x,y,side,side,0,0,512,512);
  rememberFrame(); $('predict').disabled = !session;
}
function fitSquare(image, width, height) {
  if (!width || !height) throw new Error('图像尚未准备好');
  const scale=512/Math.max(width,height), w=width*scale, h=height*scale;
  const ctx=source.getContext('2d');ctx.clearRect(0,0,512,512);ctx.drawImage(image,(512-w)/2,(512-h)/2,w,h);
  rememberFrame();$('predict').disabled=!session;
}
function rememberFrame() {
  const ctx=baseFrame.getContext('2d');ctx.clearRect(0,0,512,512);ctx.drawImage(source,0,0);
  hasImage=true;resetFrame();
}
function resetFrame() {
  $('zoom').value='1';$('positionX').value='50';$('positionY').value='50';applyFrame();
}
function applyFrame() {
  $('zoomValue').textContent=Number($('zoom').value).toFixed(2);$('positionXValue').textContent=$('positionX').value;$('positionYValue').textContent=$('positionY').value;
  if(!hasImage)return;
  const size=512/Number($('zoom').value), remaining=512-size;
  const x=remaining*Number($('positionX').value)/100, y=remaining*Number($('positionY').value)/100;
  const ctx=source.getContext('2d');ctx.clearRect(0,0,512,512);ctx.drawImage(baseFrame,x,y,size,size,0,0,512,512);
  processImage();
}
function processImage() {
  if (!hasImage) return;
  const w=512, h=512, ctx=scan.getContext('2d'), pixels=source.getContext('2d').getImageData(0,0,w,h);
  const gray=new Uint8Array(w*h), integral=new Float64Array((w+1)*(h+1));
  const blackBackground=$('polarity').value === 'black';
  for(let y=0;y<h;y++){ let row=0; for(let x=0;x<w;x++){
    const i=y*w+x, p=i*4, alpha=pixels.data[p+3]/255, raw=.299*pixels.data[p]+.587*pixels.data[p+1]+.114*pixels.data[p+2], v=Math.round(raw*alpha+(blackBackground?0:255)*(1-alpha));
    gray[i]=blackBackground?255-v:v; row+=gray[i]; integral[(y+1)*(w+1)+x+1]=integral[y*(w+1)+x+1]+row;
  }}
  const enhanced=$('mode').value === 'scan', stride=w+1, radius=20;
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    const i=y*w+x, p=i*4; let v=gray[i];
    if(enhanced){ const x0=Math.max(0,x-radius), x1=Math.min(w,x+radius+1), y0=Math.max(0,y-radius), y1=Math.min(h,y+radius+1);
      const sum=integral[y1*stride+x1]-integral[y0*stride+x1]-integral[y1*stride+x0]+integral[y0*stride+x0]; v=v < sum/((x1-x0)*(y1-y0))-12 ? 0 : 255; }
    pixels.data[p]=pixels.data[p+1]=pixels.data[p+2]=v; pixels.data[p+3]=255;
  }
  ctx.putImageData(pixels,0,0);
  const ic=input.getContext('2d'); ic.imageSmoothingEnabled=true; ic.imageSmoothingQuality='high'; ic.drawImage(scan,0,0,32,32);
  const actual=ic.getImageData(0,0,32,32); for(let i=0;i<actual.data.length;i+=4){const v=255-actual.data[i];actual.data[i]=actual.data[i+1]=actual.data[i+2]=v;actual.data[i+3]=255;} ic.putImageData(actual,0,0);
  imageVersion++; $('result').textContent='图像已就绪'; $('scores').textContent='';
}
async function predict() {
  if(!session||!hasImage) return;
  $('predict').disabled=true; const version=imageVersion;
  try { const data=input.getContext('2d').getImageData(0,0,32,32).data, values=new Float32Array(1024);
    for(let i=0;i<1024;i++) values[i]=data[i*4]/255;
    const result=await session.run({image:new ort.Tensor('float32',values,[1,1,32,32])}), logits=Array.from(result.logits.data), max=Math.max(...logits);
    const exp=logits.map(x=>Math.exp(x-max)),sum=exp.reduce((a,b)=>a+b,0), order=exp.map((v,i)=>({i,p:v/sum})).sort((a,b)=>b.p-a.p);
    if(version!==imageVersion){status('图片已调整，请重新点击识别。');return;}
    $('result').textContent='识别为：'+names[order[0].i];$('scores').textContent='前三类模型分数：'+order.slice(0,3).map(x=>names[x.i]+' '+(x.p*100).toFixed(1)+'%').join('　');status('识别完成。');
  } catch(e){status('识别失败：'+e.message);} finally{$('predict').disabled=false;}
}
$('start').addEventListener('click',startCamera);
$('capture').addEventListener('click',()=>{try{cropSquare(video,video.videoWidth,video.videoHeight);stopCamera();status('已拍照，可检查处理结果。');}catch(e){status('拍照失败：'+e.message);}});
$('file').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;const url=URL.createObjectURL(file);try{const img=new Image();img.src=url;await img.decode();fitSquare(img,img.naturalWidth,img.naturalHeight);stopCamera();status('照片已等比例补成正方形，请确认字符完整。');}catch(err){status('无法读取照片：'+err.message);}finally{URL.revokeObjectURL(url);e.target.value='';}});
$('mode').addEventListener('change',processImage); $('polarity').addEventListener('change',processImage); $('predict').addEventListener('click',predict);
['zoom','positionX','positionY'].forEach(id=>$(id).addEventListener('input',applyFrame));$('resetFrame').addEventListener('click',resetFrame);
(async()=>{try{if(!window.ort)throw new Error('推理库未加载');ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=new URL('vendor/',document.baseURI).href;session=await ort.InferenceSession.create('model-external.onnx',{executionProviders:['wasm']});$('predict').disabled=!hasImage;status('模型已就绪。请打开摄像头或选择照片。');}catch(e){status('模型加载失败：'+e.message+'。请确认网页、model-external.onnx 和 vendor 文件均已上传。');}})();
