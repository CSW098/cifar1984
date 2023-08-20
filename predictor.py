import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
classes=('Plane','Car  ','Bird ','Cat  ','Deer  ','Dog  ', 'Frog ', 'Horse', 'Ship ','Truck')
def predictor(model= None,
              folder: str ="C:\\Users\\Admin\\Desktop\\sTUDY\\2023-07 ML HTML\\testerimg\\",
              img_type: str='path',
              img_path: str = None,
              img_tensor: torch.tensor = None,
              transform: torchvision.transforms = None,
              transf_tf: bool=True,
              label : str = None):
    if img_path is not None and img_tensor is not None:
        return print("Error: Got 2 imput")
    if img_path==img_tensor is None:
        return print("No input")
    if img_path is not None:
        rrat = Image.open(folder+img_path)
        width,height=rrat.size
        mi=min(rrat.size)
        rrat=rrat.resize((round(32*width/mi),round(32*height/mi)))
        mi2=min(rrat.size)
        rrat_sani=rrat.crop(( (rrat.size[0]-mi2)/2, (rrat.size[1]-mi2)/2, 
                 (rrat.size[0]-mi2)/2+32,(rrat.size[1]-mi2)/2+32 ))
        ttt=transforms.ToTensor()
        final=ttt(rrat_sani).unsqueeze(0)
    if img_tensor is not None:
        assert img_tensor.shape == (1,3,32,32), "Error: Tensor shape should be [1,3,32,32]"
        final=img_tensor
        
    mean1, std1 =(final.mean((2,3))),(final.std((2,3)))
    TTT=transforms.Normalize((mean1[0][0],mean1[0][1],mean1[0][2]),(std1[0][0],std1[0][1],std1[0][2]))     
    if transf_tf == True: final=TTT(final)  
    out=model(final) 
    out_soft=F.softmax(out, dim=1)
    
    prediction={}
    sortt=[]
    
    for i in range(10):
        prediction[classes[i]]=round(out_soft[0][i].item()*100,2)
    #print(prediction)    
    for k, v in sorted(prediction.items(), key=lambda item: item[1], reverse=True)[0:3]:
        sortt.append(k)
        sortt.append(v)   
    return sortt 