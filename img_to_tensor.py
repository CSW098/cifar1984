import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
from PIL import Image

def img_to_tensor(img_obj = None,
                  folder: str ="C:\\Users\\Admin\\Desktop\\sTUDY\\2023-07 ML HTML\\testerimg\\",  
                  threshold = 4/3,
                  obj_type: str='path',
              transform: torchvision.transforms = None,
              label : str = None,
              display = True,
              out_trans = True):

    if img_obj is None:  return print("No input")
    if obj_type =='path':rrat = Image.open(folder+img_obj)
    if obj_type == 'raw': rrat = img_obj
        
    print(type(rrat))
    width,height=rrat.size
    mi=min(rrat.size)
    rrat=rrat.resize((round(32*width/mi),round(32*height/mi)))
    mi2=min(rrat.size)
    ttt=transforms.ToTensor()
    final=ttt(rrat)
    print(final.size())
    if final.size(0)>3: final=final[0:3,:,:]  #removes alpha layer if exist
    print(final.size())
    if max(width/height,height/width)>=threshold:
        dim_size=math.floor(width/height)+1
        if dim_size>2: dim_size=2
        print("w-h= %5s"%(width/height))
        print("thres= %5s"%threshold)
        print("dim size= %5s"%dim_size)
        testten1=torch.tensor_split(final,(3), dim=dim_size) #splittensor into 1,35,32 tuples of tensor    
        sumof=0 
        temp_list=[]
        mod_list=[0,0,0]     #[a*b for a,b in zip(mylist,tt)]
        for i in range(3): 
            temp_list.append(testten1[i].std())
            sumof+=testten1[i].std()
        for j in range (3):
            temp_list[j]=temp_list[j].item()/sumof.item()     #sumof=sumof.item()
        print("raw std: %s"% temp_list)   
        temp_sort=sorted(temp_list, reverse=True)
        if abs((temp_sort[0]-temp_sort[1])-(temp_sort[1]-temp_sort[2]))<=0.07:
             mod_list=temp_list         

        else:
            for i in range(3):
                if temp_list[i]<temp_list[(i+1)%3] and temp_list[i]<temp_list[(i+2)%3]:
                    mod_list[i]=min(temp_list)/4#20
                elif temp_list[i]>temp_list[(i+1)%3] and temp_list[i]>temp_list[(i+2)%3]:
                            mod_list[i]=temp_list[i]+min(temp_list)*2/4
                else: mod_list[i]=temp_list[i]+min(temp_list)/4    
    
        # print(mod_list)
        start_point=(-1*mod_list[0]+0*mod_list[1]+1*mod_list[2])#*threshold/2+0.5
        #start_point=(-1*temp_list[0]+0*temp_list[1]+1*temp_list[2])*threshold/2+0.5
        tsize=final.size()
        rounded=int(round(start_point/3*tsize[dim_size])+8)
        print(mod_list)
        print("rounded= %s"%rounded)
        print("Max= %s"%(32*width/height))
        if rounded<0: rounded =0
        if rounded+32>tsize[dim_size]:rounded=tsize[dim_size]-32
        if dim_size==1: 
            final=final[:,rounded:rounded+32,:].unsqueeze(0)
            print("w<h, 0")
            css=[0,(rounded+10)*100/(32*height/width)]
            print("css= %s"%css)
            
        if dim_size==2: 
            final=final[:,:,rounded:rounded+32].unsqueeze(0)
            print("w>h, 0")
            css=[(rounded+10)*100/(32*(width/height)), 0]
            print("css= %s"%css)

    elif threshold>width/height>1 :
        final=final[:,:,round(final.size(2)/2-16):round(final.size(2)/2)+16].unsqueeze(0)
        css=[50,50]
        print("w>h,1")
    elif threshold>height/width>=1:
        final=final[:,round(final.size(1)/2-16):round(final.size(1)/2)+16,:].unsqueeze(0)
        css=[50,50]
        print("w<h,1")
    #final=final.unsqueeze(0)  
    #shape= 3, 32, 2
        #rrat_sani=rrat.crop(( (rrat.size[0]-mi2)/2, (rrat.size[1]-mi2)/2, 
             #    (rrat.size[0]-mi2)/2+32,(rrat.size[1]-mi2)/2+32 ))
        #ttt=transforms.ToTensor()
        #final=ttt(rrat_sani).unsqueeze(0)
   # if img_tensor is not None:
   #     assert img_tensor.shape == (1,3,32,32), "Error: Tensor shape should be [1,3,32,32]"
   #     final=img_tensor
    
    if display == True:
        disp=final.squeeze(0)
        #disp=TTT(final).squeeze(0)
        nping=disp.numpy()
        #print(nping)
        # print(nping.size())
        plt.imshow(np.transpose(nping,(1,2,0)))
        plt.show()
        '''
    if out_trans==True:
        mean1, std1 =(final.mean((2,3))),(final.std((2,3)))
        TTT=transforms.Normalize((mean1[0][0],mean1[0][1],mean1[0][2]),(std1[0][0],std1[0][1],std1[0][2]))  
        return TTT(final)
        '''
    return final , css[0], css[1]