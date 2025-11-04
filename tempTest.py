import torch
import torch.nn.functional as F

def self_softmax(x) :
    exp_val = torch.exp(torch.tensor([[1,2,3,4],[1,2,3,4]], dtype=torch.float32) - torch.tensor([[0],[0]], dtype=torch.float32))
    exp_sum = torch.sum(exp_val,1)[0]
    
    
    print( exp_val / exp_sum)

if __name__ == "__main__" :
    x = torch.tensor([[1,2,3,4],[1,2,3,4]], dtype=torch.float32)
    ret = F.softmax(x,dim = 1)
    print("softmax=", ret)
    print("self_soft = ",self_softmax(x)) 
    