import torchvision
import torch
print(torch.__version__)
print(torchvision.__version__)

def main():
    a = torch.tensor([7])
    # print(a.item(), a.shape)
    # 7

    b = torch.tensor([[[1, 2, 3],
            [3, 6, 9],
            [2, 4, 5]]])
    print(b.ndim, b.shape)
    # 3, torch.size([1,3,3])

    zero_to_ten = torch.arange(start=0, end=10, step=1)
    print(zero_to_ten)
    #tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    x = torch.rand(size=(2,3))
    y = torch.rand(size=(3,10))
    # print(torch.matmul(x,y).shape)

    # One of the most common errors in deep learning (shape errors)
    x = torch.rand(size=(3,2))
    y = torch.rand(size=(3,10))
    print(torch.transpose(x, 1, 0).shape, x.T.shape)


    linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input 
                            out_features=6) # out_features = describes outer value 
    x = torch.ones(size=(10,2))
    output = linear(x)
    print(output.shape)

    output = output.type(torch.float16)

    def testing():
        x = torch.arange(1., 8.)
        print(x, x.shape)

if __name__ == "__main__":
    main()