from layers.layer import layer
import numpy as np

class maxPoolLayer(layer):
    pad = False
    def __init__(self, inputShape, mode="max") -> None:
        self.inputShape = inputShape
        
        inputDepth, inputHeight, inputWidth = inputShape
        self.outputShape = (inputDepth, (inputHeight+1)//2, (inputWidth+1)//2)
        self.inputDepth = inputDepth
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
        self.method = mode
        self.kernelShape = (2,2); 
        self.heightOffset = self.inputHeight%self.kernelShape[0]
        self.widthOffset = self.inputWidth%self.kernelShape[1]
        if(self.heightOffset != 0 or self.widthOffset != 0): self.pad = True
    def reinit(self) -> None:
        pass
    
    #takes input as matrix, for use inside the network
    def forward(self, input): #recheck later
        self.output = np.zeros(self.outputShape)
        
        if self.pad:
            input = np.pad(input, ((0,0),(0,self.heightOffset),(0,self.widthOffset)))
            
        for i in range(self.inputDepth):
            self.output[i] = self.pool2D(input[i])
        return self.output
    
    def backPropagation(self, input, gradient):
        dx = np.kron(gradient, np.ones(self.kernelShape))
        dx = np.where(np.kron(self.output, np.ones(self.kernelShape)) == input, dx, 0)
        if self.pad:
            dx = dx[:,:-self.heightOffset,:]
            dx = dx[:,:,:-self.widthOffset]
        return dx
    
    def pool2D(self, mat, ksize=(2,2)):
        m, n = mat.shape[:2]
        ky,kx=ksize

        _ceil=lambda x,y: int(np.ceil(x/float(y)))

        if self.pad:
            ny=_ceil(m,ky)
            nx=_ceil(n,kx)
            size=(ny*ky, nx*kx)+mat.shape[2:]
            mat_pad=np.full(size,np.nan)
            mat_pad[:m,:n,...]=mat
        else:
            ny=m//ky
            nx=n//kx
            mat_pad=mat[:ny*ky, :nx*kx, ...]

        new_shape=(ny,ky,nx,kx)+mat.shape[2:]

        if self.method=='max':
            result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
        else:
            result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

        return result