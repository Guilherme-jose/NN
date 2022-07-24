from layers.layer import layer
import numpy as np

class maxPoolLayer(layer):
    def __init__(self, inputShape, outputShape, mode="max") -> None:
        self.inputShape = inputShape
        self.outputShape = outputShape
        inputDepth, inputHeight, inputWidth = inputShape
        self.inputDepth = inputDepth
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
        self.method = mode
        self.kernelShape = (2,2); 
        
    def reinit(self) -> None:
        pass
    
    #takes input as matrix, for use inside the network
    def forward(self, input): #recheck later
        self.output = np.zeros(self.outputShape)
        heightOffset = self.inputHeight%self.kernelShape[0]
        widthOffset = self.inputWidth%self.kernelShape[1]
        if heightOffset != 0 or widthOffset != 0:
            input = np.pad(input, ((0,0),(0,heightOffset),(0,widthOffset)))
            
        for i in range(self.inputDepth):
            self.output[i] = self.pool2D(input[i])
        return self.output
    
    def backPropagation(self, input, gradient):#SLOOOOOOOW
        C = self.inputDepth # number channels
        H = self.inputHeight # height input
        W = self.inputWidth # width input
        pool_height = self.kernelShape[0]
        pool_width = self.kernelShape[1]
        stride = self.kernelShape[0]
        
        H_out = int(1 + (H - pool_height) / stride)
        W_out = int(1 + (W - pool_width) / stride)

        dx = np.zeros_like(input)
        
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    # get the index in the region i,j where the value is the maximum
                    i_t, j_t = np.where(np.max(input[c, i * stride : i * stride + pool_height, j * stride : j * stride + pool_width]) == input[c, i * stride : i * stride + pool_height, j * stride : j * stride + pool_width])
                    i_t, j_t = i_t[0], j_t[0]
                    # only the position of the maximum element in the region i,j gets the incoming gradient, the other gradients are zero
                    dx[c, i * stride : i * stride + pool_height, j * stride : j * stride + pool_width][i_t, j_t] = gradient[c, i, j]

        
        return dx
    
    def pool2D(self, mat, ksize=(2,2), pad=False):
        m, n = mat.shape[:2]
        ky,kx=ksize

        _ceil=lambda x,y: int(np.ceil(x/float(y)))

        if pad:
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