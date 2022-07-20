from layers.layer import layer
import numpy as np

class maxPoolLayer(layer):
    def __init__(self, inputShape, outputShape, mode="max") -> None:
        self.inputShape = inputShape
        self.outputShape = outputShape
        inputDepth, inputHeight, inputWidth = inputShape
        self.inputDepth = inputDepth
        self.method = mode
        
    def reinit(self) -> None:
        pass
    
    #takes input as matrix, for use inside the network
    def forward(self, input): #recheck later
        output = np.zeros(self.outputShape)
        for i in range(self.inputDepth):
            output[i] = self.pool2D(input[i])
        return output
    
    def backPropagation(self, input, output, gradient):
        out = np.zeros(self.inputShape)
        kMatrix = np.array(((0,0),(0,1)))
        for i in range(self.inputDepth):
            out[i] = np.kron(gradient[i], kMatrix)
        return out
    
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