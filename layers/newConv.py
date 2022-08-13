from layers.layer import layer
from scipy import signal
import activationFunctions
import numpy as np
from weightInit import he

class kernelLayer(layer):
    def __init__(self, inputShape, kernelSize, kernelDepth, activation=activationFunctions.leakyRelu, activationD=activationFunctions.leakyReluD, pad=0, mode='valid') -> None:
        self.inputShape = inputShape #3 dimensions
        self.kernelSize = kernelSize
        self.kernelDepth = kernelDepth
        inputDepth, inputHeight, inputWidth = inputShape
        self.inputDepth = inputDepth
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
        if(mode=='valid'):
            self.outputShape = (kernelDepth, inputHeight - kernelSize + 1, inputWidth - kernelSize + 1)
        else:
            self.outputShape = (kernelDepth, inputHeight, inputWidth)
        self.kernelShape = (kernelDepth, inputDepth, kernelSize, kernelSize)
        self.initWeights()
        self.initBias()
        self.pad = pad
        self.actFunc = activation
        self.actFuncDerivative = activationD
        self.mode = mode
        if(self.mode == 'same'):
            self.topPad = (self.kernelShape[2]-1)//2
            self.bottomPad = (self.kernelShape[2])//2
            self.leftPad = (self.kernelShape[3]-1)//2
            self.rightPad = (self.kernelShape[3])//2
        
    def reinit(self) -> None:
        self.initWeights()
        self.initBias()
    
    def backPropagation(self, input, gradient):
        db = np.sum(gradient, axis=(0,1,2), keepdims=True)
        
        if self.mode=="same": 
            pad = (w.shape[0]-1)//2
        else: #padding is valid - i.e no zero padding
            pad =0 
        x_padded = np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),'constant', constant_values = 0)
        
        #this will allow us to broadcast operations
        x_padded_bcast = np.expand_dims(x_padded, axis=-1) # shape = (m, i, j, c, 1)
        dZ_bcast = np.expand_dims(gradient, axis=-2) # shape = (m, i, j, 1, k)
        
        dW = np.zeros_like(self.weights)
        f=w.shape[0]
        w_x = x_padded.shape[1]
    
        dx = np.zeros_like(x_padded,dtype=float) 
        Z_pad = f-1
        dZ_padded = np.pad(gradient,((0,0),(Z_pad,Z_pad),(Z_pad,Z_pad),(0,0)),'constant', constant_values = 0)  
        
        for m_i in range(x.shape[0]):
            for k in range(w.shape[3]):
                for d in range(x.shape[3]):
                    dx[m_i,:,:,d]+= signal.convolve(dZ_padded[m_i,:,:,k],w[:,:,d,k])[f//2:-(f//2),f//2:-(f//2)]
        dx = dx[:,pad:dx.shape[1]-pad,pad:dx.shape[2]-pad,:]
        return dx,dW,db
    
    def deltaWeights(self, input, gradient):
        if self.mode=="same": 
            pad = (w.shape[0]-1)//2
        else: #padding is valid - i.e no zero padding
            pad =0 
        x_padded = np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),'constant', constant_values = 0)
        
        #this will allow us to broadcast operations
        x_padded_bcast = np.expand_dims(x_padded, axis=-1) # shape = (m, i, j, c, 1)
        dZ_bcast = np.expand_dims(gradient, axis=-2) # shape = (m, i, j, 1, k)
        
        dW = np.zeros_like(self.weights)
        f=w.shape[0]
        w_x = x_padded.shape[1]
        for a in range(f):
            for b in range(f):
                #note f-1 - a rather than f-a since indexed from 0...f-1 not 1...f
                dW[a,b,:,:] = np.sum(dZ_bcast*x_padded_bcast[a:w_x-(f-1 -a),b:w_x-(f-1 -b),:,:],axis=(0,1,2))  

    def updateWeights(self, delta):
        self.weights -= self.learningRate * delta
    
    def updateBias(self, gradient):
        self.bias = np.subtract(self.bias, self.learningRate * gradient)
    
    def forward(self, input):
        if self.mode=="same": 
            pad = (w.shape[0]-1)//2
        else: #padding is valid - i.e no zero padding
            pad =0 
        n = (x.shape[1]-w.shape[0]+2*pad) +1 #ouput width/height
        
        y = np.zeros((x.shape[0],n,n,w.shape[3])) #output array
        
        #pad input
        x_padded = np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),'constant', constant_values = 0)
        
        #flip filter to cancel out reflection
        w = np.flip(w,0)
        w = np.flip(w,1)
        
        f = w.shape[0] #size of filter
            
        for m_i in range(x.shape[0]): #each of the training examples
            for k in range(w.shape[3]): #each of the filters
                for d in range(x.shape[3]): #each slice of the input 
                    y[m_i,:,:,k]+= signal.convolve(x_padded[m_i,:,:,d],w[:,:,d,k])[f//2:-(f//2),f//2:-(f//2)] #sum across depth
                
        y = y + b #add bias (this broadcasts across image)
        return y

    def initWeights(self):
        self.weights = he(self.kernelShape, self.inputDepth * self.inputHeight * self.inputWidth)
            
    def initBias(self):
        self.bias = he(self.outputShape, self.inputDepth * self.inputHeight * self.inputWidth)
        
    