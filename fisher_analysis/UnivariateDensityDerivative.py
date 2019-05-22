import math

class UnivariateDensityDerivative(object):

    def __init__(self, NSources, MTargets, pSources, pTargets, Bandwidth, Order):
        self.N = NSources
        self.M = MTargets
        self.px = pSources
        self.h = Bandwidth
        self.r = Order
        self.py = pTargets
        self.pD = [None] * MTargets


    def evaluate(self):
        two_h_square = 2*self.h*self.h
        q = (math.pow(-1,self.r))/(math.sqrt(2*math.pi)*self.N*(math.pow(self.h,(self.r+1))))
        for j in range(self.M):
            self.pD[j]=0.0
            for i in range(self.N):
                temp = self.py[j]-self.px[i]
                norm = temp*temp
                self.pD[j] = self.pD[j]+(hermite(temp/self.h,self.r)*math.exp(-norm/two_h_square))
            self.pD[j]=self.pD[j]*q


def hermite(x, r):
    if r == 0:
        return 1.0
    elif r == 1:
        return x
    else:
        return (x*hermite(x,r-1))-((r-1)*hermite(x,r-2))
