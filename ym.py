#coding=utf8
import math,random,copy

def gweightarr(arr):
    ret=[]
    for i in range(len(arr)):
        ret+=[i] * int(100 * arr[i])
    return ret
def getrandomvalue(weightarr):
    i=random.randint(0,len(weightarr)-1)
    return weightarr[i] 
class ym():
    def __init__(self,pi=None,A=None,B=None):
        self.pi=pi
        self.A=A
        self.B=B
        self.Aarr=[]
        self.Barr=[]
        if pi != None:
            self.piarr=gweightarr(self.pi)
            for i in range(len(self.A)):
                self.Aarr.append(gweightarr(self.A[i]))
            for i in range(len(self.B)):
                self.Barr.append(gweightarr(self.B[i]))
        #print self.pi,self.piarr
        #print self.A,self.Aarr
        #print self.B,self.Barr
            
    def train(self,data,loopcnt=10):
        N=5
        M=2
        pi=[1/float(N)] * N
        A= [ [1/float(N)  for j in range(N) ] for i in range(N)]
        B=[ [ 1/float(M) for j in range(M)] for i in range(N)]
        T=len(data)
    
        print data
        print pi,A,B,T
        def r(t,i):
            self.A=A
            self.B=B
            self.pi=pi
            nouse,alphaarr = self.forward(data,t)
            nouse,betaarr=self.backward(data,t)
            tmp1=alphaarr[i]
            tmp2=betaarr[i]
            porders = self.forward(data)
            return tmp1 * tmp2/porders[0]
        def lw(i,j,t):
            self.A=A
            self.B=B
            self.pi=pi
            nouse,alphaarr = self.forward(data,t)
            nouse,betaarr=self.backward(data,t+1)
            tmp1=alphaarr[i]
            tmp2=betaarr[j]
            porders= self.forward(data)
            #print "aa",porders
            return tmp1 *self.A[i][j]*self.B[j][data[t+1]]*tmp2/porders[0]
        for n in range(loopcnt):
            tmpA=copy.deepcopy(A)
            tmpB=copy.deepcopy(B)
            tmppi=copy.deepcopy(pi)
            for i in range(N):
                for j in range(N):
                    tmpA[i][j]=sum([  lw(i,j,t) for t in range(T-1)] )\
                                /sum([ r(i,t) for t in range(T-1)])
            for j in range(N):
                for k in range(M):
                    tmpB[j][k]=sum([ r(j,t ) for t in range(T-1) if data[t]==k])\
                                /sum([ r(j,t) for t in range(T-1)])
            for i in range(N):
                tmppi[i]=r(i,1)
            A=tmpA
            B=tmpB
            pi=tmppi
            print n
            print pi
    def forecast(self,count=20):
        q=getrandomvalue(self.piarr)
        ret=[]
        for i in range(count):
            y = getrandomvalue(self.Barr[q])
            ret.append(y)
            q=getrandomvalue(self.Aarr[q])
        return ret
    def forward(self,orders,split=None):
        alphaarr=[0]*len(self.A)
        T=len(orders)
        N=len(self.A)
        if split==None:split=T-1
        
        for i in range(N):
            alphaarr[i]=self.pi[i]*self.B[i][orders[0]]
        #print alphaarr
        for t in range(1,T):
            if t >split:
                break
            alphaarr_tmp=[0] * N
            for i in range(N):
                sum1=sum( [ alphaarr[j]*self.A[j][i] for j in range(N)])
                sum2=self.B[i][orders[t]]
                alphaarr_tmp[i]=sum1 * sum2
                #print t,i,sum1,sum2,alphaarr_tmp[i]
            alphaarr=alphaarr_tmp[:]    
            #print t,alphaarr    
        #print alphaarr
        ret = sum([ alphaarr[i] for i in range(N)])
        return ret,alphaarr
    def backward(self,orders,split=None):
        N=len(self.A)
        M=len(self.B[0])
        beta=[1]* N
        #print beta
        datanum = len(orders)
        if split==None or split >datanum -1:
            split=0
        for t in range(datanum-2,split-1,-1):
            betatmp=[]
            for i in range(N):
                v=sum([ beta[j]*self.A[i][j]*self.B[j][orders[t+1]] for j in range(N)])
                betatmp.append(v)
            beta=betatmp[:]
            #print t,beta
            
        ret=sum([ self.pi[i]*self.B[i][orders[0]] *beta[i]   for i in range(N)])     
        return ret,beta
    def for_back(self,orders,split=None):
        if split == None:
            split =len(order)/2
        p1,alpha = self.forward(orders,split)
        p2,beta = self.backward(orders,split+1)
        ret = 0
        for i in range(len(self.A)):
            for j in range(len(self.A)):
                ret +=alpha[i]* self.A[i][j]*self.B[j][orders[split+1]]*beta[j]
        print 'for_back',split,p1,p2,ret
        return ret
    def vitbi(self,orders):
        N=len(self.A)
        M=len(self.B[0])
        T=len(orders)
        delta=[[ self.pi[i]*self.B[i][orders[0]]for i in range(N)]]
        path=[[0]*N]
        
        print 0,delta[0],path[0]
        for t in range(1,T):
            delta.append( [0]*N )
            path.append([0]*N)
            for i in range(N):
                tmparr=[ (delta[t-1][j]*self.A[j][i]*self.B[i][orders[t]] ,j) \
                                  for j in range(N)]
                tmparr.sort(key=lambda x:x[0],reverse=True)
                delta[t][i]=tmparr[0][0]
                path[t][i]=tmparr[0][1]
            print t,delta[t],path[t]
        pos =  delta[-1].index(max(delta[-1]))
        retarr=[pos]*T
        for t in range(T-2,-1,-1):
            retarr[t]=path[t+1][retarr[t+1]]
        return retarr,max(delta[-1])
def test_forecast():
    pi=(0.25,) * 4
    A =[ [0,1,0,0],[0.4,0,0.6,0],[0,0.4,0,0.6],[0,0,0.5,0.5] ]
    B= [ [0.5,0.5],[0.3,0.7],[0.6,0.4],[0.8,0.2] ]
    ymins = ym(pi,A,B)
    print ymins.forecast()
    
def test_op():
    pi=[0.2,0.4,0.4]
    A=[[0.5,0.2,0.3],[0.3,0.5,0.2] ,[0.2,0.3,0.5]]
    B=[[0.5,0.5],[0.4,0.6],[0.7,0.3]]
    O=[0,1,0]
    ymins=ym(pi,A,B)
    print ymins.forward(O)
    for i in range(5):
        print i,ymins.forward(O,i)
 
    print "back"
    print ymins.backward(O)
    for i in range(5):
        print i, ymins.backward(O,i)
    print 'for_back'
    for i in range(2):
        print ymins.for_back(O,i)
        
def train():
    O=[[0,0,1,1,0],[1,0,1,1,0],[0,0,1,1,1]]
    O=[0,0,1,1,0]
    ymins=ym()
    ymins.train(O)
    print 'train reuslt:'
    print 'pi',ymins.pi
    print 'A',ymins.A
    print "B",ymins.B

def test_vitbi():
    pi=[0.2,0.4,0.4]
    A=[[0.5,0.2,0.3],[0.3,0.5,0.2] ,[0.2,0.3,0.5]]
    B=[[0.5,0.5],[0.4,0.6],[0.7,0.3]]
    O=[0,1,0]
    ymins=ym(pi,A,B)
    print ymins.vitbi(O)
    
if __name__=='__main__':
    #test_forecast()
    #test_op()
    #train()
    test_vitbi()
   
