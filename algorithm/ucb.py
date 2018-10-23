
import numpy as np

class UCB1:
    '''
        first run each arm to count ucb_values (by sqrt(2*log(N)/count) )
        and then choice the largest ucb_values + mean 
        at  the same time update be chose arm's ucb_values (by sqrt(2*log(N)/count) )
    '''
    def __init__(self,mu):
        self.mu= mu
        self.mean =0
        self.n=0 # how many times run
        self.ucb_values = 0
    
    def pull(self):
        return np.random.randn()+self.mu
    
    def update(self, xn):
        self.n +=1
        self.mean = ( 1 - 1.0/self.n ) * self.mean + 1.0/self.n * xn
 
    def run(mu1, mu2, mu3, epsilon=0.1, N=100000):
        bandits=[UCB1(mu1), UCB1(mu2), UCB1(mu3)]
        data=np.empty(N)
        bandits_len = len(bandits)

        for i in range(bandits_len):
            x=bandits[i].pull()
            bandits[i].update(x)
            data[i] = x
            bonus = np.sqrt( (2 * np.log(N)) / bandits[i].n )
            bandits[i].ucb_values += bonus

        for i in range(bandits_len,N):
            j = 0
            for k in range(1,bandits_len):
                a = bandits[j].ucb_values + bandits[j].mean
                b = bandits[k].ucb_values + bandits[k].mean
                if(a < b): j = k

            x=bandits[j].pull()
            bandits[j].update(x)
            data[i] = x
            
            bonus = np.sqrt( (2 * np.log(N)) / bandits[j].n )
            bandits[j].ucb_values = bonus
        
        cumul_average =np.cumsum(data)/(np.arange(N)+1)
        return cumul_average