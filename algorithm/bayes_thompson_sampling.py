import numpy as np
from scipy.stats import beta

class BayesThompsonSampling:
    '''
        Bayes - in each round to count pos and neg
        Thompson Sampling - 
            in init we set a, b is 1.
            in each round we count every arms a, b by:
                a += pos
                b += neg
            and then we get the best sample (by sample = np.random.beta(bandits[j].a, bandits[j].b))
            at above to choice the arms. 
    '''
    def __init__(self,mu):
        self.mu= mu
        self.mean = 0
        self.n= 0 # how many times run
        self.a = 1
        self.b = 1
        self.pos = 0
        self.neg = 0

    def pull(self):
        return np.random.randn()+self.mu
    
    def update(self, xn):
        self.n +=1
        self.mean = ( 1 - 1.0/self.n ) * self.mean + 1.0/self.n * xn

    def run(mu1, mu2, mu3, epsilon=0.1, N=100000):
        bandits=[BayesThompsonSampling(mu1),
                BayesThompsonSampling(mu2),
                BayesThompsonSampling(mu3)]
        data=np.empty(N)
        bandits_len = len(bandits)
        pre_bandit = -1

        for i in range(N):
            # take a sample from each bandit
            sampled_theta = []
            for j in range(bandits_len):
                bandits[j].a += bandits[j].pos
                bandits[j].b += bandits[j].neg
                bandits[j].neg += 1
                sample = np.random.beta(bandits[j].a, bandits[j].b)
                sampled_theta += [ sample ]
            j = sampled_theta.index( max(sampled_theta) )
            bandits[j].neg -= 1
            bandits[j].pos += 1

            x = bandits[j].pull()
            bandits[j].update(x)
            data[i] = x

        cumul_average =np.cumsum(data)/(np.arange(N)+1)
        return cumul_average