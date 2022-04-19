import numpy as np
def traidcheck(A, B, pi) -> tuple:
    assert len(A.shape) == len(B.shape) == len(pi.shape) == 2, "Matrices must be 2-Dimensional."
    assert pi.shape[1] == 1, "Initial Probability Distribution must be a Col-vector."
    assert A.shape[0] == A.shape[1] == B.shape[1] == pi.shape[0], "Num of states must be the same."
    return A.shape[0], B.shape[0]
"""
def viterbi_shortest_trans(A, num_states) -> list[tuple]:
    res = []
    for k in range(num_states):
        current_max = A[ : , k ].max()
        for l in range(num_states):
            if A[ l , k ] == current_max:
                res.append((l, k))
    return res
"""
class HMM_Traid(object):
    """
    Traid of HMM (Hidden Markov Model), l = (A, B, pi)
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, pi: np.ndarray, Q:list=None, U:list=None):
        self.__transmat = np.array(A) # State Transition Probability Distribution
        self.__emission = np.array(B) # Emission Prob Distribution
        self.__initprob = np.array(pi) # Initial Prob Distribution
        _ = traidcheck(self.__transmat, self.__emission, self.__initprob)
        self.__statesidx = list(range(_[0]))
        self.__states = Q or self.__statesidx
        self.__observationsidx = list(range(_[1]))
        self.__observations = U or self.__observationsidx
        assert (len(self.__states), len(self.__observations)) == _, ""
        #self.__viterbi_shortest = viterbi_shortest_trans(self.__transmat, len(self.__states))
    def __repr__(self) -> str:
        return """<%s λ = (A, B, π)
Transition Mat A:
%s
Emission Mat B:
%s
Initial Prob π:
%s
@ %s>""" % (type(self), self.__transmat, self.__emission, self.__initprob, hex(id(self)))
    @property
    def transition(self):   return self.__transmat.copy()
    @property
    def emission(self):     return self.__emission.copy()
    @property
    def initprob(self):     return self.__initprob.copy()
    @property
    def shape(self):        return len(self.__states), len(self.__observations)
    @property
    def states(self):       return self.__states
    @property
    def observations(self): return self.__observations
    # Process of Generating Probability Sequence of Observation
    def probseq_observe(self, T: int) -> np.ndarray:
        res = np.zeros((len(self.__observations), T))
        cur = self.__initprob.copy()
        for k in range(T):
            res[ : , k : k + 1 ] += self.__emission @ cur
            cur = self.__transmat @ cur
        return res
    # Generating Observation Sequence
    def observe(self, T: int, return_names:bool=False) -> list:
        res = []
        cur = np.random.choice(self.__statesidx, p=self.__initprob[ : , 0 ])
        for k in range(T):
            res.append(np.random.choice((self.__observations if return_names else self.__observationsidx), p=self.__emission[ : , cur ]))
            cur = np.random.choice(self.__statesidx, p=self.__transmat[ : , cur ])
        return res
    # Backward Probability
    def __pr_back(self, obseq: list, current_state: int, t:int=0) -> float:
        return np.sum([
            self.__transmat[ k , current_state ] * self.__emission[ obseq[t + 1] , k ] * self.__pr_back(obseq, k, t + 1)
            for k in self.__statesidx
            ]) if t % len(obseq) < len(obseq) - 1 else 1.
    # Backward Algorithm
    def prob_back(self, obseq: list) -> float:
        return np.sum([
            self.__initprob[ k , 0 ] * self.__emission[ obseq[0] , k ] * self.__pr_back(obseq, k)
            for k in self.__statesidx
            ])
    # Probability Prediction via Viterbi Algorithm
    def predict(self, obseq: list) -> tuple:
        cur_most_prob = [ self.__initprob[ k , 0 ] * self.__emission[ obseq[0] , k ] for k in self.__statesidx ]
        idx_prev_node = [[ 0 for k in self.__statesidx ]]
        for t in range(1, len(obseq)):
            transient_prob = [
                [ cur_most_prob[l] * self.__transmat[ k , l ] for l in self.__statesidx ]
                for k in self.__statesidx
                ]
            idx_prev_node.append([
                np.argmax(transient_prob[k])
                for k in self.__statesidx
                ])
            cur_most_prob = [
                np.max(transient_prob[k]) * self.__emission[ obseq[t] , k ]
                for k in self.__statesidx
                ]
        res = [ np.argmax(cur_most_prob) ]
        for t in range(len(obseq) - 1):
            res.append(idx_prev_node[-1 - t][res[-1]])
        return res[ :: (-1) ], cur_most_prob[res[0]]
    # Forward Probability
    def __pr_fore(self, obseq: list, current_state: int, t:int=-1) -> float:
        return np.sum([
            self.__pr_fore(obseq, k, t - 1) * self.__transmat[ current_state , k ]
            for k in self.__statesidx
            ]) * self.__emission[ obseq[t] , current_state ] if t % len(obseq) else self.__initprob[ current_state , 0 ] * self.__emission[ obseq[0] , current_state ]
    # Forward Algorithm
    def prob_fore(self, obseq: list) -> float:
        return np.sum([ self.__pr_fore(obseq, k) for k in self.__statesidx ])
    # Probability at Specified Moment
    def __pr_spec(self, obseq: list, current_state: int, t: int) -> float:
        probs = [ self.__pr_fore(obseq, k, t) * self.__pr_back(obseq, k, t) for k in self.__statesidx ]
        return probs[current_state] / np.sum(probs)
    # Probability between 2 Adjacent Moments
    def __pr_adja(self, obseq: list, cur_states: list[2], t: int) -> float:
        probs = [ [
                self.__pr_fore(obseq, k, t) * self.__transmat[ l , k ] * self.__emission[ obseq[t + 1] , l ] * self.__pr_back(obseq, l, t + 1)
                for l in self.__statesidx
                ]
            for k in self.__statesidx
            ]
        return probs[cur_states[0]][cur_states[1]] / np.sum(probs)
    # Testing
    def prprpr(self, obseq, cur_states, t):
        return self.__pr_spec(obseq, cur_states[0], t), self.__pr_adja(obseq, cur_states, t), self.__pr_spec(obseq, cur_states[1], t + 1)
    # Parameters-Fitting via Baum Welch
    def fit_seq(self, obseq: list):
        timelength = len(obseq)
        self.__transmat = np.array([ [
                np.sum([
                    self.__pr_adja(obseq, [ k , l ], t)
                    for t in range(timelength - 1)
                    ]) / np.sum([
                    self.__pr_spec(obseq, k, t)
                    for t in range(timelength - 1)
                    ])
                for k in self.__statesidx
                ]
            for l in self.__statesidx
            ])
        prob_spec_times = [ [ self.__pr_spec(obseq, k, t) for t in range(timelength) ] for k in self.__statesidx ]
        prob_spec_time_observed = []
        for k in self.__statesidx:
            prob_spec_time_observed.append([])
            for l in self.__observationsidx:
                prob_spec_time_observed[-1].append(0.)
                for t in range(timelength):
                    if obseq[t] == l:
                        prob_spec_time_observed[-1][-1] += prob_spec_times[k][t]
        for k in self.__statesidx:
            for l in self.__observationsidx:
                self.__emission[ l , k ] = prob_spec_time_observed[k][l] / np.sum(prob_spec_times[k])
        for k in self.__statesidx:
            self.__initprob[ k , 0 ] = prob_spec_times[k][0]
        return self
def new_hmm(shape, name_states=None, name_observations=None):
    return HMM_Traid(
        A = np.identity(shape[0]),
        B = np.ones((shape[1] , shape[0])) / shape[1],
        pi = np.ones((shape[0] , 1)) / shape[0],
        Q = name_states, U = name_observations
        )