import numpy as np
def traidcheck(A, B, pi) -> tuple:
    assert len(A.shape) == len(B.shape) == len(pi.shape) == 2, "Matrices must be 2-Dimensional."
    assert pi.shape[0] == 1, "Initial Probability Distribution must be a Row-vector."
    assert A.shape[0] == A.shape[1] == B.shape[0] == pi.shape[1], "Num of states must be the same."
    return A.shape[1], B.shape[1]
"""
def viterbi_shortest_trans(A, num_states) -> list[tuple]:
    res = []
    for k in range(num_states):
        current_max = A[ k , : ].max()
        for l in range(num_states):
            if A[ k , l ] == current_max:
                res.append((k, l))
    return res
"""
class Hidden_Markov_Model(object):
    """
    Traid of HMM (Hidden Markov Model), l = (A, B, pi)
    """
    def __init__(ミク, A: np.ndarray, B: np.ndarray, pi: np.ndarray, Q:list=None, V:list=None):
        ミク.__transmat = np.array(A) # State Transition Probability Distribution
        ミク.__emission = np.array(B) # Emission Prob Distribution
        ミク.__initprob = np.array(pi) # Initial Prob Distribution
        _ = traidcheck(ミク.__transmat, ミク.__emission, ミク.__initprob)
        ミク.__statesidx = list(range(_[0]))
        ミク.__states = Q or ミク.__statesidx
        ミク.__observationsidx = list(range(_[1]))
        ミク.__observations = V or ミク.__observationsidx
        assert (len(ミク.__states), len(ミク.__observations)) == _, ""
        #ミク.__viterbi_shortest = viterbi_shortest_trans(ミク.__transmat, len(ミク.__states))
    def __repr__(ミク) -> str:
        return """<%s λ = (A, B, π)
Transition Mat A:
%s
Emission Mat B:
%s
Initial Prob π:
%s
@ %s>""" % (type(ミク).__name__, ミク.__transmat, ミク.__emission, ミク.__initprob, hex(id(ミク)))
    @property
    def transition(ミク):   return ミク.__transmat.copy()
    @property
    def emission(ミク):     return ミク.__emission.copy()
    @property
    def initprob(ミク):     return ミク.__initprob.copy()
    @property
    def shape(ミク):        return len(ミク.__states), len(ミク.__observations)
    @property
    def states(ミク):       return ミク.__states
    @property
    def observations(ミク): return ミク.__observations
    # Process of Generating Probability Sequence of Observation
    def probseq_observe(ミク, T: int) -> np.ndarray:
        res = np.zeros((T, len(ミク.__observations)))
        cur = ミク.__initprob.copy()
        for k in range(T):
            res[ k : k + 1 , : ] += cur @ ミク.__emission
            cur = cur @ ミク.__transmat
        return res
    # Generating Observation Sequence
    def observe(ミク, T: int, return_names:bool=False) -> list:
        res = []
        cur = np.random.choice(ミク.__statesidx, p=ミク.__initprob[ 0 , : ])
        for k in range(T):
            res.append(np.random.choice((ミク.__observations if return_names else ミク.__observationsidx), p=ミク.__emission[ cur , : ]))
            cur = np.random.choice(ミク.__statesidx, p=ミク.__transmat[ cur , : ])
        return res
    # Backward Probability
    def __pr_back(ミク, obseq: list, current_state: int, t:int=0) -> float:
        return np.sum([
            ミク.__transmat[ current_state , k ] * ミク.__emission[ k , obseq[t + 1] ] * ミク.__pr_back(obseq, k, t + 1)
            for k in ミク.__statesidx
            ]) if t % len(obseq) < len(obseq) - 1 else 1.
    # Backward Algorithm
    def prob_back(ミク, obseq: list) -> float:
        return np.sum([
            ミク.__initprob[ 0 , k ] * ミク.__emission[ k , obseq[0] ] * ミク.__pr_back(obseq, k)
            for k in ミク.__statesidx
            ])
    # Probability Prediction via Viterbi Algorithm
    def predict(ミク, obseq: list) -> tuple:
        cur_most_prob = [ ミク.__initprob[ 0 , k ] * ミク.__emission[ k , obseq[0] ] for k in ミク.__statesidx ]
        idx_prev_node = [[ 0 for k in ミク.__statesidx ]]
        for t in range(1, len(obseq)):
            transient_prob = [
                [ cur_most_prob[l] * ミク.__transmat[ l , k ] for l in ミク.__statesidx ]
                for k in ミク.__statesidx
                ]
            idx_prev_node.append([
                np.argmax(transient_prob[k])
                for k in ミク.__statesidx
                ])
            cur_most_prob = [
                np.max(transient_prob[k]) * ミク.__emission[ k , obseq[t] ]
                for k in ミク.__statesidx
                ]
        res = [ np.argmax(cur_most_prob) ]
        for t in range(len(obseq) - 1):
            res.append(idx_prev_node[-1 - t][res[-1]])
        return res[ :: (-1) ], cur_most_prob[res[0]]
    # Forward Probability
    def __pr_fore(ミク, obseq: list, current_state: int, t:int=-1) -> float:
        return np.sum([
            ミク.__pr_fore(obseq, k, t - 1) * ミク.__transmat[ k , current_state ]
            for k in ミク.__statesidx
            ]) * ミク.__emission[ current_state , obseq[t] ] if t % len(obseq) else ミク.__initprob[ 0 , current_state ] * ミク.__emission[ current_state , obseq[0] ]
    # Forward Algorithm
    def prob_fore(ミク, obseq: list) -> float:
        return np.sum([ ミク.__pr_fore(obseq, k) for k in ミク.__statesidx ])
    # Probability at Specified Moment
    def __pr_spec(ミク, obseq: list, current_state: int, t: int) -> float:
        probs = [ ミク.__pr_fore(obseq, k, t) * ミク.__pr_back(obseq, k, t) for k in ミク.__statesidx ]
        return probs[current_state] / np.sum(probs)
    # Probability between 2 Adjacent Moments
    def __pr_adja(ミク, obseq: list, cur_states: list[2], t: int) -> float:
        probs = [ [
                ミク.__pr_fore(obseq, k, t) * ミク.__transmat[ k , l ] * ミク.__emission[ l , obseq[t + 1] ] * ミク.__pr_back(obseq, l, t + 1)
                for l in ミク.__statesidx
                ]
            for k in ミク.__statesidx
            ]
        return probs[cur_states[0]][cur_states[1]] / np.sum(probs)
    # Testing
    def prprpr(ミク, obseq, cur_states, t):
        return ミク.__pr_spec(obseq, cur_states[0], t), ミク.__pr_adja(obseq, cur_states, t), ミク.__pr_spec(obseq, cur_states[1], t + 1)
    # Parameters-Fitting via Baum Welch
    def fit_seq(ミク, obseq: list):
        timelength = len(obseq)
        ミク.__transmat = np.array([ [
                np.sum([
                    ミク.__pr_adja(obseq, [ k , l ], t)
                    for t in range(timelength - 1)
                    ]) / np.sum([
                    ミク.__pr_spec(obseq, k, t)
                    for t in range(timelength - 1)
                    ])
                for l in ミク.__statesidx
                ]
            for k in ミク.__statesidx
            ])
        prob_spec_times = [ [ ミク.__pr_spec(obseq, k, t) for t in range(timelength) ] for k in ミク.__statesidx ]
        prob_spec_time_observed = []
        for k in ミク.__statesidx:
            prob_spec_time_observed.append([])
            for l in ミク.__observationsidx:
                prob_spec_time_observed[-1].append(0.)
                for t in range(timelength):
                    if obseq[t] == l:
                        prob_spec_time_observed[-1][-1] += prob_spec_times[k][t]
        for k in ミク.__statesidx:
            for l in ミク.__observationsidx:
                ミク.__emission[ k , l ] = prob_spec_time_observed[k][l] / np.sum(prob_spec_times[k])
        for k in ミク.__statesidx:
            ミク.__initprob[ 0 , k ] = prob_spec_times[k][0]
        return ミク
def new_hmm(shape, name_states=None, name_observations=None):
    return Hidden_Markov_Model(
        A = np.identity(shape[0]),
        B = np.ones((shape[0] , shape[1])) / shape[1],
        pi = np.ones((1 , shape[0])) / shape[0],
        Q = name_states, V = name_observations
        )
HMModel = HMM_Traid = HMM = Hidden_Markov_Model