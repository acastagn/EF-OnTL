import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from random import Random
import operator
import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.join(current_directory, '..')
sys.path.append(parent_directory)
from utils import transpose
sys.path.remove(parent_directory)




class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
       
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)



SEED=None

def set_seed(inSEED):
    global SEED
    SEED=inSEED
    if SEED:
        torch.manual_seed(SEED)
        Random.seed(SEED)

_BETA_START=.4


class prioritezed_replay_buffer:
    def __init__(self, memory_size,batch_size,random_generator,_BETA_FRAMES=100000,prob_alpha=.6, beta=0.4):
        assert(prob_alpha>0)
        self.original_apha=prob_alpha
        self.alpha=prob_alpha
        self.memory_size = memory_size
        self.buffer = []
        self.batch_size=batch_size
        self.random=random_generator
        self.pos=0
        self.current_frame=0
        self.beta=beta
        self._BETA_START=beta
        self._BETA_FRAMES= _BETA_FRAMES 
        it_capacity= 1
        while (it_capacity<self.memory_size):
            it_capacity*=2
        self._it_sum= SumSegmentTree(it_capacity)
        self._it_min= MinSegmentTree(it_capacity)
        self._max_priority= 1.0
        
    
    def clear(self):
        self.buffer.clear()
        self.pos=0
        self.alpha= self.original_apha
        self.pos=0
        it_capacity= 1
        while (it_capacity<self.memory_size):
            it_capacity*=2
        self._it_sum= SumSegmentTree(it_capacity)
        self._it_min= MinSegmentTree(it_capacity)
        self._max_priority= 1.0
        self.beta= self._BETA_START
        self.current_frame=0
    
    def push(self, experience) -> None:
        idx=self.pos
        if(self.size()<self.memory_size):
            self.buffer.append(experience)
        else:
            self.buffer[self.pos]= experience
        #update importante in pos
        self._it_sum[idx] = self._max_priority ** self.alpha
        self._it_min[idx] = self._max_priority ** self.alpha
        #pointing next.
        self.pos= (self.pos+1)%self.memory_size


    def size(self):
        return len(self.buffer)


    def _sample_proportional(self):
        res=[]
        for _ in range(self.batch_size):
            mass = self.random.random()*self._it_sum.sum(0, self.size() - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res


    def sample(self):
        assert(self.beta>0), "Beta is {}".format(self.beta)
        assert(self.batch_size<self.size()),"Not enough instances"
        idxes= self._sample_proportional()
        weights= []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.size()) ** (-self.beta)
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.size()) ** (-self.beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in idxes]
        self.update_beta()
        return samples, idxes, weights


    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities), "size mismatch"
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert (0 <= idx < self.size()), "Index {} out of range".format(idx)
            self._it_sum[idx] = priority ** self.alpha
            self._it_min[idx] = priority ** self.alpha
            self._max_priority = max(self._max_priority, priority)

    def update_beta(self):
        self.current_frame+=1
        idx= self.current_frame
        v= self._BETA_START + idx *(1 - self._BETA_START) /  self._BETA_FRAMES
        0.4+ idx *(0.6)/10000
        self.beta = min(1, v)
        return self.beta



class dueling_network(nn.Module):
    def __init__(self, state_dim, action_dim,n_latent_var, activation_f=nn.ReLU()):
        assert type(n_latent_var)==list, "N_latent_var is not a list"
        super(dueling_network, self).__init__()
        self.activation_f= activation_f
        self.input_layer = nn.Linear(state_dim, n_latent_var[0])
        if(len(n_latent_var)>1):
            self.hidden_layers = nn.ModuleList([nn.Linear(n_latent_var[idx-1], n_latent_var[idx]) for idx in range(1, len(n_latent_var))])
        else:
            self.hidden_layers=nn.ModuleList([])
        self.value_output=nn.Linear(n_latent_var[-1],1)
        #adv branch
        self.adv_output= nn.Linear(n_latent_var[-1], action_dim)
        self.softmax= nn.Softmax(dim=-1)

    def forward(self, state):
        y= self.activation_f(self.input_layer(state))
        for hl in self.hidden_layers:
            y= self.activation_f(hl(y))
        value = self.value_output(y)
        adv = self.adv_output(y)
        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage
        return Q
    

    def _get_stateValue(self, state):
        y= self.activation_f(self.input_layer(state))
        for hl in self.hidden_layers:
            y= self.activation_f(hl(y))
        value = self.value_output(y)
        return value


    def act(self, state):
        action_probs = self.forward(state)
        action_probs= self.softmax(action_probs)
        return action_probs

    def get_action(self, state):
        action_probs = self.forward(state)
        return torch.argmax(action_probs, dim=1).item()

    


class agent:
    def __init__(self, state_dim, action_dim, 
                lr,betas,gamma,  memory_size, batch_size, name,
                n_latent_var, update_iter=2000, 
                max_episode=10000, end_timestep=-1, exploration_mode="thompson",
                device=None, TEMPERATURE=1, WITHENTROPY=False):
        self.random= Random()
        self.random.seed(SEED)
        self.lr=lr
        self.softmaxTemperature=TEMPERATURE
        self.betas=betas
        self.gamma=gamma
        self.batch_size=batch_size
        self.memory_size=memory_size
        if(device is None):
            self.device="cpu"
        else:
            self.device=device
        self.Qnet= dueling_network(state_dim,action_dim, n_latent_var).to(self.device)
        self.target_net=dueling_network(state_dim,action_dim, n_latent_var).to(self.device)
        self.target_net.load_state_dict(self.Qnet.state_dict())
        #self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.Qnet.parameters(), lr=lr, betas=betas) if betas is not None else torch.optim.Adam(self.Qnet.parameters(), lr=lr)
        self.replay_buffer= prioritezed_replay_buffer(self.memory_size, self.batch_size, self.random)
        self.update_iter=update_iter
        self.learn_iter=0
        self.episode=0
        self.max_episode=max_episode
        self.avg_reward=[]
        self.avg_steps=[]
        self.id=name
        self.action_dim=action_dim
        self.timestep=0
        self.end_timestep=end_timestep
        self.exploration_mode= exploration_mode
        self.withEntropy=WITHENTROPY
        if(exploration_mode=="epsilon"):
            self.withEntropy=False
            self.epsilon= 0.99
            self.min_epsilon= 0.05
            self.epsilon_delta = (self.epsilon- self.min_epsilon)/(self.max_episode//1.2)
        print("DDQN INITIALISED. Exploration is {}, uses entropy {} lr is {}\n Update target network every {}".format(exploration_mode, self.withEntropy, self.lr, self.update_iter))
        if(self.withEntropy):
            self.alpha= 1.2
            self.min_alpha=5e-3
            self.alpha_delta= (self.alpha - self.min_alpha) / (self.max_episode//2)
            print("Entropy alpha is set to {}, anneals of {} per episode,  minimum to {}".format(self.alpha, self.alpha_delta, self.min_alpha))

    def get_action(self,state, training=False):
        if not training:
            return self.Qnet.get_action(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        if self.exploration_mode=="thompson":
            action_probs=self.Qnet.act(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            action_probs/= self.softmaxTemperature
            entropy= self.estimate_entropy(action_probs)
            list_action_probs=action_probs[0].tolist()
            return self.random.choices(list(range(len(list_action_probs))), k=1, weights=list_action_probs)[0],entropy.item()
        elif self.exploration_mode=="epsilon":
            a= self.Qnet.get_action(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            action_probs=self.Qnet.act(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            entropy= self.estimate_entropy(action_probs)
            if(self.random.uniform(0,1)<self.epsilon):
                a= self.random.randint(0,self.action_dim-1)
            else:
                return a, entropy.item()
    
    def _get_qval(self, state):
        return self.Qnet(torch.FloatTensor(state).unsqueeze(0).to(self.device))
    
    def observe(self,state,action, reward,next_state,done):
        self.replay_buffer.push((state, action, reward,next_state, done))
    
    def train(self):
        batch,batch_indices,batch_weights=self.replay_buffer.sample()
        if(len(batch)==0): print("not enough samples collected"); return
        batch_t=transpose(batch)
        batch_state = torch.FloatTensor(np.array(batch_t[0])).to(self.device)
        batch_action = torch.tensor(np.array(batch_t[1])).to(self.device).unsqueeze(1)
        batch_reward = torch.FloatTensor(np.array(batch_t[2])).to(self.device)
        batch_done = torch.FloatTensor(np.array(batch_t[4])).to(self.device)
        batch_next_state = torch.FloatTensor(np.array(batch_t[3])).to(self.device)
        batch_weights= torch.FloatTensor(np.array(batch_weights)).to(self.device)
        with torch.no_grad(): 
            qmax_action= self.Qnet(batch_next_state).max(1)[1].unsqueeze(-1)
            target_q_next= self.target_net(batch_next_state)
        if(self.withEntropy):
            y = batch_reward + (1 - batch_done) * self.gamma* (target_q_next.gather(1, qmax_action).squeeze(-1).detach() + self.estimate_entropy(target_q_next))#heere add entropy...
        else:
            y=batch_reward + (1 - batch_done) * self.gamma* (target_q_next.gather(1, qmax_action).squeeze(-1).detach())
        batch_qvals=self.Qnet(batch_state)
        loss= (batch_qvals.gather(1,batch_action).squeeze(-1) - y )**2 #compute by hand to obtain sample-related loss
        loss_v=batch_weights * loss
        composed_loss=loss_v
        self.optimizer.zero_grad()
        composed_loss.mean().backward()
        self.optimizer.step()
        #last, we update the priorites based on the local-loss of each sample
        self.replay_buffer.update_priorities(batch_indices, (loss_v + 1e-5).data.cpu().numpy())
        self.learn_iter+=1
        if(self.learn_iter%self.update_iter==0):
            print("Updating target network.")
            self.target_net.load_state_dict(self.Qnet.state_dict())       
        return composed_loss.mean().tolist()

    def estimate_entropy(self, batch_qvals):
        return Categorical(batch_qvals).entropy()

    def increment_episode(self):
        self.episode+=1
        if(self.exploration_mode=="epsilon"):
            self.epsilon-=self.epsilon_delta
            self.epsilon= max(self.epsilon, self.min_epsilon)
        if(self.withEntropy):
            self.alpha=  max(self.min_alpha, self.alpha - self.alpha_delta)
            
    def get_loss(self, state, action, reward,state1,done):
        batch_state = torch.FloatTensor(state).to(self.device)
        batch_action = torch.tensor(action).to(self.device).unsqueeze(1)#.unsqueeze(0)
        batch_reward = torch.tensor(reward).to(self.device)#.unsqueeze(0)
        if(type(done) == int or type(done)==bool):
            done=[done]
        batch_done = torch.FloatTensor(done).to(self.device)#.unsqueeze(0)
        batch_next_state = torch.FloatTensor(state1).to(self.device)
        with torch.no_grad(): 
            qmax_action= self.Qnet(batch_next_state).max(1)[1].unsqueeze(-1)
            target_q_next= self.target_net(batch_next_state)
            y = batch_reward + (1 - batch_done) * self.gamma* target_q_next.gather(1, qmax_action).squeeze(-1).detach()
            loss= (self.Qnet(batch_state).gather(1,batch_action).squeeze(-1) - y )**2 
        
        #compute by hand to obtain sample-related loss
        return loss.tolist()

    def get_advice(self, state, which="adv"):
        """
            interface to access value or adventage calculator.
        """
        if(which== "adv"):
            return self.Qnet._get_adv_estimation(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        elif(which=="val"):
            return self.Qnet._get_stateValue(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        raise Exception("please define which advice (<adv> or <val>)  you want...")


    def learn_from_samples(self, batch):
        batch_action=[]; batch_done=[]; batch_state=[]; batch_next_state=[]; batch_reward=[]
        batch_state, batch_action, batch_reward, batch_next_state, batch_done,_,_=transpose(batch)
        batch_state = torch.FloatTensor(batch_state).to(self.device)
        batch_action = torch.tensor(batch_action).to(self.device).unsqueeze(1)
        batch_reward = torch.FloatTensor(batch_reward).to(self.device)
        batch_done = torch.FloatTensor(batch_done).to(self.device)
        
        with torch.no_grad(): 
            batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)
            qmax_action= self.Qnet(batch_next_state).max(1)[1].unsqueeze(-1)
            target_q_next= self.target_net(batch_next_state)

        y = batch_reward + (1 - batch_done) * self.gamma* target_q_next.gather(1, qmax_action).squeeze(-1).detach()
        loss= (self.Qnet(batch_state).gather(1,batch_action).squeeze(-1) - y )**2 #compute by hand to obtain sample-related loss
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        return loss.mean()


    def load(self, path_state_dict):
        self.Qnet.load_state_dict(torch.load(path_state_dict, map_location=self.device))
        self.Qnet.eval()
    

    def save(self, filename, path=None):
        if path is None:
            torch.save(self.Qnet.state_dict(), './{}.pth'.format(filename))
        else:
            torch.save(self.Qnet.state_dict(), '{}/{}.pth'.format(path,filename))
