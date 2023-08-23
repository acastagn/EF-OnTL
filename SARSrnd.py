import numpy as np
import torch
import torch.nn as nn
import dill as pickle
import os

from utils import mkdir

RND_ENCODER_FILENAME="rnd_encoder_w.pth"
PRED_NET_FILENAME="pred_network_w.pth"
RND_PARAMS_FILENAME="RND_parameters.pkl"


class net(nn.Module):
    def __init__(self, n_s, n_a, rew_dim, s_encode_size,n_latent_var=[], act=nn.ReLU()):
        assert(type(n_latent_var)==list), "n_latenr_var is not a list."
        super(net, self).__init__()
        _layers= np.concatenate(([n_s*2+rew_dim+n_a], n_latent_var, [s_encode_size]))
        self.layers=nn.ModuleList(nn.Linear(int(_layers[idx]), int(_layers[idx+1])) for idx in range(0, len(_layers)-1))
        self.act=act

    def forward(self, x):
        for l in self.layers[:-1]:
            x= self.act(l(x))
        return self.layers[-1](x)



"""

@param:  minibatch_size if != 1 the network has a delay into estimating the updated uncertainty but the training will be faster.


"""
class UE():
    def __init__(self, n_s,n_a, rew_dim=1, lr=1e-3, id="SARS-rnd", s_encode_size=1024,n_latent_var=[], device= None, minibatch_size=1):
        self.n_s = n_s
        self.n_a = n_a
        self.rew_dim= rew_dim
        self.lr = lr
        self.s_encode_size = s_encode_size
        self.id=id
        self.n_latent_var=n_latent_var
        self.temp_batch=[]
        self.minibatch_size=minibatch_size
        self.device=device    
        self._initialize()

    def get_uncertainty(self,s,a, r, n_s):
        if(type(a)!= list):
            a=[a]
        if(type(r)!= list):
            r=[r]
        x=np.concatenate( (s,a,r,n_s) )
        return self._compute_uncertainty(x).mean().tolist()
    
    def _compute_uncertainty(self, x):
        x=torch.FloatTensor(np.array(x)).to(self.device)
        with torch.no_grad():
            y1=self.rnd_net(x)
        y2=self.pred_net(x)
        return ((y2-y1)**2)
    
    def learn(self, s,a, r, n_s):
        if(type(a)!= list):
            a=[a]
        if(type(r)!= list):
            r=[r]
        x=np.concatenate( (s,a,r,n_s) )
        self.temp_batch.append(x)
        if(len(self.temp_batch)<self.minibatch_size):
            with(torch.no_grad()):
                loss=self._compute_uncertainty(x)
            return loss.mean().tolist()
        complessive_loss=self._get_multi_uncertainty(self.temp_batch)
        self.optimizer.zero_grad()
        complessive_loss.mean().backward()
        self.optimizer.step()
        self.temp_batch=[]
        return complessive_loss[-1].tolist()

    def get_uncertainty_from_batch(self, s,a, r, n_s):
        x=np.array([np.concatenate( (s[i],[ a[i], r[i], n_s[i]])) for i in range(len(a))])
        return self._get_multi_uncertainty(x).tolist()

    def _get_multi_uncertainty(self, x):
        return self._compute_uncertainty(x).mean(axis=-1)

    def save(self, path, subfolderName="SARSrnd"):
        if (path[-1]!="/"):
            path+="/"
        path+= subfolderName+"/"
        mkdir(path)
        to_store= self._store_params()
        with open(path+RND_PARAMS_FILENAME, 'wb') as pickle_file:
                    pickle.dump(to_store, pickle_file)
        #store models
        torch.save(self.rnd_net.state_dict(),path+RND_ENCODER_FILENAME)
        torch.save(self.pred_net.state_dict(),path+PRED_NET_FILENAME)
    
    def _store_params(self):
        to_store={}
        to_store["n_s"]=self.n_s
        to_store["n_a"]= self.n_a
        to_store["rew_dim"] = self.rew_dim
        to_store["lr"]=self.lr 
        to_store["s_encode_size"]=self.s_encode_size
        to_store["id"]=self.id
        to_store["n_latent_var"]= self.n_latent_var
        return to_store
    
    def _load(self, path):
        path+="SARSrnd/"
        if(not os.path.isdir(path)):
            raise Exception("\n\t\t Cannot find SARSrnd main folder to restore files.")
        files_needed=[RND_ENCODER_FILENAME,PRED_NET_FILENAME,RND_PARAMS_FILENAME]
        for fname in files_needed:
            if(not os.path.isfile(path+fname)):
                raise Exception("\n\t\t Not found: {} \t at {}".format(fname, path)) 
        #RESTORE TENSORS AND PARAMS
        self._restore_fields(path)
        self._initialize()
        self.rnd_net.load_state_dict(torch.load(path+RND_ENCODER_FILENAME))
        self.pred_net.load_state_dict(torch.load(path+PRED_NET_FILENAME))
        self.rnd_net.eval()
        print("SARSrnd restored")

    def _restore_fields(self, path):
        with open(path+RND_PARAMS_FILENAME, 'rb') as pickle_file:
            loaded = pickle.load(pickle_file)
        for key in loaded:
            try:
                setattr(self, key, loaded[key])
            except:
                print("impossible to set key {} with value {}".format(key,loaded[key]))
                pass
    
    def _initialize(self):
        self.rnd_net= net(self.n_s, self.n_a, self.rew_dim, s_encode_size=self.s_encode_size, n_latent_var=self.n_latent_var).to(self.device)
        self.rnd_net.eval()
        self.pred_net= net(self.n_s, self.n_a, self.rew_dim, s_encode_size=self.s_encode_size, n_latent_var=self.n_latent_var).to(self.device)
        self.optimizer = torch.optim.RMSprop(params=self.pred_net.parameters(), lr=self.lr)

        
    
