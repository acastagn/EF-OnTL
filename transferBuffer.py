from collections import deque
import numpy as np
import pandas as pd
from utils import transpose

class buffer(deque):
    def __init__(self, memory_size, id, with_n_step_return, store_path=None, tocsv=False):
        assert(bool== type(with_n_step_return))
        self.with_n_step_return=with_n_step_return
        self.memory_size= memory_size
        self.id= id
        self.tocsv=tocsv
        if(store_path is not None and store_path[-1]!="/"):
            store_path+="/"
        self.store_path=store_path
        self.column_names=["State", "Action", "Reward", "Next_state", "Terminal", "Uncertainty", "N_step_return"]
        self.idxs_mapper= {k:v for k,v in zip(self.column_names, range(len(self.column_names)))} 

    def push(self,   state, action, reward, next_state,  terminal,  uncertainty, n_step_return=None):
        if(self.with_n_step_return):
            interaction=[state, action, reward, next_state,  terminal, uncertainty, n_step_return]
        else:
            interaction=[state, action, reward, next_state,  terminal, uncertainty]
        self.append(interaction)
        if(len(self)>self.memory_size):
            _=self.popleft()

    #def get_uncertainty(self, interaction):
    #    return interaction[self.idxs_mapper["Uncertainty"]]

    def get_avg_uncertainty(self):
        return np.mean(transpose(self)[self.idxs_mapper["Uncertainty"]])

    def get_all_uncertainties(self):
        return transpose(self)[self.idxs_mapper["Uncertainty"]]

    def get_rl_interactions(self):
        t = transpose(self)
        toret=t[self.idxs_mapper["State"]:self.idxs_mapper["Uncertainty"]]
        if(not self.with_n_step_return):
            return toret
        toret.append(t[self.idxs_mapper["N_step_return"]])
        return toret

    def get_sars(self):
        t = transpose(self)
        toret=[]
        for k in self.column_names:
            idx=self.idxs_mapper[k]
            if(k!="Terminal" and k!="Uncertainty" and k!= "N_step_return"):
                toret.append(t[idx])
        return toret


    def get_all_states(self):
        t = transpose(self)
        return t[self.idxs_mapper["State"]]

    def to_dataframe(self):
        return pd.DataFrame(self, columns=self.column_names)

    def store_for_transfer(self):
        df= self.to_dataframes()
        storing_path= self.store_path+self.id
        if(self.tocsv):
            df.to_csv(storing_path+".csv")
        else:
            df.to_pickle(storing_path+".pkl")