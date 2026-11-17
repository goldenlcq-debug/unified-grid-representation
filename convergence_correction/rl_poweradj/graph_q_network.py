"""Shared candidate scorer: parameter count is independent of grid size."""
import numpy as np
import torch
from torch import nn

def mlp(i,o):
    return nn.Sequential(nn.Linear(i,o),nn.ReLU(),nn.Linear(o,o),nn.ReLU())

class CandidateGraphQ(nn.Module):
    def __init__(self, width=128, layers=2):
        super().__init__()
        if width < 1 or layers < 1:
            raise ValueError('Positive width/layers required')
        self.width,self.depth = width,layers
        self.node = mlp(192+8,width)
        self.edge = mlp(192+5,width)
        self.messages = nn.ModuleList(mlp(2*width,width) for _ in range(layers))
        self.updates = nn.ModuleList(mlp(2*width,width) for _ in range(layers))
        self.context = mlp(3072+2*width+5,width)
        self.score = nn.Sequential(mlp(3*width+5+2,width),nn.Linear(width,1))

    def forward(self, obs, chunk_size=1024):
        obs.validate()
        if chunk_size < 1: raise ValueError('Positive chunk size required')
        device = next(self.parameters()).device
        def tensor(a, dtype=torch.float32):
            # Pickle/deepcopy may change NumPy strides. Canonical contiguous
            # layout keeps live and checkpoint-restored kernels identical.
            return torch.tensor(np.ascontiguousarray(a),dtype=dtype,device=device)
        h = self.node(torch.cat((tensor(obs.graph.nodes),tensor(obs.nodes)),1))
        e = self.edge(torch.cat((tensor(obs.graph.edges),tensor(obs.edges)),1))
        index = tensor(obs.graph.edge_index,torch.long)
        active = tensor((obs.edges[:,0] == 1) & (obs.edges[:,1] == 0),torch.bool)
        src = torch.cat((index[0,active],index[1,active]))
        dst = torch.cat((index[1,active],index[0,active]))
        edge = torch.cat((e[active],e[active]))
        for message,update in zip(self.messages,self.updates):
            values = message(torch.cat((h[src],edge),1))
            agg = torch.zeros_like(h).index_add(0,dst,values)
            degree = torch.zeros(len(h),device=device).index_add(0,dst,torch.ones(len(dst),device=device))
            h = h + update(torch.cat((h,agg/degree.clamp_min(1)[:,None]),1))
        context = self.context(torch.cat((tensor(obs.graph.globals).flatten(),h.mean(0),h.max(0).values,tensor(obs.globals))))
        ids = tensor(obs.candidate_edges,torch.long)
        if not len(ids): return h.sum().reshape(1)[:0]
        targets = torch.nn.functional.one_hot(tensor(obs.targets,torch.long),2).float()
        dynamic = tensor(obs.edges)
        chunks = []
        for start in range(0,len(ids),chunk_size):
            ix = ids[start:start+chunk_size]
            z = torch.cat((h[index[0,ix]],e[ix],context.expand(len(ix),-1),dynamic[ix],targets[start:start+chunk_size]),1)
            chunks.append(self.score(z).squeeze(1))
        return torch.cat(chunks)

    def forward_batch(self, observations):
        # Micrographs retain separate argmax domains; never mix candidates.
        values = [self(o) for o in observations]
        ptr = np.cumsum([0]+[len(o.targets) for o in observations])
        return torch.cat(values) if values else next(self.parameters()).new_empty(0), ptr
