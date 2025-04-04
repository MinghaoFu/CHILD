"""Prior Network"""
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from torch.func import jacfwd, vmap
from .mlp import NLayerLeakyMLP, NLayerLeakyNAC
from .base import GroupLinearLayer
import ipdb as pdb

class NPTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        num_layers=3, 
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=lags*latent_size+1, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
    
    def forward(self, x, masks=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = [ ]

        hist_jac = []

        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((yy, xx[:,:,i]),dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((yy*mask, xx[:,:,i]),dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
            #     pdd = jacobian(self.gs[i], inputs, create_graph=True, vectorize=True)
            # # Determinant of low-triangular mat is product of diagonal entries
            # logabsdet = torch.log(torch.abs(torch.diag(pdd[:,0,:,-1])))
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            logabsdet = torch.log(torch.abs(pdd[:,0,-1]))
            hist_jac.append(torch.unsqueeze(pdd[:,0,:-1], dim=1))
            # replaced with faster jac (up)
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac

class NPChangeTransitionPrior(nn.Module):


    def __init__(
        self, 
        lags, 
        latent_size,
        embedding_dim, 
        num_layers=3,
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=hidden_dim+lags*latent_size+1, 
                             out_features=1, 
                             num_layers=0, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
        self.fc = NLayerLeakyMLP(in_features=embedding_dim,
                                 out_features=hidden_dim,
                                 num_layers=2,
                                 hidden_dim=hidden_dim)

    def forward(self, x, embeddings, masks=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        # embeddings: [BS, embed_dims]
        batch_size, length, input_dim = x.shape
        # import pdb
        # pdb.set_trace()
        embeddings = self.fc(embeddings)
        embeddings = embeddings.unsqueeze(1).repeat(1, length-self.L, 1).reshape(-1, embeddings.shape[-1])
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((embeddings, yy, xx[:,:,i]),dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((embeddings, yy*mask, xx[:,:,i]),dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                # pdd = jacobian(self.gs[i], inputs, create_graph=True, vectorize=True)
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            # Determinant of low-triangular mat is product of diagonal entries
            logabsdet = torch.log(torch.abs(pdd[:,0,-1]))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian
    
class NPChangeInstantaneousTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size,
        embedding_dim, 
        num_layers=3,
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=hidden_dim+lags*latent_size+1+i, 
                             out_features=1, 
                             num_layers=0, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
        self.fc = NLayerLeakyMLP(in_features=embedding_dim,
                                 out_features=hidden_dim,
                                 num_layers=2,
                                 hidden_dim=hidden_dim)

    def forward(self, x, embeddings, alphas):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        # embeddings: [BS, embed_dims]
        batch_size, length, input_dim = x.shape
        embeddings = self.fc(embeddings)
        embeddings = embeddings.unsqueeze(1).repeat(1, length-self.L, 1).reshape(-1, embeddings.shape[-1])
        # prepare data
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        # get residuals and |J|
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            inputs = torch.cat([xx[:,:,j] * alphas[i][j] for j in range(i)] + [embeddings, yy, xx[:,:,i]], dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            # Determinant: product of diagonal entries, sum of last entry
            logabsdet = torch.log(torch.abs(pdd[:,0,-1]))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian
 
class NPInstantaneousTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        num_layers=3, 
        hidden_dim=64,
        z_dim_list=[1, 2, 4]
        ):
        super().__init__()
        self.L = lags
        self.z_dim_list = z_dim_list
        self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        # gs = [NLayerLeakyMLP(in_features=lags*latent_size+1+i, 
        #                      out_features=1, 
        #                      num_layers=num_layers, 
        #                      hidden_dim=hidden_dim) for i in range(latent_size)]

        # gs = [NLayerLeakyMLP(in_features=1+1+1, 
        #                      out_features=1, 
        #                      num_layers=num_layers, 
        #                      hidden_dim=hidden_dim) for i in range(latent_size)]
        
        gs = []
        
        for layer, z_dim in enumerate(z_dim_list):
            for _ in range(z_dim):
                if layer == 0: 
                    gs.append(NLayerLeakyMLP(in_features=1 + z_dim * lags, 
                                    out_features=1, 
                                    num_layers=num_layers, 
                                    hidden_dim=hidden_dim))
                else:
                    gs.append(NLayerLeakyMLP(in_features=1 + z_dim_list[layer-1] + z_dim * lags, 
                                    out_features=1, 
                                    num_layers=num_layers, 
                                    hidden_dim=hidden_dim))

        # gs.append(NLayerLeakyMLP(in_features=1+1+latent_size, 
        #                      out_features=1, 
        #                      num_layers=num_layers, 
        #                      hidden_dim=hidden_dim) )
        # gs.append(NLayerLeakyMLP(in_features=1+1+latent_size, 
        #                      out_features=1, 
        #                      num_layers=num_layers, 
        #                      hidden_dim=hidden_dim) )
        # gs.append(NLayerLeakyMLP(in_features=1+1+latent_size, 
        #                      out_features=1, 
        #                      num_layers=num_layers, 
        #                      hidden_dim=hidden_dim) )
        # gs.append(NLayerLeakyMLP(in_features=1+1+latent_size, 
        #                      out_features=1, 
        #                      num_layers=num_layers, 
        #                      hidden_dim=hidden_dim) )
        
        # gs = [NLayerLeakyMLP(in_features=4, 
        #                      out_features=1, 
        #                      num_layers=num_layers, 
        #                      hidden_dim=hidden_dim)]
        
        # gs.append(NLayerLeakyMLP(in_features=5, 
        #                      out_features=1, 
        #                      num_layers=num_layers, 
        #                      hidden_dim=hidden_dim))
        
        # gs.append(NLayerLeakyMLP(in_features=5, 
        #                      out_features=1, 
        #                      num_layers=num_layers, 
        #                      hidden_dim=hidden_dim))
        
        self.gs = nn.ModuleList(gs)
    
    def forward(self, x):
        # x: bs, t, d
        # x^2: bs, t, [0]
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # prepare data
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        # get residuals and |J|
        residuals = []

        hist_jac = []
        

        sum_log_abs_det_jacobian = 0
        for layer, z_dim in enumerate(self.z_dim_list): 
            # inputs = torch.cat([yy] + [xx[:,:,j] for j in range(i)] + [xx[:,:,i]], dim=-1)
            # inputs = torch.cat([yy[:,:,i]] + [xx[:,:,j] for j in range(i)] + [xx[:,:,i]], dim=-1)
            # layer is the index of i in the z_dim_list
            for j in range(z_dim):
                i = sum(self.z_dim_list[:layer]) + j
                if layer < 1:
                        # inputs = torch.cat([yy[:,:,i]] + [xx[:,:,i]], dim=-1)
                    if self.L == 1:
                        inputs = torch.cat([yy[:, :z_dim]] + [xx[:,:,i]], dim=-1)
                    # else:
                    #     inputs = torch.cat([yy[:, torch.arange(0, yy.shape[1], input_dim)]] + [xx[:, :, i]], dim=-1)
                else:
                    # inputs = torch.cat([yy[:,:,i]] + [xx[:,:,i-1]] + [xx[:,:,i]], dim=-1) 
                    if self.L == 1:
                        inputs = torch.cat([yy[:, sum(self.z_dim_list[:layer]):sum(self.z_dim_list[:layer + 1])]] \
                                           + [xx[:,:, sum(self.z_dim_list[:layer - 1]):sum(self.z_dim_list[:layer])].squeeze(1)] + [xx[:,:,i]], dim=-1)
                    # else:
                    #     lag_z_slice = [i for i in range(self.L*8) if i not in range(0, self.L*8, 8)]
                    #     inputs = torch.cat([yy[:, lag_z_slice]] + [xx[:, :, :4]] + [xx[:, :, i]], dim=-1)
                # if i == 0:
                #     # inputs = torch.cat([torch.unsqueeze(yy[:,i], dim=-1)] + [xx[:,:,i]], dim=-1)
                #     inputs = torch.cat([yy] + [xx[:,:,i]], dim=-1)
                # if i != 0:
                #     # inputs = torch.cat([torch.unsqueeze(yy[:,i], dim=-1)] + [xx[:,:,i]] + [xx[:,:,i-1]], dim=-1)
                #     inputs = torch.cat([yy] + [xx[:,:,i]] + [xx[:,:,i-1]], dim=-1)
                #     pass
                # print(i, inputs.shape)
                residual = self.gs[i](inputs)
                
                with torch.enable_grad():
                    pdd = vmap(jacfwd(self.gs[i]))(inputs)
                # Determinant: product of diagonal entries, sum of last entry
                logabsdet = torch.log(torch.abs(pdd[:,0,-1]))

                # hist_jac.append(torch.unsqueeze(pdd[:,0,:self.L*input_dim], dim=1))
                hist_jac.append(torch.unsqueeze(pdd[:,0,:-1], dim=1))
                # hist_jac.append(torch.unsqueeze(torch.abs(pdd[:,0,:self.L*input_dim]), dim=1))

                sum_log_abs_det_jacobian += logabsdet
                residuals.append(residual)

        # hist_jac = torch.cat(hist_jac, dim=1) # BS * input_dim * (L * input_dim)
        # hist_jac = torch.mean(hist_jac, dim=0) # input_dim * (L * input_dim)
        

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim) # batch x 3
        # print(residuals.shape)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac
