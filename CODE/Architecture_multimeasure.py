import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchsummary
import torch.nn.functional 


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Assuming the input is a flattened 84x84 array, the input features would be 84*84
        self.fc1 = nn.Linear(84*84, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)     # Second fully connected layer
        self.fc3 = nn.Linear(64, 2)       # Final layer that outputs a scalar

    def forward_train(self, x):
        # Flatten the input tensor
        x = x.view(-1, 84*84)
        x = F.relu(self.fc1(x))  # Activation function applied to the output of the first layer
        x = F.relu(self.fc2(x))  # Activation function applied to the output of the second layer
        x = self.fc3(x)          # Output layer
        return x


class SiteEncoding(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.site_1_encoding = nn.Parameter(torch.randn(1, dim), requires_grad=True)
        self.site_2_encoding = nn.Parameter(torch.randn(1, dim), requires_grad=True)
        self.site_3_encoding = nn.Parameter(torch.randn(1, dim), requires_grad=True)
        # self.dim = dim

    def forward(self, z, sitecode):
        """z: latent space of previous encoder """
        sitecode = sitecode.cpu().numpy()
        ##print(sitecode)
        if sitecode == 1:
            ##print("Site code 1", self.site_1_encoding)
            return torch.add(self.site_1_encoding,z)
            # return [self.site_1_encoding[i] + z[i] for i in range(len(z))]
        elif sitecode == 2:
            ##print("Site code 2", self.site_2_encoding)
            return torch.add(self.site_2_encoding,z)
        elif sitecode ==3 :
            ##print("Site code 3", self.site_3_encoding)
            return torch.add(self.site_3_encoding,z)
        else:
            #print("ERRRORRRRRR Incorrect site code provided with data")
            exit

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=10         #128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=10, out_features=512
        )
        self.decoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["input_shape"]
        )
        self.site_integration_layer = SiteEncoding(10)

        self.prediction_layer = nn.Linear(
            in_features=10, out_features=1
        )

    def forward(self, features):
        latent = self.encoder_hidden_layer(features)
        latent = torch.relu(latent)
        latent = self.encoder_output_layer(latent)
        latent = torch.relu(latent)

        #latent_with_site = self.site_integration_layer(latent,sitecode)
        #latent_with_site = torch.FloatTensor(latent_with_site)
        #prediction = self.prediction_layer(latent_with_site)
        prediction = self.prediction_layer(latent)
        #prediction=1

        #reconstructed = self.decoder_hidden_layer(latent)
        reconstructed = self.decoder_hidden_layer(latent)
        reconstructed = torch.relu(reconstructed)
        reconstructed = self.decoder_output_layer(reconstructed)

        return reconstructed, prediction, latent

class CAE(nn.Module):
    def __init__(self, in_dim, z_dim, out_dim):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=in_dim, out_features=in_dim
        )
        self.encoder_output_layer = nn.Linear(
            in_features=in_dim, out_features=z_dim         #128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=z_dim+1, out_features=in_dim
        )
        self.decoder_output_layer = nn.Linear(
            in_features=in_dim, out_features=in_dim
        )

        self.prediction_layer_modularity_site1 = MLP(z_dim, 1, layers=[z_dim, z_dim])
        self.prediction_layer_modularity_site2 = MLP(z_dim, 1, layers=[z_dim, z_dim])
        self.prediction_layer_participation_site1 = MLP(z_dim, 1, layers=[z_dim, z_dim])
        self.prediction_layer_participation_site2 = MLP(z_dim, 1, layers=[z_dim, z_dim])

    def forward(self, features, sites):
        latent = self.encoder_hidden_layer(features)
        latent = torch.relu(latent)
        latent = self.encoder_output_layer(latent)
        latent = torch.relu(latent)
        latent_with_site = torch.cat((latent,sites),axis=1)

        #latent_with_site = self.site_integration_layer(latent,sitecode)
        #latent_with_site = torch.FloatTensor(latent_with_site)
        #prediction = self.prediction_layer(latent_with_site)
        prediction_site1 = self.prediction_layer_site1(latent)
        prediction_site2 = self.prediction_layer_site2(latent)
        #prediction=1

        #reconstructed = self.decoder_hidden_layer(latent)
        reconstructed = self.decoder_hidden_layer(latent_with_site)
        reconstructed = torch.relu(reconstructed)
        reconstructed = self.decoder_output_layer(reconstructed)

        return reconstructed, prediction_site1, prediction_site2, latent

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, layers=[512,512,512]):
        super(MLP, self).__init__()
        self.mlp_list = nn.ModuleList()

        self.mlp_list.append(nn.Linear(in_dim,layers[0]))
        self.mlp_list.append(torch.nn.ReLU())
        #self.mlp_list.append(torch.nn.Tanh())
        for layer_idx in range(len(layers)-1):
            self.mlp_list.append(nn.Linear(layers[layer_idx],layers[layer_idx+1]))
            #self.mlp_list.append(torch.nn.ReLU())
            self.mlp_list.append(torch.nn.ReLU())
        self.mlp_list.append(nn.Linear(layers[-1],out_dim))
        self.mlp = torch.nn.Sequential(*self.mlp_list)

    def forward(self,x):
        return self.mlp(x)

class Conditional_VAE(nn.Module):
    def __init__(self,in_dim, c_dim, z_dim, num_measures):
        super(Conditional_VAE, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        #self.enc = MLP(in_dim, 2*z_dim)
        #self.dec = MLP(z_dim+c_dim, in_dim)
        self.encoder_hidden_layer = nn.Linear(
            in_features=in_dim, out_features=in_dim
        )
        self.encoder_output_layer = nn.Linear(
            in_features=in_dim, out_features=2*z_dim         #128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=z_dim+1, out_features=in_dim
        )
        self.decoder_output_layer = nn.Linear(
            in_features=in_dim, out_features=in_dim
        )
        self.prediction_layer_site1_hidden = nn.Linear(z_dim,z_dim)
        self.prediction_layer_site2_hidden = nn.Linear(z_dim,z_dim)
        #self.prediction_layer_site1 = nn.Linear(z_dim,7)
        #self.prediction_layer_site2 = nn.Linear(z_dim,7)
        self.prediction_layer_site1 = nn.Linear(z_dim,num_measures)
        self.prediction_layer_site2 = nn.Linear(z_dim,num_measures)
        self.prediction_layer_age = nn.Linear(z_dim,1)

        self.prediction_layer_age_hidden = nn.Linear(z_dim,z_dim)
        self.prediction_layer_sex_hidden_1 = nn.Linear(z_dim,50)
        self.prediction_layer_sex_hidden_2 = nn.Linear(50,25)
        self.prediction_layer_sex = nn.Linear(25,2)
        

        
        self.softmax = nn.LogSoftmax(dim=1) 
        self.sig = nn.Sigmoid()

    def to_z_params(self,x):
        #z = self.enc(x)
        z = self.encoder_hidden_layer(x)
        z = torch.relu(z)
        z = self.encoder_output_layer(z)
        z = torch.relu(z)
        ##print("z shape",z.shape)
        #z_mu = torch.tanh(z[...,:self.z_dim])
        z_mu = z[...,:self.z_dim]
        ##print("zmu shape",z_mu.shape)
        z_log_sigma_sq = z[...,self.z_dim:]
        ##print("z log sigma shape",z_log_sigma_sq.shape)
        return z_mu, z_log_sigma_sq

    def to_z(self, z_mu, z_log_sigma_sq):
        std = torch.exp(0.5 * z_log_sigma_sq)
        eps = torch.randn_like(std)
        z = eps * std + z_mu
        #z = z_mu
        return z


    def forward_train(self,x,c=None,cat_order=0):
        ##print(x.size())
        z_mu, z_log_sigma_sq = self.to_z_params(x)
        ##print(z_mu.size(), x.size())
        z = self.to_z(z_mu, z_log_sigma_sq)
        ##print("xshape", x.shape, "cshape" ,c.shape)
        #z_withsite = self.site_integration_layer(z, site)
        #prediction = self.prediction_layer(z)
        #out = self.dec(z)
        z = torch.relu(z)
        prediction_site1 = self.prediction_layer_site1(self.prediction_layer_site1_hidden(self.prediction_layer_site1_hidden(z)))
        prediction_site2 = self.prediction_layer_site2(self.prediction_layer_site2_hidden(self.prediction_layer_site2_hidden(z)))
        prediction_age = self.prediction_layer_age(self.prediction_layer_age_hidden(self.prediction_layer_age_hidden(z)))
        
        
        prediction_sex = self.prediction_layer_sex(F.relu(self.prediction_layer_sex_hidden_2(F.relu(self.prediction_layer_sex_hidden_1(z)))))
        #prediction_sex = torch.relu(self.softmax(torch.relu(self.prediction_layer_sex(self.prediction_layer_sex_hidden(torch.relu(self.prediction_layer_sex_hidden((z))))))))
        
        #prediction_sex = torch.sigmoid(prediction_sex)
        out = self.decoder_hidden_layer(torch.cat((z,c),axis=1))
        out = self.decoder_output_layer(out)

        """
        prediction_site1_modularity = self.prediction_layer_site1_modularity(self.prediction_layer_site1_hidden_modularity(self.prediction_layer_site1_hidden_modularity(z)))
        prediction_site2_modularity = self.prediction_layer_site2_modularity(self.prediction_layer_site2_hidden_modularity(self.prediction_layer_site2_hidden_modularity(z)))
        prediction_site1_participation = self.prediction_layer_site1_participation(self.prediction_layer_site1_hidden_participation(self.prediction_layer_site1_hidden_participation(z)))
        prediction_site2_participation = self.prediction_layer_site2_participation(self.prediction_layer_site2_hidden_participation(self.prediction_layer_site2_hidden_participation(z)))
        prediction_site1_clustering = self.prediction_layer_site1_clustering(self.prediction_layer_site1_hidden_clustering(self.prediction_layer_site1_hidden_clustering(z)))
        prediction_site2_clustering = self.prediction_layer_site2_clustering(self.prediction_layer_site2_hidden_clustering(self.prediction_layer_site2_hidden_clustering(z)))
        prediction_site1_strengths = self.prediction_layer_site1_strengths(self.prediction_layer_site1_hidden_strengths(self.prediction_layer_site1_hidden_strengths(z)))
        prediction_site2_strengths = self.prediction_layer_site2_strengths(self.prediction_layer_site2_hidden_strengths(self.prediction_layer_site2_hidden_strengths(z)))
        prediction_site1_density = self.prediction_layer_site1_density(self.prediction_layer_site1_hidden_density(self.prediction_layer_site1_hidden_density(z)))
        prediction_site2_density = self.prediction_layer_site2_density(self.prediction_layer_site2_hidden_density(self.prediction_layer_site2_hidden_density(z)))

        out = self.decoder_hidden_layer(torch.cat((z,c),axis=1))
        out = self.decoder_output_layer(out)
        #out = self.dec(torch.cat((z,c),axis=1))
        prediction_site1 = torch.hstack((prediction_site1_modularity, prediction_site1_participation, prediction_site1_clustering, prediction_site1_strengths, prediction_site1_density))
        prediction_site2 = torch.hstack((prediction_site2_modularity, prediction_site2_participation, prediction_site2_clustering, prediction_site2_strengths, prediction_site2_density))
        """
        return out, z, z_mu, z_log_sigma_sq, prediction_site1, prediction_site2, prediction_age, prediction_sex



class Conditional_VAE_3sites(nn.Module):
    def __init__(self,in_dim, c_dim, z_dim, num_measures):
        super(Conditional_VAE_3sites, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        #self.enc = MLP(in_dim, 2*z_dim)
        #self.dec = MLP(z_dim+c_dim, in_dim)
        self.encoder_hidden_layer = nn.Linear(
            in_features=in_dim, out_features=in_dim
        )
        self.encoder_output_layer = nn.Linear(
            in_features=in_dim, out_features=2*z_dim         #128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=z_dim+1, out_features=in_dim
        )
        self.decoder_output_layer = nn.Linear(
            in_features=in_dim, out_features=in_dim
        )
        self.prediction_layer_site1_hidden = nn.Linear(z_dim,z_dim)
        self.prediction_layer_site2_hidden = nn.Linear(z_dim,z_dim)
        self.prediction_layer_site3_hidden = nn.Linear(z_dim,z_dim)
        #self.prediction_layer_site1 = nn.Linear(z_dim,7)
        #self.prediction_layer_site2 = nn.Linear(z_dim,7)
        self.prediction_layer_site1 = nn.Linear(z_dim,num_measures)
        self.prediction_layer_site2 = nn.Linear(z_dim,num_measures)
        self.prediction_layer_site3 = nn.Linear(z_dim,num_measures)
        self.prediction_layer_age = nn.Linear(z_dim,1)

        self.prediction_layer_age_hidden = nn.Linear(z_dim,z_dim)
        self.prediction_layer_sex_hidden_1 = nn.Linear(z_dim,50)
        self.prediction_layer_sex_hidden_2 = nn.Linear(50,25)
        self.prediction_layer_sex = nn.Linear(25,2)
        

        
        self.softmax = nn.LogSoftmax(dim=1) 
        self.sig = nn.Sigmoid()

    def to_z_params(self,x):
        #z = self.enc(x)
        z = self.encoder_hidden_layer(x)
        z = torch.relu(z)
        z = self.encoder_output_layer(z)
        z = torch.relu(z)
        ##print("z shape",z.shape)
        #z_mu = torch.tanh(z[...,:self.z_dim])
        z_mu = z[...,:self.z_dim]
        ##print("zmu shape",z_mu.shape)
        z_log_sigma_sq = z[...,self.z_dim:]
        ##print("z log sigma shape",z_log_sigma_sq.shape)
        return z_mu, z_log_sigma_sq

    def to_z(self, z_mu, z_log_sigma_sq):
        std = torch.exp(0.5 * z_log_sigma_sq)
        eps = torch.randn_like(std)
        z = eps * std + z_mu
        #z = z_mu
        return z


    def forward_train(self,x,c=None,cat_order=0):
        ##print(x.size())
        z_mu, z_log_sigma_sq = self.to_z_params(x)
        ##print(z_mu.size(), x.size())
        z = self.to_z(z_mu, z_log_sigma_sq)
        ##print("xshape", x.shape, "cshape" ,c.shape)
        #z_withsite = self.site_integration_layer(z, site)
        #prediction = self.prediction_layer(z)
        #out = self.dec(z)
        z = torch.relu(z)
        prediction_site1 = self.prediction_layer_site1(self.prediction_layer_site1_hidden(self.prediction_layer_site1_hidden(z)))
        prediction_site2 = self.prediction_layer_site2(self.prediction_layer_site2_hidden(self.prediction_layer_site2_hidden(z)))
        prediction_site3 = self.prediction_layer_site3(self.prediction_layer_site3_hidden(self.prediction_layer_site3_hidden(z)))
        prediction_age = self.prediction_layer_age(self.prediction_layer_age_hidden(self.prediction_layer_age_hidden(z)))
        
        
        prediction_sex = self.prediction_layer_sex(F.relu(self.prediction_layer_sex_hidden_2(F.relu(self.prediction_layer_sex_hidden_1(z)))))
        #prediction_sex = torch.relu(self.softmax(torch.relu(self.prediction_layer_sex(self.prediction_layer_sex_hidden(torch.relu(self.prediction_layer_sex_hidden((z))))))))
        
        #prediction_sex = torch.sigmoid(prediction_sex)
        out = self.decoder_hidden_layer(torch.cat((z,c),axis=1))
        out = self.decoder_output_layer(out)

        """
        prediction_site1_modularity = self.prediction_layer_site1_modularity(self.prediction_layer_site1_hidden_modularity(self.prediction_layer_site1_hidden_modularity(z)))
        prediction_site2_modularity = self.prediction_layer_site2_modularity(self.prediction_layer_site2_hidden_modularity(self.prediction_layer_site2_hidden_modularity(z)))
        prediction_site1_participation = self.prediction_layer_site1_participation(self.prediction_layer_site1_hidden_participation(self.prediction_layer_site1_hidden_participation(z)))
        prediction_site2_participation = self.prediction_layer_site2_participation(self.prediction_layer_site2_hidden_participation(self.prediction_layer_site2_hidden_participation(z)))
        prediction_site1_clustering = self.prediction_layer_site1_clustering(self.prediction_layer_site1_hidden_clustering(self.prediction_layer_site1_hidden_clustering(z)))
        prediction_site2_clustering = self.prediction_layer_site2_clustering(self.prediction_layer_site2_hidden_clustering(self.prediction_layer_site2_hidden_clustering(z)))
        prediction_site1_strengths = self.prediction_layer_site1_strengths(self.prediction_layer_site1_hidden_strengths(self.prediction_layer_site1_hidden_strengths(z)))
        prediction_site2_strengths = self.prediction_layer_site2_strengths(self.prediction_layer_site2_hidden_strengths(self.prediction_layer_site2_hidden_strengths(z)))
        prediction_site1_density = self.prediction_layer_site1_density(self.prediction_layer_site1_hidden_density(self.prediction_layer_site1_hidden_density(z)))
        prediction_site2_density = self.prediction_layer_site2_density(self.prediction_layer_site2_hidden_density(self.prediction_layer_site2_hidden_density(z)))

        out = self.decoder_hidden_layer(torch.cat((z,c),axis=1))
        out = self.decoder_output_layer(out)
        #out = self.dec(torch.cat((z,c),axis=1))
        prediction_site1 = torch.hstack((prediction_site1_modularity, prediction_site1_participation, prediction_site1_clustering, prediction_site1_strengths, prediction_site1_density))
        prediction_site2 = torch.hstack((prediction_site2_modularity, prediction_site2_participation, prediction_site2_clustering, prediction_site2_strengths, prediction_site2_density))
        """
        return out, z, z_mu, z_log_sigma_sq, prediction_site1, prediction_site2, prediction_site3, prediction_age, prediction_sex

class Conditional_VAE_v2(nn.Module):
    def __init__(self,in_dim, c_dim, z_dim):
        super(Conditional_VAE, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        #self.enc = MLP(in_dim, 2*z_dim)
        #self.dec = MLP(z_dim+c_dim, in_dim)
        self.encoder_hidden_layer = nn.Linear(
            in_features=in_dim, out_features=in_dim
        )
        self.encoder_output_layer = nn.Linear(
            in_features=in_dim, out_features=2*z_dim         #128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=z_dim+1, out_features=in_dim
        )
        self.decoder_output_layer = nn.Linear(
            in_features=in_dim, out_features=in_dim
        )
        self.prediction_layer_site1_hidden = nn.Linear(z_dim,z_dim)
        self.prediction_layer_site2_hidden = nn.Linear(z_dim,z_dim)
        self.prediction_layer_site1 = nn.Linear(z_dim,1)
        self.prediction_layer_site2 = nn.Linear(z_dim,1)
        #self.prediction_layer_site1 = MLP(z_dim, 1, layers=[z_dim, z_dim])
        #self.prediction_layer_site2 = MLP(z_dim, 1, layers=[z_dim,z_dim])
        #self.prediction_layer = nn.Linear(
        #    in_features=z_dim+c_dim, out_features=1
        #)
        self.site_integration_layer = SiteEncoding(z_dim)

    def to_z_params(self,x):
        #z = self.enc(x)
        z = self.encoder_hidden_layer(x)
        z = torch.relu(z)
        z = self.encoder_output_layer(z)
        z = torch.relu(z)
        ##print("z shape",z.shape)
        #z_mu = torch.tanh(z[...,:self.z_dim])
        z_mu = z[...,:self.z_dim]
        ##print("zmu shape",z_mu.shape)
        z_log_sigma_sq = z[...,self.z_dim:]
        ##print("z log sigma shape",z_log_sigma_sq.shape)
        return z_mu, z_log_sigma_sq

    def to_z(self, z_mu, z_log_sigma_sq):
        std = torch.exp(0.5 * z_log_sigma_sq)
        eps = torch.randn_like(std)
        z = eps * std + z_mu
        #z = z_mu
        return z


    def forward_train(self,x,c=None,cat_order=0):
    # predict CPL and Modularity
        ##print(x.size())
        z_mu, z_log_sigma_sq = self.to_z_params(x)
        ##print(z_mu.size(), x.size())
        z = self.to_z(z_mu, z_log_sigma_sq)
        ##print("xshape", x.shape, "cshape" ,c.shape)
        #z_withsite = self.site_integration_layer(z, site)
        #prediction = self.prediction_layer(z)
        #out = self.dec(z)
        z = torch.relu(z)

        prediction_site1_cpl = self.prediction_layer_site1(self.prediction_layer_site1_hidden(self.prediction_layer_site1_hidden(z)))
        prediction_site2_cpl = self.prediction_layer_site2(self.prediction_layer_site2_hidden(self.prediction_layer_site2_hidden(z)))
        prediction_site1_mod = self.prediction_layer_site1(self.prediction_layer_site1_hidden(self.prediction_layer_site1_hidden(z)))
        prediction_site2_mod = self.prediction_layer_site2(self.prediction_layer_site2_hidden(self.prediction_layer_site2_hidden(z)))

        out = self.decoder_hidden_layer(torch.cat((z,c),axis=1))
        out = self.decoder_output_layer(out)
        #out = self.dec(torch.cat((z,c),axis=1))

        return out, z, z_mu, z_log_sigma_sq, prediction_site1_cpl, prediction_site2_cpl, prediction_site1_mod, prediction_site2_mod
