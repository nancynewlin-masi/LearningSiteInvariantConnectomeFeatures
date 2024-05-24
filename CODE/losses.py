

import torch


def kl_loss_gaussians( mu, log_sigma_sq, eps=1e-8):

    kl_loss_term1 = -(mu * mu)
    kl_loss_term2 = -(log_sigma_sq.exp())
    kl_loss_term3 = 1 + log_sigma_sq
    print(kl_loss_term1.shape, kl_loss_term2.shape, kl_loss_term3.shape)
    kl_loss = -0.5 * (kl_loss_term1 + kl_loss_term2 + kl_loss_term3)#.sum(axis=0)

    #TODO: axes check
    return kl_loss.mean()


def all_pairs_gaussian_kl(mu, log_sigma_sq, dim_z):

    sigma_sq = log_sigma_sq.exp()
    sigma_sq_inv = 1.0 / sigma_sq

    first_term = torch.matmul(sigma_sq, sigma_sq_inv.transpose(0,1))

    r = torch.matmul( mu * mu, sigma_sq_inv.squeeze().transpose(0,1) )
    r2 = mu * mu * sigma_sq_inv.squeeze()
    r2 = r2.sum(axis=1,keepdim=True)

    second_term = 2*torch.matmul(mu, (mu*sigma_sq_inv.squeeze()).transpose(0,1))
    second_term = r - second_term + r2
    r = torch.matmul( mu * mu, sigma_sq_inv.squeeze().transpose(0,1) )
    r2 = mu * mu * sigma_sq_inv.squeeze()
    r2 = r2.sum(axis=1,keepdim=True)

    second_term = 2*torch.matmul(mu, (mu*sigma_sq_inv.squeeze()).transpose(0,1))
    second_term = r - second_term + r2

    #this is det(Sigma), because we don't have off diag elements
    det_sigma = log_sigma_sq.sum(axis=1)
    # from each det(sigma) we need to subtract every other det(Sigma)
    third_term = torch.unsqueeze(det_sigma, dim=1) - torch.unsqueeze(det_sigma, dim=1).transpose(0,1)

    return (0.5 * ( first_term + second_term + third_term - dim_z )).mean()

#def beta_vae_loss(x,x_hat,z_mu, z_log_sigma_sq, beta=1):
#    MSE = torch.square( x - x_hat ).mean(axis=1).sum()
#    KL_div = kl_loss_gaussians(z_mu, z_log_sigma_sq)
#
#    return MSE + beta * KL_div

class BetaVAE_Loss(torch.nn.Module):

    def __init__(self, beta=1):
        super(BetaVAE_Loss, self).__init__()
        self.beta = beta
        self.MSELoss = torch.nn.MSELoss()

    def forward(self,x, x_hat, z_mu, z_log_sigma_sq):
        mse_loss = self.MSELoss(x_hat,x)
        kl_loss = kl_loss_gaussians(z_mu, z_log_sigma_sq)
        return mse_loss + self.beta * kl_loss

    def forward_components(self,x, x_hat, z_mu, z_log_sigma_sq):
        mse_loss = self.MSELoss(x_hat,x)
        kl_loss = kl_loss_gaussians(z_mu, z_log_sigma_sq)
        return mse_loss, kl_loss

class Inv_Loss_Moyer_2018(torch.nn.Module):

    def __init__(self,recon_weight=1, prior_weight=1, marginal_weight=1):
        super(Inv_Loss_Moyer_2018, self).__init__()
        self.recon_weight = recon_weight
        self.prior_weight = prior_weight
        self.marginal_weight = marginal_weight
        self.MSELoss = torch.nn.MSELoss()

    def forward(self, x, x_hat, z_mu, z_log_sigma_sq):
        mse_loss = self.MSELoss(x_hat,x)
        prior_loss = kl_loss_gaussians(z_mu, z_log_sigma_sq)
        marginal_loss = all_pairs_gaussian_kl(z_mu, z_log_sigma_sq, z_mu.size()[1])

        loss = (self.recon_weight * mse_loss * (28*28)) + \
            self.prior_weight * prior_loss + \
            self.marginal_weight * marginal_loss
        return loss

    def forward_components(self, x, x_hat, z_mu, z_log_sigma_sq):
        mse_loss = self.MSELoss(x_hat,x)
        prior_loss = kl_loss_gaussians(z_mu, z_log_sigma_sq)
        marginal_loss = all_pairs_gaussian_kl(z_mu, z_log_sigma_sq, z_mu.size()[1])

        return mse_loss, prior_loss, marginal_loss