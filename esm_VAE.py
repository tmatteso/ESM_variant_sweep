import umap.umap_ as umap
import torch.utils.data as data_utils
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PARAMS = {
    'input_dim': 1280,
    'intermediate_dim': 128,
    'latent_dim': 15,
    'kl_weight': 0.1,
    'pred_weight': 50,
    'batch_size': 512,
    'train_size': 0.8,
    'n_epochs': 400,
    'lr': 1e-03,
}

class VAE(nn.Module):
    
    def __init__(self, input_dim, intermediate_dim, latent_dim):
        
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim

        # Encoder
        self.input_enc = nn.Linear(in_features = self.input_dim, out_features = self.intermediate_dim)
        self.batch_norm_enc = nn.BatchNorm1d(self.intermediate_dim)
        self.drop_enc = nn.Dropout(p = 0.05)
        self.mu_enc = nn.Linear(in_features = self.intermediate_dim, out_features = self.latent_dim)
        self.logvar_enc = nn.Linear(in_features = self.intermediate_dim, out_features = self.latent_dim)
 
        # Decoder 
        self.latent_dec = nn.Linear(in_features = self.latent_dim, out_features = self.intermediate_dim)
        self.batch_norm_dec = nn.BatchNorm1d(self.intermediate_dim)
        self.drop_dec = nn.Dropout(p = 0.05)
        self.out_dec = nn.Linear(in_features = self.intermediate_dim, out_features = self.input_dim)
        self.y_dec = nn.Linear(in_features = self.latent_dim, out_features = 1, bias = False)
                
    def encode(self, x):
        x = self.drop_enc(F.relu(self.batch_norm_enc(self.input_enc(x))))
        x_mu = self.mu_enc(x)
        x_logvar = self.logvar_enc(x)
        return x_mu, x_logvar
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        r = torch.randn_like(std)
        z = mu + r * std
        return z
    
    def decode(self, z):
        return self.out_dec(self.drop_dec(F.relu(self.batch_norm_dec(self.latent_dec(z)))))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var
    
    def predict(self, x_mu):
        return self.y_dec(x_mu)
    
    def loss_mse(self, x, x_hat):
        return F.mse_loss(x_hat, x, reduction = 'sum') / x.shape[0]
    
    def loss_kld(self, mu, log_var):
        # KL divergence between the distribution determined by the encoder (by mu and log_var) and the standard normal
        # distribution N(0, 1):
        # KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]
    
    def loss_pred(self, mu, y):
        return F.mse_loss(self.predict(mu), y, reduction = 'sum') / mu.shape[0]
    
class DummyWith:
    
    def __enter__(self, *args):
        pass
    
    def __exit__(self, *args):
        pass

def run_model(model, data_loader, fit = True, optimizer = None, labeled = False, kl_weight = 1, pred_weight = 1, \
        device = 'cpu'):
    
    if fit:
        model.train()
    else:
        model.eval()
    
    n_batches = 0
    total_loss = 0.0
    total_loss_mse = 0.0
    total_loss_kld = 0.0
    total_loss_pred = 0.0
    
    with (DummyWith() if fit else torch.no_grad()):
        for i, batch_data in enumerate(data_loader):

            n_batches += 1

            if labeled:
                x, y = batch_data
                x = x.to(device)
                y = y.to(device)
            else:
                x = batch_data
                x = x.to(device)
                y = None
            
            if fit:
                optimizer.zero_grad()
            
            x_hat, mu, log_var = model(x)

            batch_loss_mse = model.loss_mse(x, x_hat)
            total_loss_mse += batch_loss_mse
            batch_loss_kld = model.loss_kld(mu, log_var)
            total_loss_kld += batch_loss_kld
            batch_loss = batch_loss_mse + kl_weight * batch_loss_kld

            if labeled:
                batch_loss_pred = model.loss_pred(mu, y)
                total_loss_pred += batch_loss_pred
                batch_loss += pred_weight * batch_loss_pred

            total_loss += batch_loss

            if fit:
                batch_loss.backward()
                optimizer.step()
        
    total_loss /= n_batches
    total_loss_mse /= n_batches
    total_loss_kld /= n_batches
    total_loss_pred /= n_batches
    
    return total_loss, total_loss_mse, total_loss_kld, total_loss_pred

#X = StandardScaler().fit_transform(np.stack(rectified_df["repr"]))# actual tensors
# make np array of zeros
#y = np.zeros((X.shape[0], 1))
def run_VAE(device, X, y):
    dataset = data_utils.TensorDataset(torch.tensor(X, dtype = torch.float32), 
                                       torch.tensor(y, dtype = torch.float32))
    train_set, valid_set = train_test_split(dataset, train_size = PARAMS['train_size'])

    train_loader = data_utils.DataLoader(train_set, batch_size = PARAMS['batch_size'], shuffle = True)
    valid_loader = data_utils.DataLoader(valid_set, batch_size = PARAMS['batch_size'], shuffle = False)

    model = VAE(input_dim = PARAMS['input_dim'], 
                intermediate_dim = PARAMS['intermediate_dim'], 
                latent_dim = PARAMS['latent_dim']).to(device)
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('# trainable parameters: %.1g' % n_total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr = PARAMS['lr'])

    for epoch in range(PARAMS['n_epochs']):
        train_loss, train_loss_mse, train_loss_kld, train_loss_pred = run_model(model, train_loader, fit = True,
                                                                                optimizer = optimizer, labeled = True, 
                                                                                kl_weight = PARAMS['kl_weight'], 
                                                                                pred_weight = PARAMS['pred_weight'])
        valid_loss, valid_loss_mse, valid_loss_kld, valid_loss_pred = run_model(model, train_loader, fit = False, 
                                                                                labeled = True, kl_weight = PARAMS['kl_weight'],
                                                                                pred_weight = PARAMS['pred_weight'])

        if epoch % 10 == 0: 
            print(('Epoch %d/%d: training loss = %.2g (MSE = %.2g, KLD = %.2g, pred = %.2g), ' + \
                    'validation loss = %.2g (MSE = %.2g, KLD = %.2g, pred = %.2g)') % (epoch, PARAMS['n_epochs'], train_loss, \
                    train_loss_mse, train_loss_kld, train_loss_pred, valid_loss, valid_loss_mse, valid_loss_kld, \
                    valid_loss_pred), end = '\n' if epoch % 100 == 0 else '\r')
    return model
    print('Done.')
    
def get_low_D(model, rectified_df, X):
    # do the calc for metric nadav wants from VAE embeds
    high_D_dataset_loader = data_utils.DataLoader(torch.tensor(X, dtype = torch.float32), 
                                                  batch_size = PARAMS['batch_size'], \
                                                  shuffle = False)
    low_D_embeddings = []
    # batch_size is 512 rn
    with torch.no_grad():
        for batch in high_D_dataset_loader:
            x_mu, _ = model.encode(batch.to("cpu"))
            low_D_embeddings.append(x_mu.cpu().numpy())


    rectified_df['VAE_embed'] = [row for row in np.concatenate( low_D_embeddings, axis=0 )]
    low_D_embeddings = np.vstack(low_D_embeddings)
    #print(low_D_embeddings.shape)
    return low_D_embeddings

def get_UMAP(low_D_embeddings, rectified_df):
    reducer = umap.UMAP(spread = 2, min_dist = 0.1, n_components = 2, random_state = 0)
    umap_embeddings = reducer.fit_transform(low_D_embeddings)

    rectified_df['umap1'] = umap_embeddings[:, 0] # added the -1
    rectified_df['umap2'] = umap_embeddings[:, 1] # so all this stems from the repr being too long
    rectified_df['umap_embed'] = [row for row in umap_embeddings]
    return rectified_df
    