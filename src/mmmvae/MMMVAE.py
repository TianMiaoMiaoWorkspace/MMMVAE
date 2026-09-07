# -*- coding: utf-8 -*-
"""
@author: mmTian
v.1.0.0
"""

# %%lib and tools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import scanpy as sc
import random
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os
import argparse
import time


# scFEA lib
from tqdm import tqdm
from .ClassFlux import FLUX
from .DatasetFlux import MyDataset
from .util import pearsonr, get_data_path




def myLoss(m, c, lamb1 = 0.2, lamb2= 0.2, lamb3 = 0.2, lamb4 = 0.2, geneScale = None, moduleScale = None):
    # balance constrain
    total1 = torch.pow(c, 2)
    total1 = torch.sum(total1, dim = 1)

    # non-negative constrain
    error = torch.abs(m) - m
    total2 = torch.sum(error, dim=1)

    # sample-wise variation constrain
    diff = torch.pow(torch.sum(m, dim=1) - geneScale, 2)
    if torch.sum(diff > 0).item() == m.shape[0]:
        total3 = torch.pow(diff, 0.5)
    else:
        total3 = diff

    # module-wise variation constrain
    if lamb4 > 0 :
        corr = torch.ones(m.shape[0], device=m.device, dtype=torch.float32)
        for i in range(m.shape[0]):
            corr[i] = pearsonr(m[i, :], moduleScale[i, :])
        corr = torch.abs(corr)
        penal_m_var = torch.ones(m.shape[0], device=m.device, dtype=torch.float32) - corr
        total4 = penal_m_var
    else:
        total4 = torch.zeros(m.shape[0], device=m.device, dtype=torch.float32)

    # loss
    loss1 = torch.sum(lamb1 * total1)
    loss2 = torch.sum(lamb2 * total2)
    loss3 = torch.sum(lamb3 * total3)
    loss4 = torch.sum(lamb4 * total4)
    loss = loss1 + loss2 + loss3 + loss4
    return loss, loss1, loss2, loss3, loss4

def set_all_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def scale_df(df):
    if df.empty or df.shape[0]==0 or df.shape[1]==0:
        return df.copy()
    arr = StandardScaler().fit_transform(df)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

# =============================================================================
class SCMetabolicEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A_norm=None):
        h = F.gelu(self.fc2(F.gelu(self.fc1(X))))
        h = self.norm(h)
        return self.dropout(h)

class SpatialMetabolicEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.W_gcn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A_norm):
        h = F.gelu(self.fc2(F.gelu(self.fc1(X))))
        h = self.norm(h)
        h = A_norm @ h
        h = F.gelu(self.W_gcn(h))
        return self.dropout(h)

class MultiModalAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_modalities):
        super(MultiModalAttentionFusion, self).__init__()
        self.num_modalities = num_modalities
        self.att_linear = nn.Linear(num_modalities * hidden_dim, num_modalities)
    def forward(self, hs):
        h_cat = torch.cat(hs, dim=1)
        att_scores = self.att_linear(h_cat)
        alpha = F.softmax(att_scores, dim=1)
        h_fused = 0
        for i in range(len(hs)):
            h_fused += alpha[:, i].unsqueeze(1) * hs[i]
        return h_fused, alpha

class SCVariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc_hidden = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, h):
        h = F.gelu(self.fc_hidden(h))
        h = self.norm(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class SpatialVariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SpatialVariationalEncoder, self).__init__()
        self.fc_hidden = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    def forward(self, h_fused, A_norm):
        h = F.gelu(self.fc_hidden(h_fused))
        h = self.norm(h)
        h = A_norm @ h
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class ConfigurableMMMVAE(nn.Module):
    def __init__(self, input_dims, hidden_dim, latent_dim,
                 selected_modalities, datatype="Spatial"):
        super().__init__()
        self.selected_modalities = selected_modalities
        self.datatype = datatype
        self.use_graph = (datatype == "Spatial")
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for mod in selected_modalities:
            if self.use_graph:
                self.encoders[mod] = SpatialMetabolicEncoder(input_dims[mod], hidden_dim)
            else:
                self.encoders[mod] = SCMetabolicEncoder(input_dims[mod], hidden_dim)
            self.decoders[mod] = nn.Linear(latent_dim, input_dims[mod])
        self.fusion = MultiModalAttentionFusion(
            hidden_dim,
            num_modalities=len(selected_modalities)
        )
        if self.use_graph:
            self.vgae_encoder = SpatialVariationalEncoder(
                hidden_dim,
                hidden_dim,
                latent_dim
            )
        else:
            self.vgae_encoder = SCVariationalEncoder(
                hidden_dim,
                hidden_dim,
                latent_dim
            )
    def forward(self, x_dict, A_norm=None):
        hs = []
        for mod in self.selected_modalities:
            if self.use_graph:
                h = self.encoders[mod](x_dict[mod], A_norm)
            else:
                h = self.encoders[mod](x_dict[mod])
            hs.append(h)
        h_fused, alpha = self.fusion(hs)
        if self.use_graph:
            z, mu, logvar = self.vgae_encoder(h_fused, A_norm)
        else:
            z, mu, logvar = self.vgae_encoder(h_fused)
        recons = {
            mod: self.decoders[mod](z)
            for mod in self.selected_modalities
        }
        return recons, mu, logvar, z, alpha









def main(args):

    root = args.root
    test_file = args.test_file
    moduleGene_file = args.moduleGene_file
    stoichiometry_matrix = args.stoichiometry_matrix
    cName_file = args.cName_file
    output_flux_file = args.output_flux_file
    output_balance_file = args.output_balance_file
    output_root = args.output_root
    sample = args.sample
    species = args.species
    datatype = args.datatype
    epochs = args.epochs
    custom_colors = args.custom_colors
    n_clusters_range = args.n_clusters_range
    n_components = args.n_components
    seed = args.seed


   
    hidden_dim=128
    latent_dim=16
    barcolor='skyblue'
    spatialweight=3
    LEARN_RATE = 0.008
    LAMB_BA = 1
    LAMB_NG = 1
    LAMB_CELL =  1
    LAMB_MOD = 1e-2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("scFEA start...")
    geneExpr = pd.read_csv(
        os.path.join(root, test_file),
        index_col=0
    ).T
    geneExpr = geneExpr * 1.0

    if geneExpr.max().max() > 50:
        geneExpr = np.log2(geneExpr + 1)

    geneExprSum = geneExpr.sum(axis=1)
    stand = geneExprSum.mean()
    geneExprScale = geneExprSum / stand
    geneExprScale = torch.FloatTensor(geneExprScale.values.copy()).to(device)
    BATCH_SIZE = geneExpr.shape[0]


    csv_file = get_data_path(moduleGene_file)
    moduleGene = pd.read_csv(
        csv_file,
        index_col=0
    )
    moduleLen = [moduleGene.iloc[i,:].notna().sum() for i in range(moduleGene.shape[0])]
    moduleLen = np.array(moduleLen)

    module_gene_all = []
    for i in range(moduleGene.shape[0]):
        for g in moduleGene.iloc[i,:].dropna():
            module_gene_all.append(g)
    module_gene_all = set(module_gene_all)
    data_gene_all = set(geneExpr.columns)
    gene_overlap = sorted(list(data_gene_all.intersection(module_gene_all)))

    csv_file = get_data_path(stoichiometry_matrix)
    cmMat = pd.read_csv(
        csv_file,
        header=None
    ).values
    cmMat = torch.FloatTensor(cmMat).to(device)

    csv_file = get_data_path(cName_file)
    cName = pd.read_csv(
        csv_file,
        header=0
    ).columns
    print("Load data done.")

    emptyNode = []
    geneExpr = geneExpr[gene_overlap]    
    gene_names = geneExpr.columns
    cell_names = geneExpr.index.astype(str)
    n_modules = moduleGene.shape[0]
    n_genes = len(gene_names)
    n_cells = len(cell_names)
    n_comps = cmMat.shape[0]

    geneExprDf_list = []
    for i in range(n_modules):
        genes = [g for g in moduleGene.iloc[i,:].dropna()]
        if not genes:
            emptyNode.append(i)
            continue
        temp = geneExpr.copy()
        temp.loc[:, [g for g in gene_names if g not in genes]] = 0
        temp = temp.T

        new_idx = [f'{i:02d}_{g}' for g in gene_names]
        temp.index = new_idx
        geneExprDf_list.append(temp)

    geneExprDf = pd.concat(geneExprDf_list, ignore_index=False, sort=False)

    arr = geneExprDf.values.T.astype(float)
    X = torch.FloatTensor(arr.copy()).to(device)
    df = geneExprDf.copy()
    df.index = [i.split('_')[0] for i in df.index]
    df.index = df.index.astype(int)
    module_scale = df.groupby(df.index).sum().T
    module_scale = module_scale[module_scale.sum(axis=1) > 0]

    module_scale = torch.FloatTensor(module_scale.values / moduleLen[:len(module_scale)]).to(device)

    print("Process data done.")

    torch.manual_seed(16)
    net = FLUX(X, n_modules, f_in=n_genes, f_out=1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARN_RATE)

    dataSet = MyDataset(X, geneExprScale, module_scale)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("Starting train neural network...")
    start = time.time()
    loss_v, loss_v1, loss_v2, loss_v3, loss_v4 = [], [], [], [], []

    for epoch in tqdm(range(epochs)):
        loss, loss1, loss2, loss3, loss4 = 0,0,0,0,0
        for X_batch, X_scale_batch, m_scale_batch in train_loader:
            X_batch = X_batch.float().to(device)
            X_scale_batch = X_scale_batch.float().to(device)
            m_scale_batch = m_scale_batch.float().to(device)

            out_m_batch, out_c_batch = net(X_batch, n_modules, n_genes, n_comps, cmMat)
            loss_batch, l1, l2, l3, l4 = myLoss(
                out_m_batch, out_c_batch,
                lamb1=LAMB_BA, lamb2=LAMB_NG, lamb3=LAMB_CELL, lamb4=LAMB_MOD,
                geneScale=X_scale_batch, moduleScale=m_scale_batch
            )

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            loss += loss_batch.cpu().detach().numpy()
            loss1 += l1.cpu().detach().numpy()
            loss2 += l2.cpu().detach().numpy()
            loss3 += l3.cpu().detach().numpy()
            loss4 += l4.cpu().detach().numpy()

        loss_v.append(loss)
        loss_v1.append(loss1)
        loss_v2.append(loss2)
        loss_v3.append(loss3)
        loss_v4.append(loss4)

    end = time.time()
    print("Training time:", end - start)

    fluxStatuTest = np.zeros((n_cells, n_modules), dtype='f')
    balanceStatus = np.zeros((n_cells, n_comps), dtype='f')
    net.eval()

    with torch.no_grad():
        for i, (X_batch, _, _) in enumerate(torch.utils.data.DataLoader(dataSet, batch_size=1, shuffle=False)):
            X_batch = X_batch.float().to(device)
            out_m_batch, out_c_batch = net(X_batch, n_modules, n_genes, n_comps, cmMat)
            fluxStatuTest[i] = out_m_batch.cpu().numpy()
            balanceStatus[i] = out_c_batch.cpu().numpy()

    setF = pd.DataFrame(fluxStatuTest, index=geneExpr.index, columns=moduleGene.index)
    setF.to_csv(os.path.join(output_root, output_flux_file))
    setB = pd.DataFrame(balanceStatus, index=geneExpr.index, columns=cName)
    setB.to_csv(os.path.join(output_root, output_balance_file))


    # ==================== MMMVAE 下游分析 ====================

    print("Starting load data...")
 
    score = pd.read_csv(os.path.join(root, f"{sample}_score.csv"), header=0, index_col=0)
    counts = pd.read_csv(os.path.join(root, f"{sample}_counts.csv"), header=0, index_col=0)
    balance = pd.read_csv(os.path.join(output_root, output_balance_file), header=0, index_col=0)
    flux = pd.read_csv(os.path.join(output_root, output_flux_file),header=0, index_col=0)




    score = scale_df(score)
    counts = scale_df(counts)
    balance = scale_df(balance)
    flux = scale_df(flux)
    x_dict = {
        'score': torch.tensor(score.values, dtype=torch.float32),
        'counts': torch.tensor(counts.values, dtype=torch.float32),
        'balance': torch.tensor(balance.values, dtype=torch.float32),
        'flux': torch.tensor(flux.values, dtype=torch.float32),
    }
    if datatype == "Spatial":
        A_norm = pd.read_csv(os.path.join(root, f"{sample}_adj_matrix.csv"),  header=0, index_col=0)
        neighbor = pd.read_csv(os.path.join(root, f"{sample}_neighbor_means_counts.csv"), header=0, index_col=0)
        neighbor = scale_df(neighbor)
        x_dict['neighbor'] = torch.tensor(neighbor.values, dtype=torch.float32)
        A_norm = torch.tensor(A_norm.values, dtype=torch.float32)
    else:
        A_norm = None

    for k in x_dict:
        x_dict[k] = x_dict[k].to(device)
    if A_norm is not None:
        A_norm = A_norm.to(device)
        
    print("Starting process data...")
    set_all_seeds(seed)
    if datatype == "Spatial":
        active_modalities = ['score','balance','flux','counts','neighbor']
    else:
        active_modalities = ['score','balance','flux','counts']
    print(f"Active modalities: {active_modalities}")
    input_dims = {m: x_dict[m].shape[1] for m in active_modalities}
    model = ConfigurableMMMVAE(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        selected_modalities=active_modalities,
        datatype=datatype
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        if datatype == "Spatial":
            recons, mu, logvar, z, alpha = model(x_dict, A_norm)
        else:
            recons, mu, logvar, z, alpha = model(x_dict, None)
        recon_loss = sum(
            F.mse_loss(recons[m], x_dict[m])
            for m in active_modalities
        )
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))

        loss = (
            recon_loss+ (min(1.0, epoch / 100) * KL)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | Loss={loss.item():.4f} | "
                f"Recon={recon_loss.item():.4f} | "
                f"KL={KL.item():.4f} | "
            )
    model.eval()
    with torch.no_grad():
        if datatype == "Spatial":
            recons, mu, logvar, z, alpha = model(x_dict, A_norm)
        else:
            recons, mu, logvar, z, alpha = model(x_dict, None)
        Z = mu.cpu().numpy()
        alpha_np = alpha.cpu().numpy()
    importance_scores = alpha_np.mean(axis=0)
    plt.figure(figsize=(6, 4))
    plt.bar(active_modalities, importance_scores, color=barcolor)
    plt.title("Modality Importance (Attention Weights)")
    plt.savefig(os.path.join(output_root, f"{sample}_Importance.pdf"), format='pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()



    print(" BIC and AIC Plotting...")
    if datatype == "Spatial":
        spatial_weight =  spatialweight
    else:
        spatial_weight = 0 
    coords = pd.read_csv(os.path.join(root, f"{sample}_coords.csv"), header=0, index_col=0)
    adata_z = sc.AnnData(Z)
    adata_z.obs_names = counts.index
    adata_z.obsm['spatial'] = coords[['x', 'y']].values
    scaler_z = StandardScaler()
    z_scaled = scaler_z.fit_transform(adata_z.X) 
    scaler_spatial = StandardScaler()
    spatial_scaled = scaler_spatial.fit_transform(adata_z.obsm['spatial'])
    features = np.hstack([z_scaled, spatial_weight * spatial_scaled])
    
    bic_scores = []
    aic_scores = []
    if n_clusters_range <= 2:
        raise ValueError("--n_clusters_range must be greater than 2")
    for n in range(2, n_clusters_range+1):
        gmm_tmp = GaussianMixture(n_components=n, covariance_type='full', random_state=seed)
        gmm_tmp.fit(features)
        bic_scores.append(gmm_tmp.bic(features))
        aic_scores.append(gmm_tmp.aic(features))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(range(2, n_clusters_range+1), bic_scores, 'o-', label='BIC', color='#e43636', linewidth=2)
    ax.plot(range(2, n_clusters_range+1), aic_scores, 's-', label='AIC', color='#74cad9', linewidth=2)
    ax.set_xlabel('Number of Clusters (n_components)')
    ax.set_ylabel('Score (Lower is Better)')
    ax.set_title('BIC/AIC vs Number of Clusters')
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.savefig(os.path.join(output_root,f"{sample}_GMM_BICAIC.pdf"), format='pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    best_n = range(2, n_clusters_range+1)[np.argmin(bic_scores)]
    print(f"Auto-selected best n_components = {best_n} (min BIC)")

    
    print("Process data done:")



    bicaic_path = os.path.join(output_root,f"{sample}_GMM_BICAIC.pdf")
    print("\n================ GMM Model Selection =================")
    print(f"[INFO] BIC/AIC curve has been saved to:\n{bicaic_path}")
    print("[INFO] Please check this file to determine the optimal number of clusters.")
    print("[INFO] If you do not specify --n_components, the model will use the BIC-optimal value.")
    print("=====================================================\n")
# =============================================================================
   # 确定聚类数
    if n_components is None:
        print(f"[AUTO] n_components is not provided. Using BIC-optimal value: {best_n}")
        n_components = best_n
    else:
        print(f"[MANUAL] Using user-specified n_components = {n_components}")
        print("[SUGGESTION] Make sure this choice is consistent with the BIC/AIC curve.")


    # 训练 GMM
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=seed)
    labels = gmm.fit_predict(features)
    proba = gmm.predict_proba(features)


    # 保存 CSV 结果
    result_df = pd.DataFrame({
        'cluster': labels,
        'x_coord': coords['x'].values if coords is not None else np.full(len(labels), np.nan),
        'y_coord': coords['y'].values if coords is not None else np.full(len(labels), np.nan)
    })
    
    result_df.to_csv(os.path.join(output_root,f"{sample}_python_clusters.csv"), index=True, index_label='cell')



    proba_df = pd.DataFrame(proba, columns=[f'prob_cluster_{i}' for i in range(proba.shape[1])])
    proba_df.to_csv(os.path.join(output_root,f"{sample}_lesion_prob.csv"), index=True, index_label='cell')





    # 空间聚类可视化
    if coords is not None and ('x' in coords.columns and 'y' in coords.columns):
        adata_z.obs['metabolic_cluster'] = labels.astype(str)
        if custom_colors is not None:
            custom_colors = [c if c.startswith('#') else f'#{c}' for c in custom_colors.split(',')]
        else:
            custom_colors = [
                "#83fc8d","#7c7afa","#fffd76","#f5ab7a","#71e6dc","#edb0f3","#fd3142",
                "#ff0000","#4092af","#bbbb0c","#72fdb7","#dfd2c9","#ce1d75","#008080","#ff6347"
            ]
        fig = sc.pl.embedding(
            adata_z, basis='spatial', color='metabolic_cluster',
            title=f'{sample} Metabolic Cluster', frameon=False,
            legend_loc='right margin', s=50, palette=custom_colors,
            show=False, return_fig=True
        )
        
        fig.savefig(os.path.join(output_root,f"{sample}_spatial_cluster.pdf"), format='pdf', dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig)
        print(os.path.join(output_root,f"{sample}_spatial_cluster.pdf"))


    return

def parse_arguments():
    #scFEA
    parser = argparse.ArgumentParser(description='MMMVAE: Multi-Modal Metabolic Variational Autoencoder')
    parser.add_argument('--sample', type=str, required=True,
                        help="Sample name of input files, which is also used to name the output files")
    parser.add_argument('--species', type=str, default="Homo_sapiens",
                        choices=["Homo_sapiens", "Mus_musculus"],
                        help="Species: 'Homo_sapiens'(human) or 'Mus_musculus'(mouse). Default: %(default)s")
    parser.add_argument('--test_file', type=str, required=False,
                        help="Tab‑separated gene profile matrix input file. Row is gene symbol, column is single‑cell or spatial spot. Default: {sample}_exp.csv under --root.")
    parser.add_argument('--moduleGene_file', type=str, default=None,
                        help="The table contains genes for each module. We provide human and mouse two models. For human model, please use module_gene_m168.csv which is default.  All candidate moduleGene files are provided in /data/ folder.")
    parser.add_argument('--stoichiometry_matrix', type=str, default=None,
                        help="The table describes relationship between compounds and modules. Each row is an intermediate metabolite and each column is metabolic module. For human model, please use cmMat_c70_m168.csv which is default. All candidate stoichiometry matrices are provided in /data/ folder. ")
    parser.add_argument('--cName_file', type=str, default="cName_c70_m168.csv",
                        help="Built‑in compound name table csv.This table contains the names of the compounds and their corresponding identifiers. Specifically, the first row represents the names of the compounds, and the second row shows the corresponding identifiers. Default: %(default)s ")
    parser.add_argument('--output_flux_file', type=str, required=False,
                        help="Filename for predicted flux output. Default: {sample}_flux.csv")
    parser.add_argument('--output_balance_file', type=str, required=False,
                        help="Filename for predicted balance output. Default: {sample}_balance.csv")
    # MMMVAE
    parser.add_argument('--root', type=str, required=True,
                        help="""The data directory for input data. The root includes Spatial folder, filtered_count_matrix folder or filtered_feature_bc_matrix.h5. 
                              The Spatial folder includes tissue_positions_list.csv, tissue_hires_image.png, tissue_lowres_image.png and scalefactors_json.json.
                              The filtered_count_matrix folder includes barcodes.tsv.gz, features.tsv.gz, and  matrix.mtx.gz""")
    parser.add_argument('--output_root', type=str, required=False,
                        help="Output directory for MMMVAE results storing output matrices. Default: {root}/output")
    parser.add_argument('--datatype', type=str, default='Spatial', choices=['Spatial','single_cell'],
                        help="The data type input by the user, either 'single_cell' or 'Spatial'. Default: %(default)s")
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs. Default: %(default)s')
    parser.add_argument('--n_clusters_range', type=int, default=15,
                        help="Upper bound for niche‑cluster search. Algorithm searches optimal k from 2 to this value. Default: %(default)s")
    parser.add_argument('--seed', type=int, default=2026,
                        help="Random seed for reproducibility. Default: %(default)s")
    parser.add_argument('--n_components', type=int, default=None,
                        help="Manually set fixed number of metabolic niche clusters, skip automatic BIC‑based optimal‑k selection.")
    parser.add_argument('--custom_colors', type=str, default=None,
                        help="Comma‑separated hex color strings for plotting metabolic ecotypes, e.g.: ['#83fc8d', '#7c7afa'].")

    args = parser.parse_args()
    return args




def cli_main():
    args = parse_arguments()
    
    # ========== species ==========
    if args.species == "Homo_sapiens":
        args.moduleGene_file = "module_gene_m168.csv"
        args.stoichiometry_matrix = "cmMat_c70_m168.csv"
    elif args.species == "Mus_musculus":
        args.moduleGene_file = "module_gene_complete_mouse_m168.csv"
        args.stoichiometry_matrix = "cmMat_complete_mouse_c70_m168.csv"
    else:
        raise ValueError(f"Unsupported species {args.species}, choose Homo_sapiens / Mus_musculus")
    
    
    if args.output_flux_file is None:
        args.output_flux_file = f"{args.sample}_flux.csv"
    if args.output_balance_file is None:
        args.output_balance_file = f"{args.sample}_balance.csv"
    if args.test_file is None:
        args.test_file = f"{args.sample}_exp.csv"
    if args.output_root is None:
        args.output_root = os.path.join(args.root, "output")
    os.makedirs(args.output_root, exist_ok=True)
    main(args)
if __name__ == "__main__":
    cli_main()
# %%
