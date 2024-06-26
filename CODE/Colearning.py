import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchsummary
from torch.utils.data import Dataset, DataLoader
from Dataloader_train import TrainingDataset
from Dataloader_test import TestDataset


from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import KFold
from Architecture_multimeasure import AE, Conditional_VAE
import losses
from sklearn.feature_selection import mutual_info_regression
from torch.utils.tensorboard import SummaryWriter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import resample

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Project running on device: ", DEVICE)

CONNECTOME_SIZE = 84*84 
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Set up Tensorboard log
tensorboard_dir = 'Tensorboard_colearn'
writer = SummaryWriter(tensorboard_dir)


# Hyperparameters
beta =      1
delta =     100 #cnm
alpha =     100 # recon
kappa =     10000  #sex loss
gamma =     1000  # klloss
batch_size = 100       # Change when add in more data
epochs =    2#501
learning_rate = 1e-4



normalize_cnms          =np.array([1,1,1/2,1,1/np.sqrt(10000000),1/1000000,1/200, 1/200,1, 1, 1/250, 1/np.sqrt(CONNECTOME_SIZE)])
normalize_cnms          =torch.from_numpy(normalize_cnms).to(DEVICE)
unnormlizefactor        = np.array([1,1,2,1,np.sqrt(10000000),1000000,200, 200,1, 1, 250, np.sqrt(CONNECTOME_SIZE)])
makepositive            =np.array([0,0,1,0,0,0,0,0,0,0,0,0])
makepositive            = torch.from_numpy(makepositive).to(DEVICE)


# Set up dataset (to be subsampled into training and validation) and folds
transform            = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset        = TrainingDataset()
test_dataset         = TestDataset()
validation_loader    = DataLoader(test_dataset, batch_size=len(test_dataset))

def cohen_d(group1, group2):
    """
    Compute Cohen's d.

    Parameters:
    - group1: array-like, first group of data
    - group2: array-like, second group of data

    Returns:
    - d: Cohen's d (effect size)
    """
    # Calculate the means of the two groups
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    # Calculate the variance of the two groups
    variance1, variance2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Calculate the sample sizes of the two groups
    n1, n2 = len(group1), len(group2)
    
    # Calculate the pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    return d
    
    
measure_names = ["betweenness_centrality","modularity","assortativity","participation","clustering","nodal_strength","local_efficiency","global_efficiency", "density","rich_club","path_length","edge_count"]

measures_on = np.ones(12)

experiment = ''.join(str(x) for x in measures_on)

# Bootstrap iterations
for iteration in np.arange(1,101):

    print('Bootstrap iter: {}'.format(iteration))
    train_dataset_subsample = resample(train_dataset, replace=False, n_samples=round(len(train_dataset)*0.8))
    train_loader            = DataLoader(train_dataset_subsample, batch_size=batch_size)

    # create a model from Arch (conditional variational autoencoder)
    # load it to the specified device, either gpu or cpu
    model = Conditional_VAE(in_dim=CONNECTOME_SIZE,c_dim=1, z_dim=100, num_measures=np.sum(measures_on==1)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define losses
    loss_fn         = losses.BetaVAE_Loss(beta)
    criterion_CE    = nn.BCEWithLogitsLoss() 

    for epoch in range(epochs):
        print("EPOCH: ",epoch)
        # Set all recording losses to zero
        loss = 0
        training_loss_total = 0
        training_loss_reconstruction = 0
        training_loss_kldivergence = 0
        training_loss_prediction_site1 = 0
        training_loss_prediction_site2 = 0
        training_loss_age = 0
        training_loss_sex = 0
        countofsite1 = 0
        countofsite2 = 0

        for batch_features, batch_predictionlabels, batch_sitelabels, batch_agelabels, batch_sexlabels, _ in train_loader:

            # Send features, prediction labels, and site labels to device and set dimensions
            batch_features          = batch_features.view(-1, CONNECTOME_SIZE).to(DEVICE) # flatten 84x84 connectome
            batch_predictionlabels  = batch_predictionlabels.view(-1,12).to(DEVICE) # 12 graph measures
            batch_sitelabels        = batch_sitelabels.view(-1,1).to(DEVICE) # 1 site label per scan
            makepositive=torch.zeros_like(batch_predictionlabels)
            makepositive[:,2]=1
            batch_predictionlabels  = normalize_cnms.view(-1,12) * (batch_predictionlabels + makepositive)


            batch_predictionlabels  = batch_predictionlabels[:,measures_on==1]
            batch_agelabels         = batch_agelabels.view(-1,1).to(DEVICE)
            batch_sexlabels         = batch_sexlabels.view(-1,2).to(DEVICE) 
            
            # Standard optimizer command
            optimizer.zero_grad()

            # Put data through forward pass
            x_hat, z, z_mu, z_log_sigma_sq, model_predictions_site1, model_predictions_site2, model_predictions_age, model_predictions_sex = model.forward_train(batch_features,batch_sitelabels)

            # Reconstruction and KL divergence loss terms
            recon_loss, kl_loss = loss_fn.forward_components(x_hat,batch_features,z_mu,z_log_sigma_sq)
            age_error = torch.mean(torch.square((batch_agelabels-model_predictions_age)/batch_agelabels))
            sex_error = criterion_CE(model_predictions_sex, batch_sexlabels)
            
            #print("Sex error",sex_error.item())
            sex_predictions = torch.nn.functional.sigmoid(model_predictions_sex)
            _,sex_predictions = torch.max(sex_predictions, 1)
            _,sex_labels = torch.max(batch_sexlabels, 1)
            sex_accuracy = (sex_predictions == sex_labels).float().mean()
            training_loss_sexacc = sex_accuracy
            
            train_sex = torch.sum(sex_labels) / len(sex_labels)

            # Weight KL loss with beta, set parameter at top
            kl_loss = beta * kl_loss

            # Prediction Loss term
            # We don't want to model to learn from predictions from another site. Thus, we will make sure the loss is zero for those elements of the batch
            site1_indicator = (batch_sitelabels == 1).to(torch.float32)
            site2_indicator = (batch_sitelabels == 2).to(torch.float32)

            # Errors by site
            model_prediction_error_site1 = site1_indicator * (model_predictions_site1 - batch_predictionlabels) / batch_predictionlabels
            model_prediction_error_site2 = site2_indicator * (model_predictions_site2 - batch_predictionlabels) / batch_predictionlabels
        
            # Sum across sites
            cnm_prediction_error = model_prediction_error_site1 + model_prediction_error_site2

            # MSE prediction loss
            cnm_prediction_error = torch.sum(torch.mean(torch.square(cnm_prediction_error),0))

            # Total Loss
            loss = gamma * kl_loss + alpha * recon_loss + delta * cnm_prediction_error + age_error + kappa * sex_error
            
            # Compute loss gradients and take a step with the Adam optimizer
            loss.backward()
            optimizer.step()

            # Add the mini-batch training loss to epoch loss
            training_loss_total             += loss.item()
            training_loss_reconstruction    += recon_loss.item()
            training_loss_kldivergence      += kl_loss.item()
            countofsite1                    += torch.sum(batch_sitelabels==1)
            countofsite2                    += torch.sum(batch_sitelabels==2)
            training_loss_prediction_site1  += model_prediction_error_site1.square().mean().sum().item()
            training_loss_prediction_site2  += model_prediction_error_site2.square().mean().sum().item()
            training_loss_age               += age_error.item()
            training_loss_sex               += sex_error.item()

        
        #measure_name=measure_names[measure_id]           
        #writer.add_scalars('Cohen\'s D Measure {} Experiment {}'.format(measure_name, experiment),   {'Site 1 {}'.format(iteration): site1predictions_cohensd, 'Site 2 {}'.format(iteration): site2predictions_cohensd, 'Baseline {}'.format(iteration): baseline_cohensd, 'Predicting to Native {}'.format(iteration): predictingnative_cohensd}, epoch)
        # compute the epoch training loss
        training_loss_total     = training_loss_total / len(train_loader)
        training_loss_age       = training_loss_age / len(train_loader)
        training_loss_sex       = training_loss_sex / len(train_loader)
        training_loss_reconstruction    = training_loss_reconstruction / len(train_loader)
        training_loss_kldivergence      = training_loss_kldivergence /  len(train_loader)

        training_loss_prediction_site1 = training_loss_prediction_site1 / countofsite1
        training_loss_prediction_site2 = training_loss_prediction_site2 / countofsite2
        training_loss_total = training_loss_reconstruction + training_loss_kldivergence + training_loss_prediction_site1 + training_loss_prediction_site2 + training_loss_age + training_loss_sex
        
        countofsite1=torch.sum(batch_sitelabels==1)
        countofsite2=torch.sum(batch_sitelabels==2)

        model_prediction_error_site1_measure = site1_indicator * (model_predictions_site1 - batch_predictionlabels) / batch_predictionlabels
        model_prediction_error_site2_measure = site2_indicator * (model_predictions_site2 - batch_predictionlabels) / batch_predictionlabels
        train_loss_prediction_site1_measure = model_prediction_error_site1_measure.square().mean().sum().item() / countofsite1
        train_loss_prediction_site2_measure = model_prediction_error_site2_measure.square().mean().sum().item() / countofsite2

        writer.add_scalars('Train Prediction Loss Experiment {} Site1'.format(experiment),   {'{}'.format(iteration): train_loss_prediction_site1_measure}, epoch)
        writer.add_scalars('Train Prediction Loss Experiment {} Site2'.format(experiment),   {'{}'.format(iteration): train_loss_prediction_site2_measure}, epoch)

        
        
        sex_predictions = torch.nn.functional.sigmoid(model_predictions_sex)
        _,sex_predictions = torch.max(sex_predictions, 1)
        _,sex_labels = torch.max(batch_sexlabels, 1)

        sex_accuracy = (sex_predictions == sex_labels).float().mean()
        validation_loss_sexacc = sex_accuracy
        
        
        measure_names = ["betweenness_centrality","modularity","assortativity","participation","clustering","nodal_strength","local_efficiency","global_efficiency", "density","rich_club","path_length","edge_count"]
        
        
       
        
        writer.add_scalars('Train Total Loss Colearning Experiment {}'.format(experiment),   {'{}'.format(iteration): training_loss_total}, epoch)
        writer.add_scalars('Train Reconstruction Loss Colearning Experiment {}'.format(experiment),   {'{}'.format(iteration): training_loss_reconstruction}, epoch)
        writer.add_scalars('Train KL Loss Colearning Experiment {}'.format(experiment),   {'{}'.format(iteration): training_loss_kldivergence}, epoch)
        writer.add_scalars('Train Age Loss Colearning Experiment {}'.format(experiment),   {'{}'.format(iteration): training_loss_age}, epoch)               
        writer.add_scalars('Train Sex Loss Colearning Experiment {}'.format(experiment),   {'{}'.format(iteration): training_loss_sex}, epoch)               
        writer.add_scalars('Train Sex Acc Colearning Experiment {}'.format(experiment),   {'{}'.format(iteration): training_loss_sexacc}, epoch)        
            
        
        validation_loss_prediction_site1 = 0
        validation_loss_prediction_site2 = 0
        validation_loss_kl =0
        validation_loss_age =0
        validation_loss_sex =0
        validation_loss_reconstruction =0
        countofsite1 = 0
        countofsite2 = 0
        
        for batch_features, batch_predictionlabels, batch_sitelabels, batch_agelabels, batch_sexlabels, batch_filepaths in validation_loader:
            batch_features          = batch_features.view(-1, CONNECTOME_SIZE).to(DEVICE)
            batch_predictionlabels  = batch_predictionlabels.view(-1,12).to(DEVICE)
            makepositive=torch.zeros_like(batch_predictionlabels)
            makepositive[:,2]=1
            batch_predictionlabels  = normalize_cnms.view(-1,12) * (batch_predictionlabels + makepositive)
            batch_predictionlabels  = batch_predictionlabels[:,measures_on==1]
            batch_sitelabels        = batch_sitelabels.view(-1,1).to(DEVICE)
            batch_agelabels         = batch_agelabels.view(-1,1).to(DEVICE)

            batch_sexlabels         = batch_sexlabels.view(-1,2).to(DEVICE)

            x_hat, z, z_mu, z_log_sigma_sq, model_predictions_site1, model_predictions_site2, model_predictions_age, model_predictions_sex = model.forward_train(batch_features,batch_sitelabels)
            recon_loss, kl_loss = loss_fn.forward_components(x_hat,batch_features,z_mu,z_log_sigma_sq)
            sex_error = criterion_CE(model_predictions_sex, batch_sexlabels)
            
            
            kl_loss = kl_loss * beta
            age_error = torch.mean(torch.square((batch_agelabels-model_predictions_age)/batch_agelabels))
            sex_predictions = torch.nn.functional.sigmoid(model_predictions_sex)
            _,sex_predictions = torch.max(sex_predictions, 1)
            _,sex_labels = torch.max(batch_sexlabels, 1)
            sex_accuracy = (sex_predictions == sex_labels).float().mean()
            validation_loss_sexacc = sex_accuracy
        
            sex_accuracy_val = validation_loss_sexacc
            site1_indicator = (batch_sitelabels == 1).to(torch.float32)
            site2_indicator = (batch_sitelabels == 2).to(torch.float32)

            measures_on_index = [index for (index, item) in enumerate(measures_on) if item == 1]
            measure_names = ["betweenness_centrality","modularity","assortativity","participation","clustering","nodal_strength","local_efficiency","global_efficiency", "density","rich_club","path_length","edge_count"]
            withinexperimentindex=0
            
            for measure_id in measures_on_index: 
                countofsite1=torch.sum(batch_sitelabels==1)
                countofsite2=torch.sum(batch_sitelabels==2)

                site1predictions_measure = model_predictions_site1.detach().cpu()
                site2predictions_measure = model_predictions_site2.detach().cpu()

                site1predictions_measure = site1predictions_measure[:,withinexperimentindex]
                site2predictions_measure = site2predictions_measure[:,withinexperimentindex]
                
                
                # Multiply measures back to their original ranges (was normalized to -1,1 for learning
                site1predictions_measure = (site1predictions_measure * unnormlizefactor[measure_id]) - makepositive[:,measure_id].detach().cpu()
                site2predictions_measure = (site2predictions_measure * unnormlizefactor[measure_id]) - makepositive[:,measure_id].detach().cpu()
       
                batch_predictionlabels_measure = batch_predictionlabels.detach().cpu().numpy()
                batch_predictionlabels_measure = batch_predictionlabels_measure[:,withinexperimentindex].squeeze()
                
               
                
                sample1=batch_predictionlabels_measure[(batch_sitelabels.detach().cpu().numpy()==1).squeeze()]
              
                sample2=batch_predictionlabels_measure[(batch_sitelabels.detach().cpu().numpy()==2).squeeze()]
                
                baseline_cohensd = cohen_d(sample1, sample2)
                U1, p_baseline = mannwhitneyu(sample1, sample2)

                sample1=site1predictions_measure[(batch_sitelabels.detach().cpu().numpy()==1).squeeze()]
                sample2=site1predictions_measure[(batch_sitelabels.detach().cpu().numpy()==2).squeeze()]
                site1predictions_cohensd = cohen_d(sample1.numpy(), sample2.numpy())
                
                U1, p_site1domain = mannwhitneyu(sample1.numpy(), sample2.numpy())
                
                sample1=site1predictions_measure[(batch_sitelabels.detach().cpu().numpy()==1).squeeze()]     
                sample2=site2predictions_measure[(batch_sitelabels.detach().cpu().numpy()==2).squeeze()]
                
                predictingnative_cohensd = cohen_d(sample1.numpy(), sample2.numpy())
                U1, p_nativesite = mannwhitneyu(sample1.numpy(), sample2.numpy())


                # site 2 domain
                
                sample1=site2predictions_measure[(batch_sitelabels.detach().cpu().numpy()==1).squeeze()]
                sample2=site2predictions_measure[(batch_sitelabels.detach().cpu().numpy()==2).squeeze()]
                site2predictions_cohensd = cohen_d(sample1.numpy(), sample2.numpy())
                U1, p_site2domain = mannwhitneyu(sample1, sample2)

                
                measure_name=measure_names[measure_id]
                experiment = measures_on #experiment = ','.join(str(x) for x in measures_on)
                
                
                model_prediction_error_site1_measure = site1_indicator * (model_predictions_site1[:,withinexperimentindex] - batch_predictionlabels[:,withinexperimentindex]) / batch_predictionlabels[:,withinexperimentindex]
                model_prediction_error_site2_measure = site2_indicator * (model_predictions_site2[:,withinexperimentindex] - batch_predictionlabels[:,withinexperimentindex]) / batch_predictionlabels[:,withinexperimentindex]
                validation_loss_prediction_site1_measure = model_prediction_error_site1_measure.square().mean().sum().item() / countofsite1
                validation_loss_prediction_site2_measure = model_prediction_error_site2_measure.square().mean().sum().item() / countofsite2
                
                report_df_data = {
                    'Experiment': [experiment],
                    'Iteration':[iteration],
                    'Measure_name': [measure_name],
                    'CD_site1':[site1predictions_cohensd],
                    'CD_site2':[site2predictions_cohensd],
                    'CD_baseline':[baseline_cohensd],
                    'Sex Acc':[sex_accuracy_val.detach().cpu().numpy()],
                    'Age Error':[age_error.detach().cpu().numpy()],
                    'Prediction Error Site 1':[validation_loss_prediction_site1_measure.detach().cpu().numpy()],
                    'Prediction Error Site 2':[validation_loss_prediction_site2_measure.detach().cpu().numpy()]
                
                }
                print(report_df_data)
                report_df=pd.DataFrame(report_df_data)
                report_df.to_csv('Validation_Colearning_Report_Current{}_Iteration{}_Experiment{}.csv'.format(measure_name, iteration, str(experiment)))
                withinexperimentindex = withinexperimentindex+1
                np.savetxt('AgeLabels_Colearning_On{}.txt'.format(measure_names[measure_id]), batch_agelabels.detach().cpu().numpy())
                np.savetxt('SexLabels_Colearning_On{}.txt'.format(measure_names[measure_id]), batch_sexlabels.detach().cpu().numpy())
                np.savetxt('MeasureLabels_Colearning_On{}.txt'.format(measure_names[measure_id]), batch_predictionlabels.detach().cpu().numpy())
                np.savetxt('MeasurePredictions_Colearning_On{}_Site1.txt'.format(measure_names[measure_id]), site1predictions_measure.detach().cpu().numpy())
                np.savetxt('MeasurePredictions_Colearning_On{}_Site2.txt'.format(measure_names[measure_id]), site2predictions_measure.detach().cpu().numpy())
                np.savetxt('Reconstructions_Colearning_On{}.txt'.format(measure_names[measure_id]), x_hat.detach().cpu().numpy())
                np.savetxt('SiteLabels_Colearning_On{}.txt'.format(measure_names[measure_id]), batch_sitelabels.detach().cpu().numpy())
                
        
        
    PATH = '/home-local/LearningConnectomeInvariance/Models/Model_EPOCH500_allON.pt'
    #PATH = '/home-local/LearningConnectomeInvariance/Models/Model_bootstrap_on_{}_kappa_{}_addonetesting.pt'.format(measure_name, kappa)
    experiment = ''.join(str(x) for x in measures_on)
    torch.save(model.state_dict(), PATH)
    
    report_train = ["EXPERIMENT:",str(iteration),", ",str(kappa), "Cohen's D Site 1", str(site1predictions_cohensd), "Cohen's D Site 2", str(site2predictions_cohensd), "Cohen's D Baseline",predictingnative_cohensd, "Sex Acc",str(training_loss_sexacc), "Sex BCE",str(training_loss_sex), "KL", str(training_loss_kldivergence), "Training Sex Prob", str(train_sex), "Age Error", str(training_loss_age)]
    validation_loss_reconstruction = validation_loss_reconstruction / len(validation_loader)
    validation_loss_prediction_site1 = validation_loss_prediction_site1 / countofsite1
    validation_loss_prediction_site2 = validation_loss_prediction_site2 / countofsite2
    validation_loss_kl = validation_loss_kl / len(validation_loader)
    validation_loss_age = validation_loss_age / len(validation_loader)
    validation_loss_sex = validation_loss_sex / len(validation_loader)
    validation_loss_total = validation_loss_prediction_site1 + validation_loss_prediction_site2 + validation_loss_kl + validation_loss_reconstruction + validation_loss_age + validation_loss_sex
    np.savetxt('Training_Report_On{}_iteration{}_Experiment{}.txt'.format(measure_name, iteration, experiment), report_train, fmt='%s')
    report_df_data = {
        'Iteration': [iteration],
        'Experiment':[experiment],
        'Sex Acc':[validation_loss_sexacc.detach().cpu().numpy()],
        'Age Error':[validation_loss_age],
        'Prediction Error Site 1':[validation_loss_prediction_site1.detach().cpu().numpy()],
        'Prediction Error Site 2':[validation_loss_prediction_site2.detach().cpu().numpy()]           
    }
    print(report_df_data)
    report_df=pd.DataFrame(report_df_data)
    report_df.to_csv('Validation_Report_iteration{}_experiment{}_allmeasures.csv'.format(iteration, experiment))
    
    

