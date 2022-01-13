# %% [code]
# parsing the FASTA file, codes from https://colab.research.google.com/github/wouterboomsma/pml_vae_project/blob/main/protein_vae_data_processing.ipynb
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from Bio import SeqIO
from scipy.stats import spearmanr
        
# Mapping from amino acids to integers
aa1_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6,
                'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12,
                'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                'Y': 19, 'X':20, 'Z': 21, '-': 22}
aa1 = "ACDEFGHIKLMNPQRSTVWYXZ-"

phyla = ['Acidobacteria', 'Actinobacteria', 'Bacteroidetes',
         'Chloroflexi', 'Cyanobacteria', 'Deinococcus-Thermus',
         'Firmicutes', 'Fusobacteria', 'Proteobacteria', 'Other']

def get_data(data_filename, calc_weights=False, weights_similarity_threshold=0.8):
    '''Create dataset from FASTA filename'''
    ids = []
    labels = []
    seqs = []
    label_re = re.compile(r'\[([^\]]*)\]')
    for record in SeqIO.parse(data_filename, "fasta"):
        ids.append(record.id)       
        seqs.append(np.array([aa1_to_index[aa] for aa in str(record.seq).upper().replace('.', '-')]))
        
        label = label_re.search(record.description).group(1)
        # Only use most common classes
        if label not in phyla:
            label = 'Other'
        labels.append(label)
                
    seqs = torch.from_numpy(np.vstack(seqs))
    labels = np.array(labels)
    
    phyla_lookup_table, phyla_idx = np.unique(labels, return_inverse=True)

    dataset = torch.utils.data.TensorDataset(*[seqs, torch.from_numpy(phyla_idx)])
    
    weights = None
    
    if calc_weights is not False:

        # Experiencing memory issues on colab for this code because pytorch doesn't
        # allow one_hot directly to bool. Splitting in two and then merging.
        # one_hot = F.one_hot(seqs.long()).to('cuda' if torch.cuda.is_available() else 'cpu')
        one_hot1 = F.one_hot(seqs[:len(seqs)//2].long()).bool()
        one_hot2 = F.one_hot(seqs[len(seqs)//2:].long()).bool()
        one_hot = torch.cat([one_hot1, one_hot2]).to('cuda' if torch.cuda.is_available() else 'cpu')
        assert(len(seqs) == len(one_hot))
        del one_hot1
        del one_hot2
        one_hot[seqs>19] = 0
        flat_one_hot = one_hot.flatten(1)

        weights = []
        weight_batch_size = 1000
        flat_one_hot = flat_one_hot.float()
        for i in range(seqs.size(0) // weight_batch_size + 1):
            x = flat_one_hot[i * weight_batch_size : (i + 1) * weight_batch_size]
            similarities = torch.mm(x, flat_one_hot.T)
            lengths = (seqs[i * weight_batch_size : (i + 1) * weight_batch_size] <=19).sum(1).unsqueeze(-1).to('cuda' if torch.cuda.is_available() else 'cpu')
            w = 1.0 / (similarities / lengths).gt(weights_similarity_threshold).sum(1).float()
            weights.append(w)
            
        weights = torch.cat(weights)
        neff = weights.sum()

    return dataset, weights

def get_baseline_data(data_filename, calc_weights=False, weights_similarity_threshold=0.8):
    ids = []
    labels = []
    seqs = []
    label_re = re.compile(r'\[([^\]]*)\]')
    for record in SeqIO.parse(data_filename, "fasta"):
        ids.append(record.id)       
        seqs.append(np.array([aa1_to_index[aa] for aa in str(record.seq).upper().replace('.', '-')]))
        
        label = label_re.search(record.description).group(1)

        if label not in phyla:
            label = 'Other'
        labels.append(label)
                
    seqs =np.vstack(seqs)
    labels = np.array(labels)
    seqs_tensor = torch.from_numpy(seqs)

    phyla_lookup_table, phyla_idx = np.unique(labels, return_inverse=True)
    dataset = torch.utils.data.TensorDataset(*[seqs_tensor, torch.from_numpy(phyla_idx)])
    
    weights = None
    if calc_weights is not False:

        # Experiencing memory issues on colab for this code because pytorch doesn't
        # allow one_hot directly to bool. Splitting in two and then merging.
        # one_hot = F.one_hot(seqs_tensor.long()).to('cuda' if torch.cuda.is_available() else 'cpu')
        one_hot1 = F.one_hot(seqs_tensor[:len(seqs_tensor)//2].long()).bool()
        one_hot2 = F.one_hot(seqs_tensor[len(seqs_tensor)//2:].long()).bool()
        one_hot = torch.cat([one_hot1, one_hot2]).to('cuda' if torch.cuda.is_available() else 'cpu')
        assert(len(seqs_tensor) == len(one_hot))
        del one_hot1
        del one_hot2
        one_hot[seqs_tensor>19] = 0
        flat_one_hot = one_hot.flatten(1)

        weights = []
        weight_batch_size = 1000
        flat_one_hot = flat_one_hot.float()
        for i in range(seqs_tensor.size(0) // weight_batch_size + 1):
            x = flat_one_hot[i * weight_batch_size : (i + 1) * weight_batch_size]
            similarities = torch.mm(x, flat_one_hot.T)
            lengths = (seqs_tensor[i * weight_batch_size : (i + 1) * weight_batch_size] <=19).sum(1).unsqueeze(-1).to('cuda' if torch.cuda.is_available() else 'cpu')
            w = 1.0 / (similarities / lengths).gt(weights_similarity_threshold).sum(1).float()
            weights.append(w)
            
        weights = torch.cat(weights)
        neff = weights.sum()
    return seqs, labels, weights, phyla_lookup_table, phyla_idx, dataset


def read_experimental_data(filename, alignment_data, measurement_col_name = '2500', sequence_offset=0):
    '''Read experimental data from csv file, and check that amino acid match those 
       in the first sequence of the alignment.
       
       measurement_col_name specifies which column in the csv file contains the experimental 
       observation. In our case, this is the one called 2500.
       
       sequence_offset is used in case there is an overall offset between the
       indices in the two files.
       '''
    
    measurement_df = pd.read_csv(filename, delimiter=',', usecols=['mutant', measurement_col_name])
    
    wt_sequence, wt_label = alignment_data[0]
    
    zero_index = None
    
    experimental_data = {}
    for idx, entry in measurement_df.iterrows():
        mutant_from, position, mutant_to = entry['mutant'][:1],int(entry['mutant'][1:-1]),entry['mutant'][-1:]  
        
        # Use index of first entry as offset (keep track of this in case 
        # there are index gaps in experimental data)
        if zero_index is None:
            zero_index = position
            
        # Corresponding position in our alignment
        seq_position = position-zero_index+sequence_offset
            
        # Make sure that two two inputs agree on the indices: the 
        # amino acids in the first entry of the alignment should be 
        # identical to those in the experimental file.
        assert mutant_from == aa1[wt_sequence[seq_position]]  
        
        if seq_position not in experimental_data:
            experimental_data[seq_position] = {}
        
        # Check that there is only a single experimental value for mutant
        assert mutant_to not in experimental_data[seq_position]
        
        experimental_data[seq_position]['pos'] = seq_position
        experimental_data[seq_position]['WT'] = mutant_from
        experimental_data[seq_position][mutant_to] = entry[measurement_col_name]
    
    experimental_data = pd.DataFrame(experimental_data).transpose().set_index(['pos', 'WT'])
    return experimental_data



def check_reconstruct(dataset, model):
    acc = []
    for i in range(len(dataset)):
        raw_sequence = dataset[i][0][np.newaxis, :].to('cuda' if torch.cuda.is_available() else 'cpu')
        z_mu, _ = model.encoder(raw_sequence)
        acc.append(torch.argmax(model.decoder(z_mu), dim=-1) == raw_sequence)
    acc = np.mean(torch.cat(acc, dim=0).cpu().numpy())
    return acc

def quantitative_assessment(dataset, model, experimental_data, sample_size):
    raw_sequence = dataset[0][0][np.newaxis, :].to('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_value = []
    predicted_value = []
    with torch.no_grad():
        log_x_wt_ELBO, _, _ = model(raw_sequence, sample_size)
        for (position, mutant_from), row in experimental_data.iterrows():
            assert aa1_to_index[mutant_from] == raw_sequence[0, position]
            for mutant_to, exp_value in row.iteritems():
                if mutant_to != mutant_from:
                    new_sequence = raw_sequence.clone()
                    new_sequence[0, position] = aa1_to_index[mutant_to]
                    experiment_value.append(exp_value)
                    log_x_mt_ELBO, _, _ = model(new_sequence, sample_size)
                    predicted_value.append((log_x_mt_ELBO - log_x_wt_ELBO).item())
    return spearmanr(experiment_value, predicted_value)

def check_reconstruct_Dirichlet(dataset, model):
    acc = []
    for i in range(len(dataset)):
        raw_sequence = dataset[i][0][np.newaxis, :].to('cuda' if torch.cuda.is_available() else 'cpu')
        z_mu = model.encoder(raw_sequence)
        acc.append(torch.argmax(model.decoder(z_mu), dim=-1) == raw_sequence)
    acc = np.mean(torch.cat(acc, dim=0).cpu().numpy())
    return acc

def quantitative_assessment_Dirichlet(dataset, model, experimental_data, alpha, sample_size):
    raw_sequence = dataset[0][0][np.newaxis, :].to('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_value = []
    predicted_value = []
    with torch.no_grad():
        log_x_wt_ELBO, _, _ = model(raw_sequence,alpha,sample_size)
        for (position, mutant_from), row in experimental_data.iterrows():
            assert aa1_to_index[mutant_from] == raw_sequence[0, position]
            for mutant_to, exp_value in row.iteritems():
                if mutant_to != mutant_from:
                    new_sequence = raw_sequence.clone()
                    new_sequence[0, position] = aa1_to_index[mutant_to]
                    experiment_value.append(exp_value)
                    log_x_mt_ELBO, _, _ = model(new_sequence, alpha, sample_size)
                    predicted_value.append((log_x_mt_ELBO - log_x_wt_ELBO).item())
    return spearmanr(experiment_value, predicted_value)


def check_reconstruct_iwae(dataset, model):
    acc = []
    for i in range(len(dataset)):
        raw_sequence = dataset[i][0][np.newaxis, :].to('cuda' if torch.cuda.is_available() else 'cpu')
        z_mu, _ = model.encoder(raw_sequence)
        acc.append(torch.argmax(model.decoder(z_mu), dim=-1) == raw_sequence)
    acc = np.mean(torch.cat(acc, dim=0).cpu().numpy())
    return acc

def quantitative_assessment_iwae(dataset, model, experimental_data, sample_size_l, sample_size_k):
    raw_sequence = dataset[0][0][np.newaxis, :].to('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_value = []
    predicted_value = []
    with torch.no_grad():
        log_x_wt_ELBO, _, _ = model(raw_sequence, sample_size_l, sample_size_k)
        for (position, mutant_from), row in experimental_data.iterrows():
            assert aa1_to_index[mutant_from] == raw_sequence[0, position]
            for mutant_to, exp_value in row.iteritems():
                if mutant_to != mutant_from:
                    new_sequence = raw_sequence.clone()
                    new_sequence[0, position] = aa1_to_index[mutant_to]
                    experiment_value.append(exp_value)
                    log_x_mt_ELBO, _, _ = model(new_sequence, sample_size_l, sample_size_k)
                    predicted_value.append((log_x_mt_ELBO - log_x_wt_ELBO).item())
    return spearmanr(experiment_value, predicted_value)