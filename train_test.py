import os

total_gpus = 8

# GPUs to exclude
exclude_gpus = {0}

# Compute the list of visible GPUs
visible_gpus = ','.join(str(i) for i in range(total_gpus) if i not in exclude_gpus)

# Set the environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus
print(f"Visible GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_TIMEOUT'] = '60'
os.environ['NCCL_DEBUG'] = 'ERROR'
os.environ['NCCL_DEBUG_SUBSYS'] = 'COLL'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from sklearn.metrics import matthews_corrcoef
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau


import random,math
import sys
import numpy as np
import pandas as pd
from transformers import AdamW
# from torch.optim import AdamW

from script.dataloader import *
from script.model import *
from script.MetricsHolder import *

import time

from transformers import logging

from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score

# Suppress warnings and only show errors
logging.set_verbosity_error()


seed_val = 41
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
# GPU training
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoches = 5


NSamples_train = -1  #For whole Dataset
# NSamples_train = 500

NSamples_test = -1  #For whole Dataset
# NSamples_test = 50


pre_path = '.'       #do not include / end of it
dataset_folder = '../Dataset/Chip_Seq_165_Dataset/'




def print_param(file_path):
    with open(file_path, 'w') as f:
        print('Epoches = %d , NSamples_train = %d ,NSamples_test = %d\n\n' %(epoches,NSamples_train,NSamples_test),file=f)
        # if(NFolders==-1):
        #     print(f'Folders: {NameFolder}',file=f)
        print('*'*100,'\n\n\n\n',file=f)

def print_to_console_and_file(message, file_path):
    # Print to console
    print(message)
    # Write to file
    with open(file_path, "a") as file:
        file.write(message + "\n")


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12331"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
class EarlyStopping:
    def __init__(self, patience=2, mode='max', save_path='best_model.pt'):
        """
        Args:
            patience (int): Number of epochs to wait for improvement.
            mode (str): 'max' for metrics that should be maximized (e.g., PR-AUC), 'min' otherwise.
            save_path (str): Path to save the best model.
        """
        self.patience = patience
        self.mode = mode
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score, model, mh :MetricsHolder, metrics, log_file):
        """Checks if early stopping criteria are met and saves the best model."""
        if self.best_score is None:
            self.best_score = score
            mh.set_val_counts(metrics[0],metrics[1],metrics[2],metrics[3])
            mh.set_val_metrics(metrics[4],metrics[5],metrics[6],metrics[7])
            mh.set_val_auc(metrics[8],metrics[9])
            self._save_checkpoint(model,log_file)
        elif (self.mode == 'max' and score <= self.best_score) or (self.mode == 'min' and score >= self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            mh.set_val_counts(metrics[0],metrics[1],metrics[2],metrics[3])
            mh.set_val_metrics(metrics[4],metrics[5],metrics[6],metrics[7])
            mh.set_val_auc(metrics[8],metrics[9])
            self.counter = 0
            self._save_checkpoint(model,log_file)

    def _save_checkpoint(self, model,log_file):
        """Saves the model to the specified path."""
        torch.save(model.module.state_dict(), self.save_path)
        print_to_console_and_file(f"Saved best model to {self.save_path}.",log_file)
    

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn : torch.nn.Module,
        gpu_id: int,
        early_stopping: EarlyStopping,
        world_size : int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.early_stopping = early_stopping
        self.world_size = world_size
        self.model = DDP(model, device_ids=[gpu_id],find_unused_parameters=True)
    
    def _run_batch(self, source, targets, epoch, max_epochs):
        # Clear the accumulated gradients 
        self.optimizer.zero_grad()
        # Data input to the model
        pred = self.model(source)
        # Calculate loss function
        loss = self.loss_fn(pred, targets)
        # Back Propagation
        loss.backward()
        
        # Model weight update
        self.optimizer.step()
        # Calculate metrics for this batch
        return self._calculate_count_metrics(pred, targets)

    def _run_epoch(self, epoch, max_epochs, isTrain=True):
        targetData = self.train_data if isTrain else self.test_data
        if isTrain:
            # b_sz = len(next(iter(self.train_data))[0])
            # print_to_console_and_file(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}",out_filename)
            self.train_data.sampler.set_epoch(epoch)
        else:
            self.test_data.sampler.set_epoch(epoch)
            
        # Initialize metric accumulators
        total_TP, total_TN, total_FP, total_FN, num_batches = 0, 0, 0, 0, 0        
        all_preds, all_targets = [], []
        for source, targets in targetData:
            # print(f'\n********Step: {num_batches}***********\n')
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            TP, TN, FP, FN, preds, targs = self._run_batch(source, targets, epoch, max_epochs)
            
            total_TP += TP
            total_TN += TN
            total_FP += FP
            total_FN += FN
            # num_batches += 1
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(targs.detach().cpu().numpy())
        
        return total_TP, total_TN, total_FP, total_FN, all_preds, all_targets

    def train(self, max_epochs: int, log_file: str, mh: MetricsHolder):
        for epoch in range(max_epochs):
            if self.gpu_id == 0:
                print_to_console_and_file('Starting training',log_file)
                
            self.model.train()
            TP_train, TN_train, FP_train, FN_train, _, _ = self._run_epoch(epoch,max_epochs)
            
            # Convert metrics into compatibale Tensor
            values = [TP_train, TN_train, FP_train, FN_train]
            tensors = [torch.tensor(val, dtype=torch.float32, device=self.gpu_id) for val in values]
            
            # Perform distributed reduce operation
            for tensor in tensors:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            
            TP_train, TN_train, FP_train, FN_train = tensors
            
            if self.gpu_id == 0:
                sum = TP_train.item() + TN_train.item() + FP_train.item() + FN_train.item()
                SN_test, SP_test, ACC_test, MCC_test = self._calculate_test_metrics(TP_train.item(), TN_train.item(), FP_train.item(), FN_train.item())
                print_to_console_and_file(f'\tTrain : Epoch {epoch + 1}, TP = {TP_train.item()}, TN = {TN_train.item()}, FP = {FP_train.item()}, FN = {FN_train.item()}, SUM = {sum}',log_file)
                print_to_console_and_file(f'\tTrain : Epoch {epoch + 1}, Sensitivity = {SN_test:.4f}, Specificity = {SP_test:.4f}, Accuracy = {ACC_test:.4f}, MCC = {MCC_test:.4f}',log_file)
            


            ############################################
            # Starting Validation    
            ############################################
            
            if self.gpu_id == 0:
                print_to_console_and_file('\n\tStarting validation',log_file)
                
            self.model.eval()
            TP_val, TN_val, FP_val, FN_val , preds, targets = self._run_epoch(epoch,max_epochs,isTrain=False)
            
            # Convert metrics into compatibale Tensor
            values = [TP_val, TN_val, FP_val, FN_val]
            tensors = [torch.tensor(val, dtype=torch.float32, device=self.gpu_id) for val in values]
            
            # Perform distributed reduce operation
            for tensor in tensors:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            
            TP_val, TN_val, FP_val, FN_val = tensors
            
            
            # All-gather targets and preds from all GPUs
            preds = torch.tensor(np.array(preds), dtype=torch.float32, device=self.gpu_id)
            targets = torch.tensor(np.array(targets), dtype=torch.float32, device=self.gpu_id)
            gathered_targets = [torch.empty_like(targets) for _ in range(self.world_size)]
            gathered_preds = [torch.empty_like(preds) for _ in range(self.world_size)]
            
            dist.all_gather(gathered_targets, targets)
            dist.all_gather(gathered_preds, preds)

            # Concatenate the gathered results
            final_targets = torch.cat(gathered_targets, dim=0)
            final_preds = torch.cat(gathered_preds, dim=0)
            
            pr_auc, roc_auc = self._calculate_pr_auc_and_roc_auc(final_preds, final_targets)
            self.scheduler.step(pr_auc)
            
            
            if self.gpu_id == 0:
                sum_val = TP_val.item()+TN_val.item()+FP_val.item()+FN_val.item()
                SN_val, SP_val, ACC_val, MCC_val = self._calculate_test_metrics(TP_val.item(), TN_val.item(), FP_val.item(), FN_val.item())
                print_to_console_and_file(f'\t\tValidation : Epoch {epoch + 1}, TP = {TP_val.item()}, TN = {TN_val.item()}, FP = {FP_val.item()}, FN = {FN_val.item()}, SUM = {sum_val}',log_file)
                print_to_console_and_file(f'\t\tValidation : Epoch {epoch + 1}, Sensitivity = {SN_val:.4f}, Specificity = {SP_val:.4f}, Accuracy = {ACC_val:.4f}, MCC = {MCC_val:.4f}',log_file)
                print_to_console_and_file(f'\t\tValidation : Epoch {epoch + 1}, PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}\n',log_file)
                metrics = [TP_val.item(), TN_val.item(), FP_val.item(), FN_val.item(), SN_val, SP_val, ACC_val, MCC_val, pr_auc,roc_auc]
                self.early_stopping(pr_auc, self.model, mh, metrics, log_file)
                
            # if self.early_stopping.early_stop:
            self.check_and_sync_early_stopping()
                
            if self.early_stopping.early_stop:
                if self.gpu_id == 0:
                    str = '\t\t'+'*'*10 + "Early stopping triggered." + '*'*10+'\n\n'
                    print_to_console_and_file(str,log_file)
                break
    
    def test(self, log_file: str, mh: MetricsHolder):
        if self.gpu_id == 0:
            print_to_console_and_file('\n\t\tStarting testing',log_file)
            
            
        self.model.eval()
        TP_test, TN_test, FP_test, FN_test , preds, targets = self._run_epoch(0,1,isTrain=False)
        
        # Convert metrics into compatibale Tensor
        values = [TP_test, TN_test, FP_test, FN_test]
        tensors = [torch.tensor(val, dtype=torch.float32, device=self.gpu_id) for val in values]
        
        # Perform distributed reduce operation
        for tensor in tensors:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        TP_test, TN_test, FP_test, FN_test = tensors
        
        
        # All-gather targets and preds from all GPUs
        preds = torch.tensor(np.array(preds), dtype=torch.float32, device=self.gpu_id)
        targets = torch.tensor(np.array(targets), dtype=torch.float32, device=self.gpu_id)
        gathered_targets = [torch.empty_like(targets) for _ in range(self.world_size)]
        gathered_preds = [torch.empty_like(preds) for _ in range(self.world_size)]
        
        dist.all_gather(gathered_targets, targets)
        dist.all_gather(gathered_preds, preds)

        # Concatenate the gathered results
        final_targets = torch.cat(gathered_targets, dim=0)
        final_preds = torch.cat(gathered_preds, dim=0)
        
        pr_auc, roc_auc = self._calculate_pr_auc_and_roc_auc(final_preds, final_targets)
        
        
        if self.gpu_id == 0:
            sum_val = TP_test.item()+TN_test.item()+FP_test.item()+FN_test.item()
            SN_test, SP_test, ACC_test, MCC_test = self._calculate_test_metrics(TP_test.item(), TN_test.item(), FP_test.item(), FN_test.item())
            print_to_console_and_file(f'\t\t\tTest : TP = {TP_test.item()}, TN = {TN_test.item()}, FP = {FP_test.item()}, FN = {FN_test.item()}, SUM = {sum_val}',log_file)
            print_to_console_and_file(f'\t\t\tTest : Sensitivity = {SN_test:.4f}, Specificity = {SP_test:.4f}, Accuracy = {ACC_test:.4f}, MCC = {MCC_test:.4f}',log_file)
            print_to_console_and_file(f'\t\t\tTest : PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}\n',log_file)
            
            mh.set_test_counts(TP_test.item(),TN_test.item(), FP_test.item(), FN_test.item())
            mh.set_test_metrics(SN_test, SP_test, ACC_test, MCC_test)
            mh.set_test_auc(pr_auc,roc_auc)

    
    def check_and_sync_early_stopping(self):
        # Initialize a tensor to hold the early stop flag
        early_stop_tensor = torch.tensor(
            int(self.early_stopping.early_stop), dtype=torch.int, device=self.gpu_id
        )

        # Use all_reduce to check if any GPU has triggered early stopping
        dist.all_reduce(early_stop_tensor, op=dist.ReduceOp.SUM)

        # If any GPU has early_stop set to True, broadcast it to all
        global_early_stop = early_stop_tensor.item() > 0
        self.early_stopping.early_stop = global_early_stop
    
    def _calculate_count_metrics(self, pred, targets):
        # Convert predictions to binary values
        pred1 = (pred > 0.5).float()
        pred_bin = (pred1.argmax(dim=1) == 1).float()
        targets_bin = (targets > 0.5).float()

        # Calculate confusion matrix components
        TP = (pred_bin * targets_bin).sum().item()  # True Positives
        TN = ((1 - pred_bin) * (1 - targets_bin)).sum().item()  # True Negatives
        FN = ((1 - pred_bin) * targets_bin).sum().item()  # False Negatives
        FP = (pred_bin * (1 - targets_bin)).sum().item()  # False Positives
        
        return TP, TN, FP, FN, pred, targets
    
    def _calculate_test_metrics(self, TP, TN, FP, FN):
    # Calculate the test metrics
        SN = TP / (TP + FN) if (TP + FN) > 0 else 0
        SP = TN / (TN + FP) if (TN + FP) > 0 else 0
        ACC = (TP + TN) / (TP + TN + FP + FN)
        MCC = (TP * TN - FP * FN) / math.sqrt(
            (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)
        ) if (math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))) != 0 else 0
        
        return SN, SP, ACC, MCC
    
    def _reduce_metric(self, metric):
        # Sum metrics across all GPUs
        #dist.all_reduce(torch.tensor(metric, dtype=torch.float32, device=self.gpu_id), op=dist.ReduceOp.SUM)
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        return metric

    

    def _calculate_pr_auc_and_roc_auc(self, preds, targets):
        # Move tensors to CPU and convert to numpy
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        y_score_positive = [score[1] for score in preds]
        
        precision, recall, _ = precision_recall_curve(targets, y_score_positive)
        return auc(recall, precision), roc_auc_score(targets, y_score_positive)

    def _load_checkpoint(self, save_path:str, log_file:str):
        """Loads the model from the specified path."""
        try:
            state_dict = torch.load(save_path,weights_only=True,map_location=f"cuda:{self.gpu_id}")
            self.model.module.load_state_dict(state_dict)
            if self.gpu_id == 0:
                print_to_console_and_file(f"Loaded model from {save_path}.", log_file)
        except FileNotFoundError:
            print_to_console_and_file(f"Checkpoint file not found at {save_path}.", log_file)
        except Exception as e:
            print_to_console_and_file(f"Error loading model: {e}", log_file)

def load_train_objs(train_file,test_file,path_preModel:str):
    # Loading dataset
    dataset = load_dataset_kmer_data(train_file,test_file,path_preModel)  # load your dataset
    input_channel = dataset["input_chanel"]
    
    # Loading model
    bert_blend_cnn = BERTNew(input_channel,path_preModel)
    
    # Select optimizer and loss function
    optimizer = AdamW(bert_blend_cnn.parameters(), lr=1.5e-5, weight_decay=1e-2, no_deprecation_warning=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    return dataset, bert_blend_cnn, optimizer, scheduler, loss_fn




def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, total_epochs: int, batch_size: int,train_file: str, test_file: str, save_model: str, result_file: str,log_file: str, metricHolder:MetricsHolder,path_preModel:str):
    try:
        ddp_setup(rank, world_size)
        try:
            print(f'\tA: {rank}')
            dataset, model, optimizer,scheduler, loss_fn = load_train_objs(train_file,test_file,path_preModel)
            print(f'\tB: {rank}')
            train_data = prepare_dataloader(dataset["train_dataset"], batch_size)
            print(f'\tC: {rank}')
            test_data = prepare_dataloader(dataset["test_dataset"], batch_size)
            print(f'\tD: {rank}')
            early_stopping = EarlyStopping(patience=2, mode='max', save_path=save_model)
            print(f'\tE: {rank}')

            trainer = Trainer(model, train_data, test_data, optimizer,scheduler, loss_fn, rank, early_stopping,world_size)
            print(f'\tF: {rank}')
            # Model Train and Validation
            trainer.train(total_epochs,log_file,metricHolder)
            print(f'\tG: {rank}')
            
            # Model Testing
            trainer._load_checkpoint(save_model,log_file)
            print(f'\tH: {rank}')
            trainer.test(log_file,metricHolder)
            print(f'\tI: {rank}')
            
            # Print metrics into Output file
            if rank == 0:
                with open(result_file,'a') as out_file:
                    content = ', '.join(f"{value}" for _, value in metricHolder.to_dict().items())
                    out_file.write(content+'\n')
            print(f'\tJ: {rank}')
        except Exception as e:
            print(f'Error occured...{e}')
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        
    finally:
        destroy_process_group()

def load_dataset(train_file,test_file):
    train_data = pd.read_csv(train_file)
    train_data.columns = train_data.columns.str.lower()
    if NSamples_test!=-1:
        train_data = train_data[0:NSamples_train]
    train_sentences = train_data["sequence"]
    train_labels = train_data["label"]
    # Test_dataset
    test_data = pd.read_csv(test_file)
    test_data.columns = test_data.columns.str.lower()
    if NSamples_test!=-1:
        test_data = test_data[0:NSamples_test]

    test_sentences = test_data["sequence"]
    test_labels = test_data["label"]
    # Transformed into token and add padding
    train_inputs, train_labels, test_inputs, test_labels = input_token(train_sentences, train_labels,
                                                                        test_sentences, test_labels)
    # Calculate the length after adding padding
    input_channel = len(train_inputs[1])-2
    train_dataset = MyDataset(train_inputs, train_labels)
    test_dataset = MyDataset(test_inputs, test_labels)
    
    # print(f'len train dataset:{len(train_sentences)},{len(train_labels)}')
    # print(f'len test dataset:{len(test_sentences)},{len(test_labels)}')
    return {"train_dataset":train_dataset, "test_dataset":test_dataset, "input_chanel":input_channel}

def load_dataset_kmer_data(train_file,test_file,path_preModel:str):
    train_data = pd.read_csv(train_file,index_col=None,sep=',',header=None)
    if NSamples_test!=-1:
        train_data = train_data[0:NSamples_train]
    train_data.columns =  ["sequence","label"]
    train_sentences = train_data["sequence"]
    train_labels = train_data["label"]
    # Test_dataset
    test_data = pd.read_csv(test_file,index_col=None,sep=',',header=None)
    if NSamples_test!=-1:
        test_data = test_data[0:NSamples_test]
    test_data.columns =  ["sequence","label"]

    test_sentences = test_data["sequence"]
    test_labels = test_data["label"]
    # Transformed into token and add padding
    train_inputs, train_labels, test_inputs, test_labels = input_token(train_sentences, train_labels,
                                                                        test_sentences, test_labels,path_preModel)
    # Calculate the length after adding padding
    input_channel = len(train_inputs[1]) - 2
    train_dataset = MyDataset(train_inputs, train_labels)
    test_dataset = MyDataset(test_inputs, test_labels)
    
    # print(f'len train dataset:{len(train_sentences)},{len(train_labels)}')
    # print(f'len test dataset:{len(test_sentences)},{len(test_labels)}')
    return {"train_dataset":train_dataset, "test_dataset":test_dataset, "input_chanel":input_channel}

def select_file(folder, keyword:str, kmer:int):
    # Convert inputs to lowercase for case-insensitive matching
    keyword_lower = keyword.lower()
    kmer_lower = 'kmer'+str(kmer)
    
    # List all files in the specified folder
    files = os.listdir(folder)
    
    # Filter files based on the keyword and kmer string
    selected_files = [file for file in files if keyword_lower in file.lower() and kmer_lower in file.lower()]
    
    # Return the first match or None if no file matches
    return selected_files[0] if selected_files else None

def check_and_create_folders(path, kmer):
    # List of folders to check/create
    folder_paths = [
        f'{path}/logs',
        f'{path}/logs/kmer{kmer}',
        f'{path}/Saved_models',
        f'{path}/Results'
    ]

    # Check for existing folders
    existing_folders = [folder for folder in folder_paths if os.path.exists(folder)]
    
    if existing_folders:
        print("\n\n\t\t\tThe following folders already exist and contents will be overwritten:")
        for folder in existing_folders:
            print(f"\t\t\t- {folder}")
        
        # Prompt user for confirmation
        proceed = input("\n\t\t\tDo you want to proceed? (y/n): ").strip().lower()
        if proceed != 'y':
            print("\nOperation aborted.")
            quit()
    
    # Create folders if they don't exist
    for folder in folder_paths:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

def choose_Bert_Embedding_path(kmer):
    if kmer == 2:
        path_preModel ='../pre-models/DNABERT-2/'
    if kmer == 3:
        path_preModel ='../pre-models/DNABERT-3/'
    if kmer == 4:
        path_preModel ='../pre-models/DNABERT-4/'
    if kmer == 5:
        path_preModel ='../pre-models/DNABERT-5/'
    if kmer == 6:
        path_preModel ='../pre-models/DNABERT-6/'
    
    print(f'\tChosen Pre Model Path: {path_preModel}\n')
    return path_preModel

if __name__ == "__main__":
    start_time = time.now()
    import argparse
    parser = argparse.ArgumentParser(description='BERT-TFBS (new)')
    parser.add_argument('-e','--total_epochs', default =15, type=int, help='Total epochs to train the model')
    parser.add_argument('-k','--K_MER', default=6, type=int, help='How often to save a snapshot')
    parser.add_argument('-b','--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('-w','--world_size', default=torch.cuda.device_count(), type=int, help='Number of GPU')
    parser.add_argument('-p','--folder_index_start', default=1, type=int, help='Starting Index of Dataset Folder')
    parser.add_argument('-q','--folder_index_end', default=None, type=int, help='Ending Index of Dataset Folder')
    parser.add_argument('-r','--sub_folder_index_start', default=1, type=int, help='Starting Index of Sub Folder for \'pth\' folder in Dataset')
    
    
    
    args = parser.parse_args()
    
    kMer = args.K_MER
    epoches = args.total_epochs
    world_size = args.world_size
    p = args.folder_index_start
    q = args.folder_index_end
    r = args.sub_folder_index_start
    
    check_and_create_folders(pre_path, kMer)
    path_preModel = choose_Bert_Embedding_path(kMer)
    
    metricHolder = MetricsHolder()
    
    result_filename = f'{pre_path}/Results/results_KMER_{kMer}.csv'
    if os.path.exists(result_filename):
        response = input(f"\n\nFile '{result_filename}' already exists. Enter your choice (r: replace, a: append, x: exit): ").strip().lower()
        if response == "r":
            print(f"Replacing the file '{result_filename}'.")
            with open(result_filename,'w') as f:
                content = ', '.join(f"{key}" for key, _ in metricHolder.to_dict().items())
                f.write(content+'\n')
        elif response == "a":
            print(f"Appending the file '{result_filename}'.")
            with open(result_filename,'a') as f:
                content = f'Appending File Information with -p: {p}, -q:{q}, -r{r}'
                content = '\n\n'+'*'*len(content)+content+'*'*len(content)+'\n\n'
                f.write(content)
        else:
            print(f"File '{result_filename}' will not be replaced.")
            quit()
    else:
        with open(result_filename,'w') as f:
            content = ', '.join(f"{key}" for key, _ in metricHolder.to_dict().items())
            f.write(content+'\n')
        
    print(f'\n\n\t\t\tOptions : Total Epochs: {epoches}, Batch_size : {args.batch_size}, #GPU\'s : {world_size}, KMER : {kMer}, p : {p}, q : {q}, r : {r},')
    print(f'\t\t\tResults saving to file : {result_filename}\n\n')
    
    folders = sorted([item for item in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, item))])
    for index, folder_name in enumerate(folders[p-1:q], start=p):
        log_filename = f'{pre_path}/logs/kmer{kMer}/log_KMER_{kMer}_{folder_name}.txt'
        print(f'\t\t\tLog-outout Saving to file : {log_filename}\n\n')
        
        mystr = f'\n\nFolder: {folder_name}, Folder Progress: {index}/{len(folders)}\n'+'*'*50+'\n'
        print_to_console_and_file(mystr,log_filename)
        
        folder_path = os.path.join(dataset_folder, folder_name.lower())
        subfolders = sorted([item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))])
        for index2, subfolder in enumerate(subfolders[r-1:None],start=r):
            r = 1
            mystr = f'\t{index2}/{len(subfolders)} || Subfolder: {subfolder}, || Folder: {folder_name}[{index}/{len(folders)}]'
            print_to_console_and_file(mystr,log_filename)
            
            metricHolder.reset()
            metricHolder.set_folderName(folder_name)
            metricHolder.set_subFolderName(subfolder)
            
            temp_path = folder_path + '/' + subfolder + '/' + subfolder
            train_file = os.path.join(temp_path, select_file(temp_path,'train',kMer))
            test_file = os.path.join(temp_path, select_file(temp_path,'test',kMer))

            save_model_name = f'{pre_path}/Saved_models/best_model_{folder_name}_{subfolder}_kmer{kMer}.pt'

            mp.spawn(main, args=(world_size, epoches, args.batch_size,train_file,test_file,save_model_name,result_filename,log_filename,metricHolder,path_preModel), nprocs=world_size, join=True)
            if 'taf1' not in folder_name:
                if os.path.exists(save_model_name):
                    os.remove(save_model_name)
            
            mystr = '\t'+'='*50
            print_to_console_and_file(mystr,log_filename)

        mystr = '\n\n\n\n\n'
        print_to_console_and_file(mystr,log_filename)
    end_time = time.time()
    print("Done....")
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
    
