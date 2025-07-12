'''
main_single.py
Mert-chan 16 Feb 2025
Single-GPU (or CPU) version for model trainings
For different models,
    1- Change yaml file path line  " config_dict = load_config("/usr1/home/s124mdg45_01/Projects/PhD/Ionosphere/Python/IonoBench/configs/SimVP.yaml")  # Load YAML config file FIX HERE"
    2- model = ... line in the main() function
'''

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1) Import your configs and data loading
from loadConfigs import *
from data import load_training_data
from source.myDataFuns import SI_categorize
from source.models.SwinLSTM import SwinLSTM_B
from source.models.SimVPv2 import SimVPv2_Model
from source.models.DCNN121 import DCNN121
from Projects.IonoBenchv1.source.myTrainFuns import *


def main():
    seedME(3)  # Seed everything for reproducibility
    print("Starting main() (non-DDP version)...")
    torch.cuda.empty_cache()  # Clear CUDA cache 
    ########################################
    # 1) Device config 
    #===============================================================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using device: {device}, name={torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (no GPU available).")
    #===============================================================================

    
    # 2) Load Initial configs
    #===============================================================================
    config_dict = load_config("/usr1/home/s124mdg45_01/Projects/PhD/Ionosphere/Python/IonoBench/configs/SwinLSTM.yaml")      # Load YAML config file
    # config_dict = load_config("/usr1/home/s124mdg45_01/Projects/PhD/Ionosphere/Python/IonoBench/configs/DCNN121.yaml")      # Load YAML config file
    # config_dict = load_config(r"/usr1/home/s124mdg45_01/Projects/PhD/Ionosphere/Python/IonoBench/configs/SimVPStratified_AllFeatures.yaml")      # Load YAML config file
    # Initialize configs
    dataConfig = DataConfig(config_dict["data"])
    modelConfig = get_model_config(config_dict["model"])  # Automatically selects model config
    testConfig = TestConfig(config_dict["test"])
    #===============================================================================

    
    # 3) Load data
    #===============================================================================
    dataDict = load_training_data(seq_len=dataConfig.seq_len, pred_horz=dataConfig.pred_horz, datasplit=dataConfig.dataSplit, selectedidx=dataConfig.selectedOMNI )
    if not dataDict:
        print("Data not loaded. Exiting.")
        return 
    #===============================================================================
    
    # 4) Extract dataDict
    #===============================================================================
    dates = dataDict["dates"]
    normTEC = dataDict["normTEC"]
    normOMNI = dataDict["normOMNI"]
    dataSplitDict = dataDict["dataSplit_Dates"]  # => train/valid/test
    NUM_OMNI = dataDict["NUM_OMNI"]
    stormInfo = dataDict["stormInfo"]
    testConfig.max_tec = dataDict["maxTEC"]
    testConfig.min_tec = dataDict["minTEC"]
    print("Loaded data and splits successfully.")
    
    B, T, C, H, W = dataConfig.batch_size, dataConfig.seq_len, NUM_OMNI+1, normTEC.shape[1], normTEC.shape[2]
    modelConfig.input_shape = (T, C, H, W)
    print(f"Batch size: {B}, T: {T}, C: {C}, H: {H}, W: {W}")
    #===============================================================================
    
    
    # 5) Load Model
    #===============================================================================
    from torchinfo import summary
    # model = SimVP_Model(modelConfig).to(device)
    model = SwinLSTM_B(modelConfig).to(device)
    # model = DCNN121(modelConfig).to(device)
    summary(model, input_size=((B, C, T, H, W),(B, dataConfig.pred_horz, H, W)))
    trainConfig = TrainConfig(config_dict["train"],model=model)
    #===============================================================================
    
    # 6) Create Dataloader objects (No DistributedSampler)
    #===============================================================================
    # OMNI_path = r"/usr1/home/s124mdg45_01/Projects/PhD/Data/OMNIdata/OMNI_data_1996to2024.txt"
    # SolarClasses = SI_categorize(dataSplitDict["test"], OMNI_path)
    # print(len(SolarClasses["Very Weak"]), len(SolarClasses["Weak"]), len(SolarClasses["Moderate"]), len(SolarClasses["Intense"]))
    trainset = TECspatiotemporalLoader(normTEC, normOMNI, dataSplitDict["train"], dates, T, dataConfig.pred_horz)
    validset = TECspatiotemporalLoader(normTEC, normOMNI, dataSplitDict["valid"], dates, T, dataConfig.pred_horz)
    testset  = TECspatiotemporalLoader(normTEC, normOMNI, dataSplitDict["test"],  dates, T, dataConfig.pred_horz)
    #===============================================================================
    
    # Create Dataloaders
    #===============================================================================
    train_loader = DataLoader(trainset, batch_size=B, shuffle=True)
    val_loader   = DataLoader(validset, batch_size=B, shuffle=False)
    test_loader  = DataLoader(testset,  batch_size=B, shuffle=False)
    #===============================================================================
    
    # 7) LR Test
    #===============================================================================
    # lr_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]  # Define LR list
    # LRrangetest(model,lr_list, train_loader, device, trainConfig, plot=True)        # Run LR range test
    
    # 8) Train
    #===============================================================================
    # trainConfig.inputNames = dataDict['OMNI_names']
    # trainDict = IonoTrainer(model, train_loader, val_loader, device, trainConfig).train()
    # IonoTrainer.learningCurve_plot(trainDict)
    
    # 9) Test
    #===============================================================================
    checkpoint = torch.load(r'/usr1/home/s124mdg45_01/Projects/PhD/Ionosphere/Python/IonoBench/training_sessions/SwinLSTM_in12_out12/SwinLSTM_in12_out12_best_checkpoint_20250411_0134.pth')
    model.load_state_dict(DDPtoSingleGPU(checkpoint["model_state_dict"]))   # Load the model state dict If DDP was used get rid of Module prefix
    testConfig = TestConfig(config_dict["test"])
    testConfig.inputNames = dataDict['OMNI_names']
    testConfig.saveResults = False
    testDict = IonoTester(model, test_loader, device, testConfig).test()
    stormRes = stormAnalysis(model, stormInfo, dataSplitDict, testConfig, dataConfig,normTEC, normOMNI, dates, T, B, device)
    # storm_results = evaluate_storms(model, stormInfo, dataSplitDict, dataDict, testConfig, dataConfig,
    #                 normTEC, normOMNI, dates, T, B, device)
    #===============================================================================
    
if __name__ == "__main__":
    main()
    ############################
