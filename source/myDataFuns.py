
#####################################################################
"""
data.py
Mert-chan 13 Feb 2025
- Functions for general purposes
- OMNI data
- date lists
- Solar Intensity categorization
- storm information
"""
#####################################################################
# from myVisualFuns import *
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import shutil, os, json
from huggingface_hub import hf_hub_download
LAT = 72
LONG = 72

############################################################################################################
def dateList(start, end,delta):
    dates = []
    i = 0   
    while start <= end:
        dates.append(start)
        start = start + delta
    return dates
############################################################################################################

#############################################################
def List_OMNI_Indices(OMNI_path, verbose=True):
#############################################################
    # Function Explanation
    #===============================================================================
    # This code returns the listed indeces inside the OMNI data file: Data downloaded from 'https://omniweb.gsfc.nasa.gov/form/dx1.html'
    # To be able to use this scirpt,
    # 1 -- first go to https://omniweb.gsfc.nasa.gov/form/dx1.html site and choose the indices and time interval you want to download.
    # 2 -- Select "List data" and press Ctrl A to select all the data and Ctrl C to copy it to a .txt file
    # 3 -- Load the .txt file into the script by specifying the path to the file in the "OMNI_path" variable
    # Inputs: OMNI_path ---> Location of OMNI data
    # Outputs: indices ---> List of indices found in the OMNI data
    #===============================================================================

    #===============================================================================
    indices = []  # initialize an empty list to store the indices names
    try:
        with open(OMNI_path, "r") as file:
            for i, line in enumerate(file):
                if i > 3:  # skip the first 4 lines
                    if line.strip() == "":  # stop reading when an empty line is encountered
                        k = -1
                        next_line = next(file).strip()  # read the next line after the empty line
                        for index_name in next_line.split():
                            k += 1
                            if not index_name.isnumeric():
                                indices.insert(k, index_name)  # store the indices names as a list of strings at the beginning of the list
                        break
                    indices.append(line[3:].strip())  # store the indices names as a list of strings
    except FileNotFoundError:
        print(f"File '{OMNI_path}' not found.")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
        exit()
    if verbose:                    
        print("The found indices in the file are: ")
        print(indices)
    return indices
    #===============================================================================
#############################################################

#############################################################
def OMNI_impute_advanced(OMNI_df):
    # Example usage
    # Checks the special missing data headers on the OMNIweb data 
    # Replaces missing value with closes value
    # If it is close to both one after and before element takes average
    # If no close value it finds the subsitute by checking the distriubtion of the data.
    missing_values_dict = {
        'Alpha/Prot. ratio':  9.999,
        'HR': 999,
        'YEAR': 9999,
        'DOY': 999,
        'Lyman_alpha': 0.999999,
        'Kp index': 99,
        'Flow pressure': 99.99,
        'E elecrtric field': 999.99,
        'R (Sunspot No.)': 999,
        'ap_index, nT': 999,
        'pc-index': 999.9,
        'f10.7_index': 999.9,
        'BZ, nT (GSE)': 999.9,
        'BY, nT (GSE)': 999.9,
        'BX, nT (GSE, GSM)': 999.9,
        'BZ, nT (GSM)': 999.9,
        'BY, nT (GSM)': 999.9,
        'Scalar B, nT': 999.9,
        'Vector B Magnitude,nT': 999.9,
        'Lat. Angle of B (GSE)': 999.9,
        'Long. Angle of B (GSE)': 999.9,
        'AE-index, nT': 999.9,
        'SW Proton Density, N/cm^3': 999.9,
        'SW Plasma flow lat. angle': 999.9,
        'SW Plasma flow long. angle': 999.9,
        'Alfen mach number': 999.9,
        'AE-index, nT': 9999,
        'SW Plasma Speed, km/s': 9999,
        'Dst-index, nT': 99999,
        'AL-index, nT': 99999,
        'AU-index, nT': 99999,
        'pc-index': 999.9,
        'SW Plasma Temperature, K': 9999999
    }

    for col in OMNI_df.columns:
        missing_value = missing_values_dict[col]            # Find the 
        col_data = OMNI_df[col]
        
        for i in range(1, len(col_data)-1):
            if col_data[i] == missing_value:
                if col_data[i-1] != missing_value and col_data[i+1] == missing_value:
                    OMNI_df.at[i, col] = col_data[i-1]
                elif col_data[i+1] != missing_value and col_data[i-1] == missing_value:
                    OMNI_df.at[i, col] = col_data[i+1]
                elif col_data[i-1] != missing_value and col_data[i+1] != missing_value:
                    OMNI_df.at[i, col] = (col_data[i-1] + col_data[i+1]) / 2
                else:
                    mean_val = np.mean(col_data[col_data != missing_value])
                    std_val = np.std(col_data[col_data != missing_value])
                    OMNI_df.at[i, col] = np.random.normal(mean_val, std_val)
                    
    return OMNI_df
##############################################################

###################################################################
def group_OMNI_data(OMNI_data):
 # Define the categories and their respective columns
    time_idx = [
        'YEAR', 'DOY', 'HR', 'HR_sin', 'DoY_sin'
    ]
    solar_idx = [
        'R (Sunspot No.)', 'f10.7_index', 'Lyman_alpha'
    ]
    geomagnetic_idx = [
        'Kp index', 'Dst-index, nT', 'ap_index, nT', 'AE-index, nT', 'AL-index, nT', 'AU-index, nT', 'pc-index'
    ]
    plasma_parameters = [
        'SW Plasma Temperature, K', 'SW Proton Density, N/cm^3', 'SW Plasma Speed, km/s', 'SW Plasma flow long. angle',
        'SW Plasma flow lat. angle', 'Alpha/Prot. ratio', 'sigma-T, K', 'sigma-n, N/cm^3', 'sigma-V, km/s',
        'sigma-phi V, degrees', 'sigma-theta V, degrees', 'sigma-ratio'
    ]
    derived_parameters = ['Flow pressure','E elecrtric field','Alfen mach number']
    imf_parameters = [
        'Scalar B, nT', 'Vector B Magnitude,nT', 'Lat. Angle of B (GSE)', 'Long. Angle of B (GSE)', 'BX, nT (GSE, GSM)',
        'BY, nT (GSE)', 'BZ, nT (GSE)', 'BY, nT (GSM)', 'BZ, nT (GSM)','RMS_magnitude, nT', 'RMS_field_vector, nT', 'RMS_BX_GSE, nT',
        'RMS_BY_GSE, nT', 'RMS_BZ_GSE, nT'
    ]

    # Combine all categories
    all_columns = solar_idx + geomagnetic_idx + plasma_parameters + derived_parameters + imf_parameters + time_idx

    # Create a new column order list based on existing columns in the dataframe
    new_column_order = [col for col in all_columns if col in OMNI_data.columns]

    # Reorder the dataframe
    df_reordered = OMNI_data[new_column_order]

        # Create a mapping of original column names to shorter names
    column_name_mapping = {
        'R (Sunspot No.)': 'SSN',
        'f10.7_index': 'F10.7',
        'Lyman_alpha': 'Lyman Alpha',
        'Kp index': 'Kp',
        'Dst-index, nT': 'Dst',
        'ap_index, nT': 'ap',
        'AE-index, nT': 'AE',
        'AL-index, nT': 'AL',
        'AU-index, nT': 'AU',
        'pc-index': 'pc',
        'SW Plasma Temperature, K': 'SW Temp',
        'SW Proton Density, N/cm^3': 'SW Density',
        'SW Plasma Speed, km/s': 'SW Speed',
        'SW Plasma flow long. angle': 'SW Long Angle',
        'SW Plasma flow lat. angle': 'SW Lat Angle',
        'Flow pressure': 'Flow Pressure',
        'E elecrtric field': 'E Field',
        'Alfen mach number': 'Ma',
        'Alpha/Prot. ratio': 'Alpha/Prot',
        'sigma-T, K': 'Sigma Temp',
        'sigma-n, N/cm^3': 'Sigma Density',
        'sigma-V, km/s': 'Sigma Speed',
        'sigma-phi V, degrees': 'Sigma Phi V',
        'sigma-theta V, degrees': 'Sigma Theta V',
        'sigma-ratio': 'Sigma Ratio',
        'Scalar B, nT': 'Scalar B',
        'Vector B Magnitude,nT': 'Vector B',
        'Lat. Angle of B (GSE)': 'Lat Angle B',
        'Long. Angle of B (GSE)': 'Long Angle B',
        'BX, nT (GSE, GSM)': 'BX (GSE,GSM)',
        'BY, nT (GSE)': 'BY (GSE)',
        'BZ, nT (GSE)': 'BZ (GSE)',
        'BY, nT (GSM)': 'BY (GSM)',
        'BZ, nT (GSM)': 'BZ (GSM)',
        'RMS_magnitude, nT': 'RMS Magnitude',
        'RMS_field_vector, nT': 'RMS Vector',
        'RMS_BX_GSE, nT': 'RMS BX (GSE)',
        'RMS_BY_GSE, nT': 'RMS BY (GSE)',
        'RMS_BZ_GSE, nT': 'RMS BZ (GSE)',
        'YEAR': 'Year',
        'DOY': 'Day of Year',
        'HR': 'Hour',
        'HR_sin': 'Hour Sin',
        'DoY_sin': 'Day Sin'
    }

    return df_reordered.rename(columns=column_name_mapping)  # Apply the name changes to the DataFrame columns
###################################################################

#############################################################
def final_OMNI_datav2(OMNI_path, indices, indexMatrix, dates, verbose=True):
    # Function Explanation
    #===============================================================================
    # This code returns the selected indices inside the OMNI data file as dataframe. Check the List_OMNI_Indices function for more information
    # Inputs: OMNI_path ---> Location of OMNI data
    #       : indices ---> List of indices found in the OMNI data
    #       : indexMatrix ---> Bitstring to select the indices. 1 for selected indices and 0 for not selected indices
    #       : dates ---> List of datetime objects representing the dates to scrape data for
    # Outputs: OMNI_data ---> Dataframe of selected indices
    #===============================================================================

    import pandas as pd
    from datetime import datetime

    # Select indices based on the indexMatrix
    desiredIndices = [index for i, index in enumerate(indices) if indexMatrix[i] == 1]
    if verbose: print("The selected Indices are: ", desiredIndices)
    
    # Convert dates to a set of strings for faster lookup
    date_set = {date.strftime('%Y %m %d %H') for date in dates}
    
    # Initialize data dictionary for selected indices
    data_dict = {index: [] for index in desiredIndices}

    # Read the selected index to a dataframe
    try:
        with open(OMNI_path, "r") as file:
            for line in file:
                values = line.strip().split()
                if len(values) >= len(indices) and values[0].isnumeric():
                    # Convert the date from the file format to a string
                    date_str = datetime.strptime(f"{values[0]} {values[1]} {values[2]}", '%Y %j %H').strftime('%Y %m %d %H')
                    
                    # Only process if the date matches one from the list
                    if date_str in date_set:
                        for j, index in enumerate(indices):
                            if index in desiredIndices:
                                data_dict[index].append(float(values[j]))

        # Create a DataFrame from the data dictionary
        OMNI_df = pd.DataFrame(data_dict)
    except FileNotFoundError:
        print(f"File '{OMNI_path}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
        return

    OMNI_df = OMNI_impute_advanced(OMNI_df)
    OMNI_df = group_OMNI_data(OMNI_df)
    return OMNI_df
    #===============================================================================
#############################################################

# Function to read the OMNI data
############################################################################################################
def readOMNIseq(DATES,d,seq_len,pred_horz, OMNIdata, lat = LAT, long = LONG):
    #############################################################
    # Function Explanation
    #===============================================================================
    # For Temporal Shift reading more than one OMNI and transforming them to 2d array
    # DATES = List of dates
    # d = datetime(year,month,day,hour,minute)
    # seq_len = Length of the sequence
    # Inputs: OMNIdata = Scalar values of OMNI indices
    #         selectedOMNI = Selected OMNI indices
    #===============================================================================
    if seq_len == 1:
        omni = OMNIdata[DATES.index(d),:]
        omni = omni.reshape(omni.shape[0],1,1)
        return np.full((omni.shape[0], *(lat,long)), omni)
    else:
        idx = DATES.index(d)
        omni = OMNIdata[(idx - (seq_len + pred_horz-1)) : idx-(pred_horz-1), :]
        try:
            omni = omni.reshape(omni.shape[1],seq_len,1,1)
        except:
            print(d)
            omni =OMNIdata[(idx - (seq_len + pred_horz-1)) : idx-(pred_horz-1), :]
        try:
            omni2d = np.full((omni.shape[0],seq_len, *(lat,long)), omni)
        except:
            print(d)
            omni2d = np.full((omni.shape[0],seq_len, *(lat,long)), omni)
            # sanitycheck = DATES[(DATES.index(d) - (seq_len-1)) : (DATES.index(d) + 1)]
    #===============================================================================
    return omni2d
#############################################################################################################


# Reverse heliocentric transformation
###############################################################################################################
def reverseHeliocentricSingle(tec, date):
    n = tec.shape[1]
    mapNumber = int(date.hour) // 2 + 1
    shift_value = (n * mapNumber / 12 + n / 2)
    return np.roll(tec, -int(shift_value), axis=1)
###############################################################################################################

#################################################################################################################
def reverseHeliocentric(tecData, date_list):
    # Function to reverse the heliocentric transformation for a list of dates
    # tecData: 3D array of TEC data
    # date_list: list of datetime objects corresponding to the TEC data
    # Returns: 3D array of TEC data with the heliocentric transformation reversed

    # Check if tecData is a numpy array
    if not isinstance(tecData, np.ndarray):
        raise ValueError("tecData must be a numpy array")

    # Check if date_list is a list of datetime objects
    if not all(isinstance(date, datetime) for date in date_list):
        raise ValueError("date_list must be a list of datetime objects")
    # Reverse the heliocentric transformation for each date in the list
    tecData = np.array([reverseHeliocentricSingle(tecData[idx], date_list[idx]) for idx in range(len(date_list))])
    return tecData
#################################################################################################################


#############################################################################################################
def SI_categorize(allDATES, desiredDates, OMNI_path, verbose=False):
    # Optimize this: Current issue without adding all dates misclassification can occur.
    import numpy as np
    # Load OMNI
    indices = List_OMNI_Indices(OMNI_path,verbose)
    Unwanted = []   # Pick the ones you don't want with the same name as the output of List_OMNI_Indices. If not empty array. 
    indexMatrix = [0 if col in Unwanted else 1 for col in indices]  # Forming a binary matrix for the desired indices.
    OMNI_df = final_OMNI_datav2(OMNI_path, indices, indexMatrix, allDATES,verbose)
    OMNI_df['Dates'] = allDATES
    # Solar Intensity Categories
    f107_categories = {
        "Very Weak": (0, 70),
        "Weak": (70, 100),
        "Moderate": (100, 150),
        "Intense": (150, np.inf)
    }

    # Map dates to F10.7
    date_to_f107 = dict(zip(OMNI_df['Dates'], OMNI_df['F10.7']))
    
    # Function to classify dates based on F10.7
    def classify_dates(dates):
        categorized = {cat: [] for cat in f107_categories}
        for date in dates:
            if date in date_to_f107:
                f107_value = date_to_f107[date]
                for cat, (low, high) in f107_categories.items():
                    if low < f107_value <= high:
                        categorized[cat].append(date)
                        break
        return {cat: categorized[cat] for cat in f107_categories}
    
    return classify_dates(desiredDates)
#############################################################################################################


###############################################################################################################
def download_dataset(dataset_name, base_path=None):

    assert dataset_name in ["stratified", "chronological"], "Choose either 'stratified' or 'chronological'"
    if base_path is None:
        base_path = Path().resolve().parents[0]
    data_path = base_path / "datasets"
    data_path.mkdir(parents=True, exist_ok=True)

    repo_id = "Mertjhan/IonoBenchv1"
    dataset_file = f"IonoBench_{dataset_name}Split.pickle"
    omni_file = "OMNI_data_1996to2024.txt"

    for fname in [dataset_file, omni_file]:
        downloaded = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
        dest = data_path / fname
        if not dest.exists():
            shutil.copy(downloaded, dest)
            os.remove(downloaded)
            print(f"{fname} downloaded to {dest}")
        else:
            print(f"{fname} already exists. Skipping.")
###############################################################################################################

##############################################################################################################
def download_model_folder(model_name, base_path=None):
    repo_id = "Mertjhan/IonoBenchv1"
    subfolder = f"training_sessions/{model_name}"
    if base_path is None:
        base_path = Path().resolve().parents[0]
    target_path = base_path / "training_sessions" / model_name
    target_path.mkdir(parents=True, exist_ok=True)

    # Download the manifest file
    filelist_path = hf_hub_download(repo_id=repo_id, filename="files.json", subfolder=subfolder, repo_type="model")
    with open(filelist_path, "r") as f:
        files = json.load(f)

    # Download all listed files
    for fname in files:
        downloaded = hf_hub_download(repo_id=repo_id, filename=fname, subfolder=subfolder, repo_type="model")
        dest = target_path / fname
        if not dest.exists():
            shutil.copy(downloaded, dest)
            os.remove(downloaded)
        else:
            print(f"{fname} already exists. Skipping.")
    print(f"Please check the {target_path} folder for the downloaded files.")
##############################################################################################################

###############################################################################################################
def getOMNIfeature(dataDict, feature_name):
    dataDict['OMNI_index'] = {name: i for i, name in enumerate(dataDict['OMNI_names'])}
    idx = dataDict['OMNI_index'][feature_name]
    return dataDict['normOMNI'][:, idx]
###############################################################################################################