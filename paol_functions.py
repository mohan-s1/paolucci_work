#/* --------------------------------------------------------------------------------
#   Paolucci Group
#   University of Virginia
#   Mohan Shankar
#
#   paolucci_functions.py
#   This file has various functions I've written in the Paolucci group
#-------------------------------------------------------------------------------- */
import numpy as np
from scipy.interpolate import CubicSpline 
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
#-------------------------------------------------------------------------------- */
def cubicspline_interp(x_orig: np.ndarray, y_orig: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """
    Args:
        x_orig (np.ndarray): original x data to interpolate over
        y_orig (np.ndarray): original y data to interpolate over
        x_new (np.ndarray): new range of x values to interpolate over

    Returns:
        np.ndarray: _description_
    """
    interp = CubicSpline(x_orig, y_orig) 
    return interp(x_new)

def nist_collector(url: str) -> np.ndarray:
    """
    Args:
        url (str): NIST Janaf table for compound of choice

    Returns:
        np.array: array with temperature (T/K), entropy (S0), enthalpy (H - H0)
    """
    
    # Send a GET request to the webpage
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table in the HTML
    table = soup.find('table')

    # Extract table headers
    headers = [header.text.strip() for header in table.find_all('th')]

    # Extract table rows
    rows= []
    for row in table.find_all('tr')[2:-4]:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        if cols:
            rows.append(cols)

    temp = [] # temperature values for compound
    enthalpy = [] # enthalpy values for compound
    entropy = [] # entropy values for compound

    for row in rows:
        if len(row) == 15: # make sure the line isn't blank
            temp.append(row[0])
            entropy.append(row[4])
            enthalpy.append(row[8])
    

    std_enthalpy_index = temp.index('298.15') # find index where the standard temperature is used

    # std_enthalpy = enthalpy[std_enthalpy_index] # find change in enthalpy associated with standard temp

    temp.remove('298.15') # remove standard temp 

    temp.append('298.15')

    enthalpy.append(enthalpy[std_enthalpy_index]) # add standard enthalpy to end of array

    enthalpy.remove(enthalpy[std_enthalpy_index]) # remove standard enthalpy value from middle of array

    entropy.append(entropy[std_enthalpy_index]) # add standard entropy to end of array

    entropy.remove(entropy[std_enthalpy_index]) # remove standard entropy value from middle of array

    temp = np.array(temp, dtype = np.float32) # convert temperature list to np.array w/ units of Kelvin

    entropy = np.array(entropy, dtype = np.float32)/1000 # units of J/K*mol --> kJ/K*mol

    enthalpy = np.array(enthalpy, dtype = np.float32) # units of kJ/mol

    return np.transpose(np.vstack((temp, entropy, enthalpy)))



def parse_frequencies(file_path):
    pattern = r'Frequencies\s+--\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)' #regex to find desires lines
    frequencies = []

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                freq_values = match.groups()
                frequencies.extend(freq_values)
    
    # turn strings to floats and cutoff at 100 cm-1
    
    no_low_freq = [float(x) for x in frequencies if float(x) >= 100]
    
    # Convert the list to a NumPy array with float values
    frequencies_array = np.array(no_low_freq, dtype=float)
    
    return frequencies_array


def vibrational_entropy(frequencies_cm1, temperatures):
    # Constants
    R = 8.314462618  # Gas constant in J/(mol*K)
    h = 6.62607015e-34  # Planck constant in Js
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    c = 2.99792458e10  # Speed of light in cm/s
    
    # Convert frequencies from cm^-1 to Hz
    frequencies_hz = frequencies_cm1 * c
    
    # Convert frequencies and temperatures to numpy arrays if they are not already
    frequencies_hz = np.asarray(frequencies_hz)
    temperatures = np.asarray(temperatures)
    
    # Reshape frequencies for broadcasting
    frequencies_hz = frequencies_hz[:, np.newaxis]
    
    print(f'Frequencies shape:{frequencies_hz.shape}')

    # Calculate x = (h * nu) / (k_B * T) for all frequencies and temperatures
    x = (h * frequencies_hz) / (k_B * temperatures)

    # print(f'x shape:{x.shape}')
    
    # Calculate the vibrational entropy using broadcasting
    entropy = x / (np.exp(x) - 1) - np.log(1 - np.exp(-x))
    S_vib = R * np.sum(entropy, axis=0)
    
    return S_vib

def extract_rt_data(file_path):
    # Define patterns to match the lines with rotational temperatures, symmetry number, and molecular mass
    temp_pattern = r"Rotational temperatures \(Kelvin\)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)"
    symmetry_pattern = r"Rotational symmetry number\s+(\d+)"
    mass_pattern = r"Molecular mass:\s+(\d+\.\d+)\s+amu\."
    
    # Initialize variables to store the extracted data
    rotational_temperatures = []
    symmetry_number = None
    molecular_mass = None
    
    # Open the file and read it line-by-line
    with open(file_path, 'r') as file:
        for line in file:
            # Search for the temperature pattern in the current line
            temp_match = re.search(temp_pattern, line)
            if temp_match:
                # Extract the temperatures from the matched line and convert them to floats
                temperatures = [float(temp_match.group(1)), float(temp_match.group(2)), float(temp_match.group(3))]
                rotational_temperatures.extend(temperatures)
            
            # Search for the symmetry pattern in the current line
            symmetry_match = re.search(symmetry_pattern, line)
            if symmetry_match:
                # Extract the symmetry number from the matched line and convert it to an integer
                symmetry_number = int(symmetry_match.group(1))
            
            # Search for the molecular mass pattern in the current line
            mass_match = re.search(mass_pattern, line)
            if mass_match:
                # Extract the molecular mass from the matched line and convert it to a float
                molecular_mass = float(mass_match.group(1))
    
    return symmetry_number, rotational_temperatures, molecular_mass

def rot_partition(temp: float, sym:float,  theta_one:float, theta_two: float, theta_three:float):
    pi = np.pi 
    return np.array(np.sqrt(pi)/sym * (temp**(3/2) / np.sqrt(theta_one * theta_two * theta_three)))

def trans_partition(mass:float, temp:np.ndarray, pres:float) -> np.ndarray:
    """
    _summary_

    Args:
        mass (float): mass in amu
        temp (np.ndarray): temperature in kelvin
        pres (float): pressure in atm

    Returns:
        np.ndarray: translational partition function at specified temperature(s)
    """
    # Constants
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    h = 6.62607015e-34  # Planck constant in Js
    N_A = 6.02214076e23  # Avogadro's number in mol^-1
    R = 8.314462618  # Gas constant in J/(mol*K)
    
    # Convert mass from amu to kg
    mass_kg = mass * 1.66053906660e-27  # 1 amu = 1.66053906660e-27 kg
    
    # Convert pressure from atm to Pa
    pressure_pa = pres * 101325  # 1 atm = 101325 Pa
    
    # Calculate volume using ideal gas law: V = (N_A * k_B * T) / P
    volume = (N_A * k_B * temp) / pressure_pa
    
    # Calculate translation partition function
    q_trans = ((2 * np.pi * mass_kg * k_B * temp) / (h**2))**(3/2) * (volume / N_A)
    
    return np.array(q_trans)


def vibrational_entropy(frequencies_cm1, temperatures):
    
    import numpy as np
    
    # Constants
    R = 8.314462618  # Gas constant in J/(mol*K)
    h = 6.62607015e-34  # Planck constant in Js
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    c = 2.99792458e10  # Speed of light in cm/s
    
    # Convert frequencies from cm^-1 to Hz
    frequencies_hz = frequencies_cm1 * c
    
    # Convert frequencies and temperatures to numpy arrays if they are not already
    frequencies_hz = np.asarray(frequencies_hz)
    temperatures = np.asarray(temperatures)
    
    # Reshape frequencies for broadcasting
    frequencies_hz = frequencies_hz[:, np.newaxis]
    
    # print(f'Frequencies shape:{frequencies_hz.shape}')

    # Calculate x = (h * nu) / (k_B * T) for all frequencies and temperatures
    x = (h * frequencies_hz) / (k_B * temperatures)

    # print(f'x shape:{x.shape}')
    
    # Calculate the vibrational entropy using broadcasting
    entropy = x / (np.exp(x) - 1) - np.log(1 - np.exp(-x))
    S_vib = R * np.sum(entropy, axis=0)
    
    return np.array(S_vib)

def calc_entropy(infile:str, temperature:np.ndarray) -> np.ndarray:
    """
    _summary_

    Args:
        infile (str): Path to Gaussian log file
        temperature (np.ndarray): Array (can be 1D) of temperature values to calculate entropy 

    Returns:
        np.ndarray: Return s_trans, s_vib, s_rot over specified temperature in units of J/mol*K
    """

    R = 8.314462618  # Gas constant in J/(mol*K)
    h = 6.62607015e-34  # Planck constant in Js
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    c = 2.99792458e10  # Speed of light in cm/s
    

    symmetry_number, rotational_temperatures, molecular_mass = extract_rt_data(infile)
    found_frequencies = parse_frequencies(infile)
    
    q_trans = trans_partition(temp = temperature, mass = molecular_mass, pres = 1)
    q_rot = rot_partition(temp = temperature, sym = symmetry_number, theta_one = rotational_temperatures[0], theta_two = rotational_temperatures[1], theta_three = rotational_temperatures[2])

    s_trans = R * (np.log(q_trans) + 5/2)
    s_rot = R * (np.log(q_rot) + 3/2)
    s_vib = vibrational_entropy(frequencies_cm1 = found_frequencies, temperatures = temperature)

    return s_trans, s_vib, s_rot

def filter_dataframe(data: pd.DataFrame, metal:str, charge: int, geom: str) -> pd.DataFrame:
    """
    Returns dataframe with grouping specified by arguments
    """
    if geom != None:
        return data[(data['Metal'] == metal) & (data['Charge'] == str(charge)) & (data['Geometry'] == geom)]
    else:
        return data[(data['Metal'] == metal) & (data['Charge'] == str(charge))]