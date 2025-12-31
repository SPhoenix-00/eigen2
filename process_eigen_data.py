import pandas as pd
import numpy as np
import time
import sys
import os
import argparse
from datetime import datetime
import re

# --- Configuration ---
INPUT_FILE = 'Eigen2_Master(GFIN)_05_skinny - MASTER.csv'
OUTPUT_FILE_PKL = 'Eigen2_Master_PY_OUTPUT_151025.pkl'
OUTPUT_FILE_CSV = 'Eigen2_Master_PY_OUTPUT_151025_FOR_COMPARE.csv'

# --- Index Map for Slicing (Step 2: ManipulateArrayString) ---
# Your VBA keeps indices: 0, 1, 2, 3, 6, 9, 10, 13, 15
# Original:
# [0] Open, [1] Close, [2] High, [3] Low
# [4] RSI_U_EMA7 (DROP)
# [5] RSI_D_EMA7 (DROP)
# [6] RSI (KEEP)
# [7] EMA12 (DROP)
# [8] EMA26 (DROP)
# [9] MACD (KEEP)
# [10] MACD_Signal (KEEP)
# [11] EMA12EMA (DROP)
# [12] EMA12EMAEMA (DROP)
# [13] Trix (KEEP)
# [14] x20DMA (DROP)
# [15] xDiffDMA (KEEP)
INDICES_TO_KEEP = [0, 1, 2, 3, 6, 9, 10, 13, 15]

# --- Low-Level "Getter" Helpers (Replaces VBA string parsing) ---
def get_close(day_data):
    """Gets the 'Close' price (index 1) from the data list."""
    if day_data is None or len(day_data) < 2:
        return np.nan
    return day_data[1]

# --- Core Calculation Helpers (Replicates VBA Functions) ---

def rsi_u(current_day, prev_day):
    """Calculates RSI_U. Replicates VBA Function."""
    curr_close = get_close(current_day)
    prev_close = get_close(prev_day)
    if np.isnan(curr_close) or np.isnan(prev_close):
        return 0.0
    return max(0, curr_close - prev_close)

def rsi_d(current_day, prev_day):
    """Calculates RSI_D. Replicates VBA Function."""
    curr_close = get_close(current_day)
    prev_close = get_close(prev_day)
    if np.isnan(curr_close) or np.isnan(prev_close):
        return 0.0
    return max(0, prev_close - curr_close)

def rsi_u_ema7(current_day_data, history):
    """Appends RSI_U_EMA7. history is 7 days before current."""
    full_history = history + [current_day_data] # 8 days total
    rsi_u_0 = rsi_u(full_history[-1], full_history[-2])
    prev_day_data = full_history[-2] # This is history[-1]
    
    # VBA: If CountChrInString(prevDay, ",") < 4 Then
    if len(prev_day_data) < 5: 
        # Calculate 7-day SMA
        rsi_u_vals = [rsi_u(full_history[j], full_history[j-1]) for j in range(1, 8)]
        v_calc = np.mean(rsi_u_vals)
    else:
        # Calculate EMA
        rsi_u_ema_prev = prev_day_data[4] # Get previous RSI_U_EMA7
        v_calc = rsi_u_ema_prev + (2 / 8) * (rsi_u_0 - rsi_u_ema_prev)
    
    return current_day_data + [v_calc]

def rsi_d_ema7(current_day_data, history):
    """Appends RSI_D_EMA7. history is 7 days before current."""
    full_history = history + [current_day_data] # 8 days total
    rsi_d_0 = rsi_d(full_history[-1], full_history[-2])
    prev_day_data = full_history[-2]
    
    # VBA: If CountChrInString(prevDay, ",") < 5 Then
    if len(prev_day_data) < 6: # Note: VBA index is < 5, so we check < 6
        # Calculate 7-day SMA
        rsi_d_vals = [rsi_d(full_history[j], full_history[j-1]) for j in range(1, 8)]
        v_calc = np.mean(rsi_d_vals)
    else:
        # Calculate EMA
        rsi_d_ema_prev = prev_day_data[5] # Get previous RSI_D_EMA7
        v_calc = rsi_d_ema_prev + (2 / 8) * (rsi_d_0 - rsi_d_ema_prev)
    
    return current_day_data + [v_calc]

def run_rsi(current_day_data, history):
    """Appends RSI, calling U_EMA7 and D_EMA7 first."""
    # VBA: If commaTest < 5 Then
    if len(current_day_data) < 5:
        current_day_data = rsi_u_ema7(current_day_data, history)
        current_day_data = rsi_d_ema7(current_day_data, history)
    
    curr_rsi_u_ema7 = current_day_data[4]
    curr_rsi_d_ema7 = current_day_data[5]
    
    if (curr_rsi_u_ema7 + curr_rsi_d_ema7) == 0:
        v_calc = 0
    else:
        v_calc = 100 * curr_rsi_u_ema7 / (curr_rsi_u_ema7 + curr_rsi_d_ema7)
    
    return current_day_data + [v_calc]

def run_ema12(current_day_data, history):
    """Appends EMA12. history is 11 days before current."""
    full_history = history + [current_day_data] # 12 days total
    prev_day_data = full_history[-2]
    
    # VBA: If commaTest < 7 Then
    if len(prev_day_data) < 8:
        # Calculate 12-day SMA of Close
        v_calc = np.mean([get_close(day) for day in full_history])
    else:
        # Calculate EMA
        curr_close = get_close(current_day_data)
        ema_prev = prev_day_data[7] # Get previous EMA12
        v_calc = (curr_close - ema_prev) * (2 / 13) + ema_prev
        
    return current_day_data + [v_calc]

def run_ema26_init(current_day_data, history):
    """Calculates initial SMA for EMA26. history is 25 days before."""
    full_history = history + [current_day_data] # 26 days total
    # Calculate 26-day SMA of Close
    v_calc = np.mean([get_close(day) for day in full_history])
    return current_day_data + [v_calc]

def run_ema26(current_day_data, prev_day_data):
    """Appends EMA26. Assumes EMA26_init was already run."""
    curr_close = get_close(current_day_data)
    ema_prev = prev_day_data[8] # Get previous EMA26
    v_calc = (curr_close - ema_prev) * (2 / 27) + ema_prev
    return current_day_data + [v_calc]

def run_macd(current_day_data):
    """Appends MACD."""
    curr_ema12 = current_day_data[7]
    curr_ema26 = current_day_data[8]
    v_calc = curr_ema12 - curr_ema26
    return current_day_data + [v_calc]

def run_macd_signal(current_day_data, history):
    """Appends MACD_Signal. history is 8 days before current."""
    full_history = history + [current_day_data] # 9 days total
    prev_day_data = full_history[-2]
    
    # VBA: If commaTest < 10 Then
    if len(prev_day_data) < 11:
        # Calculate 9-day SMA of MACD
        v_calc = np.mean([day[9] for day in full_history]) # Get MACD (index 9)
    else:
        # Calculate EMA
        curr_macd = current_day_data[9]
        signal_prev = prev_day_data[10] # Get previous MACD_Signal
        v_calc = (curr_macd - signal_prev) * (2 / 10) + signal_prev
    
    return current_day_data + [v_calc]

def run_ema12ema(current_day_data, history):
    """Appends EMA(EMA12). history is 11 days before current."""
    full_history = history + [current_day_data] # 12 days total
    prev_day_data = full_history[-2]

    # VBA: If commaTest < 11 Then
    if len(prev_day_data) < 12:
        # Calculate 12-day SMA of EMA12
        v_calc = np.mean([day[7] for day in full_history]) # Get EMA12 (index 7)
    else:
        # Calculate EMA
        curr_ema12 = current_day_data[7]
        ema12ema_prev = prev_day_data[11] # Get previous EMA12EMA
        v_calc = (curr_ema12 - ema12ema_prev) * (2 / 13) + ema12ema_prev
        
    return current_day_data + [v_calc]

def run_ema12emaema(current_day_data, history):
    """Appends EMA(EMA(EMA12)). history is 11 days before current."""
    full_history = history + [current_day_data] # 12 days total
    prev_day_data = full_history[-2]
    
    # VBA: If commaTest < 12 Then
    if len(prev_day_data) < 13:
        # Calculate 12-day SMA of EMA12EMA
        v_calc = np.mean([day[11] for day in full_history]) # Get EMA12EMA (index 11)
    else:
        # Calculate EMA
        curr_ema12ema = current_day_data[11]
        ema12emaema_prev = prev_day_data[12] # Get previous EMA12EMAEMA
        v_calc = (curr_ema12ema - ema12emaema_prev) * (2 / 13) + ema12emaema_prev
        
    return current_day_data + [v_calc]

def run_trix(current_day_data, prev_day_data):
    """Appends Trix."""
    ema_curr = current_day_data[12] # EMA12EMAEMA
    ema_prev = prev_day_data[12] # EMA12EMAEMA
    
    v_calc = 100 * (ema_curr / ema_prev - 1) if ema_prev != 0 else 0
    return current_day_data + [v_calc]

def run_x20dma(current_day_data, history):
    """Appends x20DMA. history is 19 days before current."""
    full_history = history + [current_day_data] # 20 days total
    # Calculate 20-day SMA of Close
    v_calc = np.mean([get_close(day) for day in full_history])
    return current_day_data + [v_calc]

def run_xdiffdma(current_day_data):
    """Appends xDiffDMA."""
    curr_close = get_close(current_day_data)
    dma_curr = current_day_data[14] # x20DMA
    v_calc = curr_close - dma_curr
    return current_day_data + [v_calc]


# --- Utility for Progress Bar ---
def print_progress(iteration, total, start_time, bar_length=50):
    """Displays a command-line progress bar with ETA."""
    progress = iteration / total
    elapsed = time.time() - start_time
    
    if progress > 0:
        eta = (elapsed / progress) * (1 - progress)
    else:
        eta = 0
    
    arrow = '=' * int(round(progress * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write(f'\rProgress: [{arrow + spaces}] {int(progress * 100)}% '
                     f'(Col {iteration}/{total}) | '
                     f'Elapsed: {time.strftime("%H:%M:%S", time.gmtime(elapsed))} | '
                     f'ETA: {time.strftime("%H:%M:%S", time.gmtime(eta))}  ')
    sys.stdout.flush()

# --- Main Processing Function (Replicates DataRun Sub) ---
def process_dataframe(df):
    """
    Applies the full stack of VBA calculations to the DataFrame.
    The DataFrame is expected to hold lists, not single numbers.
    """
    num_cols = len(df.columns)
    num_rows = len(df)
    start_time = time.time()
    
    # VBA loops column by column ("B1:CW1")
    for col_idx, col_name in enumerate(df.columns):
        print_progress(col_idx + 1, num_cols, start_time)
        
        # Get a copy of the column to modify
        # Using .tolist() is much faster than iloc looping
        col_data = df[col_name].tolist()

        # --- 1. Running RSI routine for the stack ---
        # VBA: For i = 8 To 3922
        for i in range(7, num_rows): # 7 is the 8th row (0-indexed)
            if col_data[i] is None: continue
            history = col_data[i-7:i]
            col_data[i] = run_rsi(col_data[i], history)
        # VBA: Array cleanup post-routine (i = 1 To 7)
        for i in range(7):
            if col_data[i] is not None: col_data[i] += [None, None, None]

        # --- 2. Running EMA12 routine for the stack ---
        # VBA: For i = 12 To 3922
        for i in range(11, num_rows): # 11 is the 12th row
            if col_data[i] is None: continue
            history = col_data[i-11:i]
            col_data[i] = run_ema12(col_data[i], history)
        # VBA: Array cleanup (i = 1 To 11)
        for i in range(11):
            if col_data[i] is not None: col_data[i] += [None]

        # --- 3. Running EMA26 routine for the stack ---
        # VBA: col.Offset(26, 0).Value = EMA26_init(...)
        if 25 < num_rows and col_data[25] is not None:
             history = col_data[0:25] # 25 rows (0-24)
             col_data[25] = run_ema26_init(col_data[25], history)
        # VBA: For i = 27 To 3922
        for i in range(26, num_rows):
            if col_data[i] is None: continue
            col_data[i] = run_ema26(col_data[i], col_data[i-1])
        # VBA: Array cleanup (i = 1 To 25)
        for i in range(25):
            if col_data[i] is not None: col_data[i] += [None]

        # --- 4. Running MACD routine for the stack ---
        # VBA: For i = 26 To 3922
        for i in range(25, num_rows):
            if col_data[i] is None: continue
            col_data[i] = run_macd(col_data[i])
        # VBA: Array cleanup (i = 1 To 25)
        for i in range(25):
            if col_data[i] is not None: col_data[i] += [None]

        # --- 5. Running MACD_Signal routine for the stack ---
        # VBA: For i = 34 To 3922
        for i in range(33, num_rows): # 33 is the 34th row
            if col_data[i] is None: continue
            history = col_data[i-8:i] # Needs 8 previous rows
            col_data[i] = run_macd_signal(col_data[i], history)
        # VBA: Array cleanup (i = 1 To 33)
        for i in range(33):
            if col_data[i] is not None: col_data[i] += [None]

        # --- 6. Running EMA12EMA routine for the stack ---
        # VBA: For i = 23 To 3922
        for i in range(22, num_rows):
            if col_data[i] is None: continue
            history = col_data[i-11:i]
            col_data[i] = run_ema12ema(col_data[i], history)
        # VBA: Array cleanup (i = 1 To 22)
        for i in range(22):
            if col_data[i] is not None: col_data[i] += [None]

        # --- 7. Running EMA12EMAEMA routine for the stack ---
        # VBA: For i = 34 To 3922
        for i in range(33, num_rows):
            if col_data[i] is None: continue
            history = col_data[i-11:i]
            col_data[i] = run_ema12emaema(col_data[i], history)
        # VBA: Array cleanup (i = 1 To 33)
        for i in range(33):
            if col_data[i] is not None: col_data[i] += [None]

        # --- 8. Running Trix routine for the stack ---
        # VBA: For i = 35 To 3922
        for i in range(34, num_rows):
            if col_data[i] is None: continue
            col_data[i] = run_trix(col_data[i], col_data[i-1])
        # VBA: Array cleanup (i = 1 To 34)
        for i in range(34):
            if col_data[i] is not None: col_data[i] += [None]

        # --- 9. Running x20DMA routine for the stack ---
        # VBA: For i = 20 To 3922
        for i in range(19, num_rows):
            if col_data[i] is None: continue
            history = col_data[i-19:i]
            col_data[i] = run_x20dma(col_data[i], history)
        # VBA: Array cleanup (i = 1 To 19)
        for i in range(19):
            if col_data[i] is not None: col_data[i] += [None]

        # --- 10. Running xDiffDMA routine for the stack ---
        # VBA: For i = 20 To 3922
        for i in range(19, num_rows):
            if col_data[i] is None: continue
            col_data[i] = run_xdiffdma(col_data[i])
        # VBA: Array cleanup (i = 1 To 19)
        for i in range(19):
            if col_data[i] is not None: col_data[i] += [None]
            
        # --- End of column loop: Assign processed list back to DataFrame ---
        df[col_name] = col_data

    sys.stdout.write('\n') # Move to next line after progress bar
    return df

# --- NEW STEP 0: Shift data UP (top-align) before processing ---
def shift_data_up(df):
    """
    Shifts all column data to the top of the DataFrame.
    This ensures processing starts from row 0 for each column,
    avoiding None values in the history lookback.
    Returns the shifted DataFrame and a dict of original offsets per column.
    """
    print("Shifting data to top-align for processing...")

    num_rows = df.shape[0]
    offsets = {}  # Store how many empty rows were at the top of each column
    result_data = {}  # Build column data as lists

    for col_name in df.columns:
        col_data = df[col_name].tolist()

        # Find first non-None value
        first_valid_idx = None
        for i, val in enumerate(col_data):
            if val is not None:
                first_valid_idx = i
                break

        if first_valid_idx is None:
            # Column is all None
            offsets[col_name] = num_rows
            result_data[col_name] = [None] * num_rows
            continue

        offsets[col_name] = first_valid_idx

        # Extract valid data and pad with None at the end
        valid_data = col_data[first_valid_idx:]
        padding = [None] * first_valid_idx
        result_data[col_name] = valid_data + padding

    # Create DataFrame from dict (preserves list objects in cells)
    new_df = pd.DataFrame(result_data, index=df.index)

    return new_df, offsets


# --- NEW STEP 1: Replicates ShiftDataDown Sub ---
def shift_data_down(df, offsets=None):
    """
    Replicates the VBA ShiftDataDown logic.
    Aligns all data to the bottom of the DataFrame, column by column.
    If offsets dict is provided, uses those to restore original positions (RECOMMENDED).
    """
    print("Shifting data to restore alignment (ShiftDataDown)...")
    
    # Create a new, empty DataFrame to hold the shifted data
    new_df = pd.DataFrame(None, index=df.index, columns=df.columns)
    
    # If offsets are provided, use them to restore original alignment EXACTLY
    if offsets is not None:
        print("  Using captured offsets to restore exact data alignment...")
        for col_name in df.columns:
            if col_name not in offsets:
                continue
                
            offset = offsets[col_name]
            col_data = df[col_name].tolist()
            
            # The data currently starts at index 0 (shifted up)
            # We want to move it down by 'offset'
            
            # 1. Take the data that was shifted up (excluding the padding we added at the end)
            #    The valid data length is (total_rows - offset)
            num_rows = len(col_data)
            valid_len = num_rows - offset
            
            # 2. Extract valid data from top
            valid_data = col_data[:valid_len]
            
            # 3. Create new column: [None]*offset + valid_data
            #    This pushes the data back to its original start index
            new_col_data = [None] * offset + valid_data
            
            # Assign to DataFrame
            new_df[col_name] = new_col_data
            
        return new_df

    # --- FALLBACK: OLD BEHAVIOR (Bottom Align) ---
    # Only used if offsets are not provided
    print("  WARNING: No offsets provided. Forcing bottom alignment (Risk of time-shift).")
    
    # --- FIX 1: Start from the beginning (index 0) ---
    pandas_start_index = 0
    num_rows = df.shape[0]
    
    # Length of the data section
    data_section_len = num_rows - pandas_start_index
    
    for col_idx, col_name in enumerate(df.columns):
        # Get all valid data from the very start
        col_data = df.loc[df.index[pandas_start_index]:, col_name].dropna()
        
        if not col_data.empty:
            values = col_data.values
            num_values = len(values)
            
            # Calculate offset to "bottom-align" the data
            offset = data_section_len - num_values
            
            paste_start_index = pandas_start_index + offset
            paste_end_index = paste_start_index + num_values
            
            # Assign the list of values to the correct slice in the new DataFrame
            new_df.iloc[paste_start_index:paste_end_index, col_idx] = values

    return new_df

# --- Cell Formatting Function (used in both normal and append modes) ---
def format_cell(x):
    """
    Converts a cell value to a list format.
    x is a string like "[43.77,43.23,43.8,43.2]" or a plain number like "1.4422"
    """
    # Already parsed list (e.g., when working with in-memory data)
    if isinstance(x, list):
        return x

    # Handle numeric input (pandas often reads CSV numeric cells as float/int, not str)
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        val = float(x)
        return [val, val, val, val]

    # Check for empty/invalid strings
    if not isinstance(x, str) or x.strip() == "" or x.strip() == "[]":
        return None

    try:
        x = x.strip()
        if x.startswith('['):
            # It's a bracketed array: "[1,2,3,4]" -> [1.0, 2.0, 3.0, 4.0]
            list_string = x[1:-1]
            return [float(part) for part in list_string.split(',')]
        else:
            # It's a plain number: "1.4422" -> [1.4422, 1.4422, 1.4422, 1.4422]
            val = float(x)
            return [val, val, val, val]

    except (ValueError, TypeError):
        # Catches errors if a part isn't a valid number
        return None

# --- NEW STEP 2: Replicates ManipulateArrayString Function ---
def manipulate_list(cell_list):
    """
    Takes a single list and returns a new list
    containing only the elements at INDICES_TO_KEEP.
    """
    # Check if the input is a valid list with enough elements
    if not isinstance(cell_list, list) or len(cell_list) < 16:
        # Replicates VBA's "[]" or "Error" state
        return None 
    
    try:
        # This is the high-speed Python equivalent of your VBA function
        return [cell_list[i] for i in INDICES_TO_KEEP]
    except (IndexError, TypeError):
        # Handle case where an element might be missing (e.g., None)
        return None

# --- Helper function to update configuration files ---
def update_config_files(new_filename):
    """
    Update configuration files to point to the new production pickle file.
    Updates:
    - process_eigen_data.py: OUTPUT_FILE_PKL
    - QUICK_START.md: References to the pickle filename
    """
    # Extract just the filename (not full path)
    new_basename = os.path.basename(new_filename)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    updated_files = []
    
    # Update process_eigen_data.py (the current script)
    script_path = os.path.abspath(__file__)
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update OUTPUT_FILE_PKL
        pattern = r"OUTPUT_FILE_PKL = '[^']+'"
        replacement = f"OUTPUT_FILE_PKL = '{new_basename}'"
        new_content = re.sub(pattern, replacement, content)
        
        if new_content != content:
            # Write to a temporary file first, then rename (safer)
            temp_path = script_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            os.replace(temp_path, script_path)
            updated_files.append('process_eigen_data.py')
    except Exception as e:
        print(f"  Warning: Could not update process_eigen_data.py: {e}")
    
    # Update QUICK_START.md
    quick_start_path = os.path.join(base_dir, 'QUICK_START.md')
    if os.path.exists(quick_start_path):
        try:
            with open(quick_start_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update references to Eigen2_Master_PY_OUTPUT_*.pkl
            # Pattern matches: Eigen2_Master_PY_OUTPUT_151025.pkl or Eigen2_Master_PY_OUTPUT.pkl
            old_pattern = r"Eigen2_Master_PY_OUTPUT(?:_\d{6})?\.pkl"
            new_content = re.sub(old_pattern, new_basename, content)
            
            if new_content != content:
                with open(quick_start_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                updated_files.append('QUICK_START.md')
        except Exception as e:
            print(f"  Warning: Could not update QUICK_START.md: {e}")
    
    if updated_files:
        print(f"  Updated: {', '.join(updated_files)}")
    else:
        print(f"  No files needed updating (or update failed)")

# --- Helper function for date suffix generation ---
def generate_output_filename(base_filename):
    """
    Generate output filename with date suffix in format _ddmmyy.{ext}
    Removes existing date suffix if present.
    Supports both .pkl and .csv files.
    """
    # Extract extension
    if base_filename.endswith('.pkl'):
        ext = '.pkl'
        pattern = r'_(\d{6})\.pkl$'
    elif base_filename.endswith('.csv'):
        ext = '.csv'
        pattern = r'_(\d{6})_FOR_COMPARE\.csv$'
    else:
        # Default to .pkl if no extension found
        ext = '.pkl'
        pattern = r'_(\d{6})\.pkl$'
        if not base_filename.endswith('.pkl'):
            base_filename = base_filename + '.pkl'
    
    # Remove existing date suffix pattern if present
    base_name = re.sub(pattern, '', base_filename)
    # Also handle case where extension was added
    if base_name.endswith(ext):
        base_name = base_name[:-len(ext)]
    
    # Add new date suffix
    today = datetime.now()
    date_suffix = today.strftime('%d%m%y')
    
    # Handle special case for CSV (add _FOR_COMPARE)
    if ext == '.csv' and '_FOR_COMPARE' not in base_name:
        output_filename = f"{base_name}_{date_suffix}_FOR_COMPARE.csv"
    else:
        output_filename = f"{base_name}_{date_suffix}{ext}"
    
    return output_filename

# --- Append Mode Function ---
def append_mode(production_file, new_data_file, *, update_config=False, allow_overlap=False):
    """
    Append mode: Process new raw data and append to existing production dataset.
    Uses last 100 rows from production as history for proper indicator calculation.
    """
    HISTORY_ROWS = 100  # Hardcoded history rows
    VBA_CALC_RAMP_UP_ROWS = 34  # Ramp-up rows to skip
    
    print("="*60)
    print("APPEND MODE")
    print("="*60)
    
    # --- Step 1: Load Production Dataset ---
    print(f"\nStep 1: Loading production dataset from {production_file}...")
    if not os.path.exists(production_file):
        print(f"Error: Production file not found: {production_file}")
        return
    
    df_production = pd.read_pickle(production_file)
    print(f"Loaded production dataset: {df_production.shape[0]} rows × {df_production.shape[1]} columns")
    
    if len(df_production) < HISTORY_ROWS:
        print(f"Error: Production dataset has only {len(df_production)} rows, but {HISTORY_ROWS} rows are required for history.")
        return
    
    # Extract last 100 rows for history
    df_history = df_production.iloc[-HISTORY_ROWS:].copy()
    print(f"Extracted last {HISTORY_ROWS} rows for history")
    
    # Store full production for later appending
    df_production_full = df_production.copy()
    
    # --- Step 2: Load and Validate New Data ---
    print(f"\nStep 2: Loading new data from {new_data_file}...")
    if not os.path.exists(new_data_file):
        print(f"Error: New data file not found: {new_data_file}")
        return
    
    df_new_raw = pd.read_csv(new_data_file, index_col=0)
    print(f"Loaded new data: {df_new_raw.shape[0]} rows × {df_new_raw.shape[1]} columns")
    
    # Remove TRANSFER column if present
    if 'TRANSFER' in df_new_raw.columns:
        df_new_raw = df_new_raw.drop(columns=['TRANSFER'])
        print("Dropped TRANSFER column from new data")
    
    # Validate column matching
    production_cols = set(df_production.columns)
    new_data_cols = set(df_new_raw.columns)
    
    if production_cols != new_data_cols:
        missing_in_new = production_cols - new_data_cols
        extra_in_new = new_data_cols - production_cols
        
        print(f"\nError: Column mismatch detected!")
        if missing_in_new:
            print(f"  Columns in production but missing in new data: {sorted(missing_in_new)}")
        if extra_in_new:
            print(f"  Columns in new data but not in production: {sorted(extra_in_new)}")
        print(f"\nProduction columns ({len(production_cols)}): {sorted(production_cols)}")
        print(f"New data columns ({len(new_data_cols)}): {sorted(new_data_cols)}")
        return
    
    print(f"✓ Column validation passed: {len(production_cols)} columns match")

    # Align new data column order to production (prevents accidental order drift)
    df_new_raw = df_new_raw.loc[:, df_production.columns]
    
    # Clean and convert new data using format_cell function
    print("Formatting new data from single values to lists...")
    df_new_raw = df_new_raw.replace(r'^\s*$', np.nan, regex=True)
    df_new_lists = df_new_raw.map(format_cell)
    
    # --- Step 3: Date Continuity Check / Overlap Guard ---
    print(f"\nStep 3: Checking date continuity...")
    production_last_date = df_production.index[-1]
    new_data_first_date = df_new_lists.index[0]
    
    print(f"Production last date: {production_last_date}")
    print(f"New data first date: {new_data_first_date}")
    
    # Convert index to timestamps for robust comparisons
    prod_index_ts = pd.to_datetime(df_production.index, errors='coerce')
    new_index_ts = pd.to_datetime(df_new_lists.index, errors='coerce')

    if prod_index_ts.isna().any():
        print("Error: Could not parse one or more production index values as dates.")
        print("Append mode requires a date-like index.")
        return
    if new_index_ts.isna().any():
        bad = df_new_lists.index[pd.isna(new_index_ts)]
        print("Error: Could not parse one or more new-data index values as dates:")
        print(f"  {list(bad[:10])}" + (" ..." if len(bad) > 10 else ""))
        print("Append mode requires a date-like index.")
        return

    prod_last_ts = prod_index_ts.iloc[-1]
    new_first_ts = new_index_ts.iloc[0]

    if new_first_ts <= prod_last_ts:
        print(f"\n⚠️  WARNING: New data does not start strictly after production.")
        print(f"   Production ends (ts): {prod_last_ts}  | raw: {production_last_date}")
        print(f"   New starts      (ts): {new_first_ts}  | raw: {new_data_first_date}")
        if not allow_overlap:
            print("\nError: Overlap/rewind detected. For safety, append mode refuses to proceed.")
            print("If you intentionally included overlapping rows, re-run with --allow-overlap to drop them.")
            return
        print("Continuing because --allow-overlap was provided; overlapping/older rows will be dropped.")
    else:
        print(f"✓ Date continuity check passed (new data starts after production)")

    # Drop any new rows that are <= production last date to prevent duplicates/corruption
    keep_mask = new_index_ts > prod_last_ts
    dropped = int((~keep_mask).sum())
    if dropped:
        print(f"Dropping {dropped} overlapping/old rows from new data (<= production last date).")
        df_new_lists = df_new_lists.loc[keep_mask].copy()

    if df_new_lists.empty:
        print("Error: After dropping overlap/old rows, there is nothing left to append.")
        return
    
    # --- Step 4: Combine History and New Data ---
    print(f"\nStep 4: Combining history ({HISTORY_ROWS} rows) with new data ({len(df_new_lists)} rows)...")
    
    # FIX: Convert history to raw format (Indices 0-3 only: Open, Close, High, Low)
    # The processed history (Skinny) does not contain the intermediate EMA states 
    # (indices 4, 5, 7, 8, etc.) required to continue calculations directly.
    # We must treat the history as raw data and let the model "warm up" again.
    def to_raw(cell_list):
        """Convert processed cell (9 elements) back to raw OHLC (4 elements)."""
        if isinstance(cell_list, list) and len(cell_list) >= 4:
            return cell_list[:4]  # Keep only [Open, Close, High, Low]
        return None
    
    # Strip history down to raw OHLC
    print("  Converting history from processed (9 elements) to raw (4 elements)...")
    df_history_raw = df_history.map(to_raw)
    
    # Combine: history (now raw) + new raw data
    df_combined = pd.concat([df_history_raw, df_new_lists], axis=0)
    # Safety: combined dataset must not contain duplicate indices
    if df_combined.index.duplicated().any():
        dupes = df_combined.index[df_combined.index.duplicated()].unique()
        print("Error: Duplicate dates detected after overlap filtering (this should not happen).")
        print(f"  Examples: {list(dupes[:10])}" + (" ..." if len(dupes) > 10 else ""))
        return
    
    print(f"Combined dataset: {len(df_combined)} rows × {len(df_combined.columns)} columns")
    print("  (History converted to raw OHLC to allow indicator warm-up)")
    
    # WARNING: shift_data_down will align data to the bottom of the DataFrame.
    # If a stock stopped trading (delisted/missing data), its old historical prices
    # will be shifted forward to fill empty new rows, causing data corruption.
    # Ensure your input CSV does not contain columns for stocks that are no longer trading.
    
    # --- Step 5: Process Combined Dataset ---
    print(f"\nStep 5: Processing combined dataset...")
    
    # Apply shift_data_up
    df_combined_shifted_up, offsets = shift_data_up(df_combined)
    
    # Apply process_dataframe
    print(f"Processing {len(df_combined_shifted_up.columns)} columns...")
    df_combined_processed = process_dataframe(df_combined_shifted_up)
    
    # Apply shift_data_down
    # Pass offsets to ensure we restore data to correct time slots
    df_combined_shifted = shift_data_down(df_combined_processed, offsets=offsets)
    
    # Apply manipulate_list to filter to 9 indices
    print("Filtering lists to final 9 elements...")
    df_combined_final = df_combined_shifted.map(manipulate_list)
    
    # Extract only the new rows
    # FIX: We have HISTORY_ROWS (100) which serves as the warm-up period.
    # Therefore, the indicators are stable by the time we hit the first new row.
    # We do NOT need to discard the first 34 rows of the new data.
    
    new_data_indices = list(df_new_lists.index)
    
    # Use intersection to find valid rows that exist in the processed output
    # (This handles the alignment safely without slicing off valid data)
    available_indices = df_combined_final.index.intersection(new_data_indices)
    
    if len(available_indices) > 0:
        df_new_processed = df_combined_final.loc[available_indices].copy()
        print(f"Extracted {len(df_new_processed)} new processed rows")
        print(f"  (Used {HISTORY_ROWS} history rows as warm-up period)")
    else:
        print(f"⚠️  Error: No valid new rows extracted.")
        print(f"  Available indices in combined: {len(df_combined_final.index)}")
        print(f"  New data indices: {len(new_data_indices)}")
        print(f"  Intersection: {len(available_indices)}")
        return

    # Final safety: do not allow overlapping indices to be appended
    overlap_with_prod = df_production_full.index.intersection(df_new_processed.index)
    if len(overlap_with_prod) > 0:
        print("Error: Processed new rows overlap production indices (refusing to append).")
        print(f"  Examples: {list(overlap_with_prod[:10])}" + (" ..." if len(overlap_with_prod) > 10 else ""))
        return
    
    # --- Step 6: Validation Check ---
    print(f"\nStep 6: Validating calculations...")
    
    # The last row of the history section in the combined dataset (at index HISTORY_ROWS - 1)
    # should match the last row of production (index -1)
    # Note: After shift operations, the indices might have changed, so we need to find
    # the row that corresponds to the last production row by matching the date/index
    if len(df_combined_final) > HISTORY_ROWS - 1:
        # Find the row in combined_final that has the same index as production's last row
        production_last_index = df_production.index[-1]
        
        if production_last_index in df_combined_final.index:
            recalculated_last_row = df_combined_final.loc[production_last_index]
            production_last_row = df_production.iloc[-1]
            
            # Compare row by row (cell by cell)
            mismatches = []
            for col in df_production.columns:
                recalc_val = recalculated_last_row[col]
                prod_val = production_last_row[col]
                
                # Handle None values
                if recalc_val is None and prod_val is None:
                    continue
                if recalc_val is None or prod_val is None:
                    mismatches.append(f"{col}: recalculated={recalc_val}, production={prod_val}")
                    continue
                
                # Compare lists
                if isinstance(recalc_val, list) and isinstance(prod_val, list):
                    if len(recalc_val) != len(prod_val):
                        mismatches.append(f"{col}: length mismatch (recalc={len(recalc_val)}, prod={len(prod_val)})")
                    else:
                        for i, (r, p) in enumerate(zip(recalc_val, prod_val)):
                            if r != p and not (pd.isna(r) and pd.isna(p)):
                                # Allow small floating point differences
                                if isinstance(r, (int, float)) and isinstance(p, (int, float)):
                                    if abs(r - p) > 1e-6:
                                        mismatches.append(f"{col}[{i}]: recalc={r}, prod={p} (diff={abs(r-p)})")
                                else:
                                    mismatches.append(f"{col}[{i}]: recalc={r}, prod={p}")
                elif recalc_val != prod_val:
                    # Allow small floating point differences for numeric values
                    if isinstance(recalc_val, (int, float)) and isinstance(prod_val, (int, float)):
                        if abs(recalc_val - prod_val) > 1e-6:
                            mismatches.append(f"{col}: recalc={recalc_val}, prod={prod_val} (diff={abs(recalc_val-prod_val)})")
                    else:
                        mismatches.append(f"{col}: recalc={recalc_val}, prod={prod_val}")
            
            if mismatches:
                print(f"⚠️  Warning: {len(mismatches)} mismatches detected in validation check:")
                for mismatch in mismatches[:10]:  # Show first 10
                    print(f"  {mismatch}")
                if len(mismatches) > 10:
                    print(f"  ... and {len(mismatches) - 10} more")
                response = input("\nValidation check failed. Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    print("Aborted by user.")
                    return
            else:
                print("✓ Validation check passed: Recalculated last row matches production")
        else:
            print(f"⚠️  Warning: Cannot find production last index {production_last_index} in combined dataset for validation")
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Aborted by user.")
                return
    else:
        print("⚠️  Warning: Cannot perform validation check (insufficient rows)")
    
    # --- Step 7: Append and Save ---
    print(f"\nStep 7: Appending new rows to production dataset...")
    
    # Append processed new rows to production
    df_final = pd.concat([df_production_full, df_new_processed], axis=0)
    print(f"Final dataset: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print(f"  (Original: {len(df_production_full)} rows, Added: {len(df_new_processed)} rows)")
    
    # Generate output filename
    output_filename = generate_output_filename(production_file)
    # For CSV, replace .pkl with _FOR_COMPARE.csv in the output filename
    output_csv_filename = output_filename.replace('.pkl', '_FOR_COMPARE.csv')
    
    # Save pickle
    print(f"\nSaving final dataset to {output_filename}...")
    tmp_pkl = output_filename + ".tmp"
    df_final.to_pickle(tmp_pkl)
    os.replace(tmp_pkl, output_filename)
    
    # Save CSV for comparison
    print(f"Converting to strings for CSV export...")
    df_string_output = df_final.map(
        lambda x: str(x) if x is not None else ""
    )
    print(f"Saving comparison CSV to {output_csv_filename}...")
    tmp_csv = output_csv_filename + ".tmp"
    df_string_output.to_csv(tmp_csv)
    os.replace(tmp_csv, output_csv_filename)
    
    # --- Step 8: Validate Overlapping Data ---
    print(f"\nStep 8: Validating overlapping data between old and new production files...")
    
    # Load the newly saved file
    df_new_production = pd.read_pickle(output_filename)
    
    # Find overlapping indices (dates that exist in both)
    overlapping_indices = df_production_full.index.intersection(df_new_production.index)
    
    if len(overlapping_indices) > 0:
        print(f"Found {len(overlapping_indices)} overlapping rows to validate...")
        
        # Compare overlapping rows
        df_old_overlap = df_production_full.loc[overlapping_indices]
        df_new_overlap = df_new_production.loc[overlapping_indices]
        
        mismatches = []
        for idx in overlapping_indices:
            old_row = df_old_overlap.loc[idx]
            new_row = df_new_overlap.loc[idx]
            
            for col in df_production_full.columns:
                old_val = old_row[col]
                new_val = new_row[col]
                
                # Handle None values
                if old_val is None and new_val is None:
                    continue
                if old_val is None or new_val is None:
                    mismatches.append(f"Row {idx}, Column {col}: old={old_val}, new={new_val}")
                    continue
                
                # Compare lists
                if isinstance(old_val, list) and isinstance(new_val, list):
                    if len(old_val) != len(new_val):
                        mismatches.append(f"Row {idx}, Column {col}: length mismatch (old={len(old_val)}, new={len(new_val)})")
                    else:
                        for i, (o, n) in enumerate(zip(old_val, new_val)):
                            if o != n and not (pd.isna(o) and pd.isna(n)):
                                # Allow small floating point differences
                                if isinstance(o, (int, float)) and isinstance(n, (int, float)):
                                    if abs(o - n) > 1e-6:
                                        mismatches.append(f"Row {idx}, Column {col}[{i}]: old={o}, new={n} (diff={abs(o-n)})")
                                else:
                                    mismatches.append(f"Row {idx}, Column {col}[{i}]: old={o}, new={n}")
                elif old_val != new_val:
                    # Allow small floating point differences for numeric values
                    if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                        if abs(old_val - new_val) > 1e-6:
                            mismatches.append(f"Row {idx}, Column {col}: old={old_val}, new={new_val} (diff={abs(old_val-new_val)})")
                    else:
                        mismatches.append(f"Row {idx}, Column {col}: old={old_val}, new={new_val}")
        
        if mismatches:
            print(f"⚠️  ERROR: {len(mismatches)} mismatches found in overlapping data!")
            print("This indicates the new production file has different values than the old one for overlapping rows.")
            for mismatch in mismatches[:20]:  # Show first 20
                print(f"  {mismatch}")
            if len(mismatches) > 20:
                print(f"  ... and {len(mismatches) - 20} more mismatches")
            print(f"\n⚠️  CRITICAL: Overlapping data validation failed!")
            print(f"   The new production file does not match the old one for {len(overlapping_indices)} overlapping rows.")
            response = input("\nValidation failed. Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Aborted. Please investigate the issue.")
                # Optionally remove the new file
                if os.path.exists(output_filename):
                    print(f"Removing {output_filename}...")
                    os.remove(output_filename)
                if os.path.exists(output_csv_filename):
                    os.remove(output_csv_filename)
                return
        else:
            print(f"✓ Overlapping data validation passed: All {len(overlapping_indices)} overlapping rows match")
    else:
        print("✓ No overlapping rows found (new data starts after old data ends)")
    
    # --- Step 9: Update Configuration Files ---
    if update_config:
        print(f"\nStep 9: Updating configuration files to point to new production file...")
        try:
            update_config_files(output_filename)
            print(f"✓ Configuration files updated")
        except Exception as e:
            print(f"⚠️  Warning: Failed to update configuration files: {e}")
            print(f"   Please manually update OUTPUT_FILE_PKL in process_eigen_data.py to: {output_filename}")
    else:
        print(f"\nStep 9: Skipping configuration file updates (run with --update-config to enable).")
    
    print(f"\n✅ Success! Append mode complete.")
    print(f"   Production dataset: {production_file}")
    print(f"   New data added: {len(df_new_processed)} rows")
    print(f"   Output saved to: {output_filename}")
    print(f"   CSV saved to: {output_csv_filename}")
    print(f"   Configuration updated to use: {output_filename}")

# --- Main Execution Function ---
def main():
    """Main execution function."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Process Eigen data: convert raw CSV to processed pickle file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Normal mode:
    python process_eigen_data.py
  
  Append mode:
    python process_eigen_data.py --append --production-file Eigen2_Master_PY_OUTPUT_151025.pkl --new-data-file new_data.csv
        """
    )
    
    parser.add_argument('--append', action='store_true',
                        help='Enable append mode: append new data to existing production dataset')
    parser.add_argument('--production-file', type=str,
                        help='Production pickle file (required for --append mode)')
    parser.add_argument('--new-data-file', type=str,
                        help='New raw CSV data file (required for --append mode)')
    parser.add_argument('--update-config', action='store_true',
                        help='(Append mode) Update OUTPUT_FILE_PKL references in repo to point at the newly written pickle')
    parser.add_argument('--allow-overlap', action='store_true',
                        help='(Append mode) Allow new-data CSV to include rows on/before the last production date; those rows will be dropped for safety')
    
    args = parser.parse_args()
    
    # Branch to append mode if requested
    if args.append:
        if not args.production_file:
            print("Error: --production-file is required when using --append mode")
            parser.print_help()
            return
        if not args.new_data_file:
            print("Error: --new-data-file is required when using --append mode")
            parser.print_help()
            return
        
        append_mode(
            args.production_file,
            args.new_data_file,
            update_config=args.update_config,
            allow_overlap=args.allow_overlap,
        )
        return
    
    # Normal mode (existing functionality)
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    print(f"Loading {INPUT_FILE}...")
    # Assume first column is the index (e.g., Date)
    df = pd.read_csv(INPUT_FILE, index_col=0)

    # Remove TRANSFER column if present (artifact from Excel, not needed for training)
    if 'TRANSFER' in df.columns:
        df = df.drop(columns=['TRANSFER'])
        print("Dropped TRANSFER column")

    # Calculate row count from the data (column A is now the index)
    DATA_ROW_COUNT = len(df)
    print(f"Detected {DATA_ROW_COUNT} rows of data from CSV.")

    # --- NEW, BETTER CLEANING STEPS ---

    # 1. Convert all empty strings ("") or whitespace-only strings to NaN
    #    (This is the "missing link" that makes dropna work)
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # 2. NOW, drop all columns that are "all" NaN
    df = df.dropna(axis=1, how='all')

    # --- Replicates 'SingleIntoArray' sub ---
    print("Formatting data from single values to lists...")
    df_lists = df.map(format_cell)

    # --- NEW STEP 0: Shift data UP (top-align) before processing ---
    # This handles columns where data doesn't start at row 1
    df_shifted_up, col_offsets = shift_data_up(df_lists)

    # --- Replicates 'DataRun' sub ---
    print(f"Processing {len(df_shifted_up.columns)} columns...")
    df_processed = process_dataframe(df_shifted_up)

    # --- NEW STEP 1: Replicates 'ShiftDataDown' sub ---
    # Shift data back down to bottom-align (original positions)
    df_shifted = shift_data_down(df_processed, offsets=col_offsets)

    # --- NEW STEP 2: Replicates 'ManipulateArrayString' ---
    print("Filtering lists to final 9 elements...")
    # .map() applies the function to every single cell
    df_final = df_shifted.map(manipulate_list)

    # --- FINAL OUTPUTS ---
    VBA_CALC_RAMP_UP_ROWS = 34 # (This is 34 rows, index 0-33)

    # --- STEP 1: Slice off the top 34 rows FIRST ---
    print(f"Slicing off top {VBA_CALC_RAMP_UP_ROWS} rows for all outputs...")
    
    # .iloc[34:] keeps every row *from* the 35th row (index 34) onwards.
    df_sliced = df_final.iloc[VBA_CALC_RAMP_UP_ROWS:]

    # --- STEP 2: Output to Pickle (for Eigen 2) ---
    print(f"Saving final (sliced) DataFrame to {OUTPUT_FILE_PKL}...")
    df_sliced.to_pickle(OUTPUT_FILE_PKL)

    # --- STEP 3: Output to CSV (for comparison) ---
    print(f"Converting sliced lists to strings for CSV export...")
    df_string_output = df_sliced.map(
        lambda x: str(x) if x is not None else ""
    )
    
    print(f"Saving comparison CSV to {OUTPUT_FILE_CSV}...")
    df_string_output.to_csv(OUTPUT_FILE_CSV)

    # --- STEP 4: FINAL DEBUG CHECK (Trust, but Verify) ---
    print("\n--- 🕵️‍♂️ Final Sanity Check ---")
    
    # Check 1: Load the PKL file we *just* saved and count its columns.
    # This is the ultimate proof of what Eigen 2 will receive.
    try:
        df_from_pkl = pd.read_pickle(OUTPUT_FILE_PKL)
        print(f"Columns in {OUTPUT_FILE_PKL} (on disk): {len(df_from_pkl.columns)}")
    except Exception as e:
        print(f"Error reading back {OUTPUT_FILE_PKL}: {e}")

    # Check 2: Count the columns in the DataFrame we *prepared* for the CSV.
    # This must match the PKL count.
    print(f"Columns in {OUTPUT_FILE_CSV} (in memory): {len(df_string_output.columns)}")
    print("--------------------------------")


    print(f"\n✅ Success! All steps complete.")
    print(f"Pickle for Eigen 2 saved to: {OUTPUT_FILE_PKL}")
    print(f"Comparison CSV saved to: {OUTPUT_FILE_CSV}")

if __name__ == "__main__":
    main()