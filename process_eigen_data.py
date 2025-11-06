import pandas as pd
import numpy as np
import time
import sys
import os

# --- Configuration ---
INPUT_FILE = 'Eigen2_Master(GFIN)_04_OutputOnly2_shiftup.csv'
OUTPUT_FILE_PKL = 'Eigen2_Master_PY_OUTPUT.pkl'
OUTPUT_FILE_CSV = 'Eigen2_Master_PY_OUTPUT_FOR_COMPARE.csv'

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

# --- NEW STEP 1: Replicates ShiftDataDown Sub ---
def shift_data_down(df):
    """
    Replicates the VBA ShiftDataDown logic.
    Aligns all data to the bottom of the DataFrame, column by column.
    This version is "future-proof" and silences pandas warnings.
    """
    print("Shifting data to bottom-align (replicating ShiftDataDown)...")
    
    # --- FIX 1: Start from the beginning (index 0) ---
    pandas_start_index = 0
    num_rows = df.shape[0]
    
    # Create a new, empty DataFrame to hold the shifted data
    new_df = pd.DataFrame(None, index=df.index, columns=df.columns)
    
    # --- FIX 2: DELETED the 'new_df.iloc[0] = df.iloc[0]' line ---
    # We no longer "preserve" any rows. All data is shifted.
    
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

# --- Main Execution Function ---
def main():
    """Main execution function."""
    
    VBA_DATA_ROW_COUNT = 3922 # (This is Excel row 2 to 3923)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    print(f"Loading {INPUT_FILE}...")
    # Assume first column is the index (e.g., Date)
    df = pd.read_csv(INPUT_FILE, index_col=0)
    
    # --- NEW, BETTER CLEANING STEPS ---
    
    # 1. Convert all empty strings ("") or whitespace-only strings to NaN
    #    (This is the "missing link" that makes dropna work)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # 2. NOW, drop all columns that are "all" NaN
    df = df.dropna(axis=1, how='all')
    
    # Force the DataFrame to match the VBA's hard-coded 3922 row-count
    if len(df) > VBA_DATA_ROW_COUNT:
        print(f"File is {len(df)} rows. Slicing to VBA limit of {VBA_DATA_ROW_COUNT} rows.")
        df = df.iloc[:VBA_DATA_ROW_COUNT]

    # --- Replicates 'SingleIntoArray' sub ---
    print("Formatting data from single values to lists...")
    def format_cell(x):
        # x is a string like "[43.77,43.23,43.8,43.2]" or ""
        
        # Check for empty/invalid strings first
        if not isinstance(x, str) or x == "" or x == "[]" or not x.startswith('['):
            return None
        
        try:
            # Remove brackets: "[1,2,3]" -> "1,2,3"
            list_string = x[1:-1]
            
            # Split by comma and convert each part to a float
            return [float(part) for part in list_string.split(',')]
        
        except (ValueError, TypeError):
            # Catches errors if a part isn't a valid number
            return None
            
    df_lists = df.map(format_cell)
    
    # --- Replicates 'DataRun' sub ---
    print(f"Processing {len(df_lists.columns)} columns...")
    df_processed = process_dataframe(df_lists)

    # --- NEW STEP 1: Replicates 'ShiftDataDown' sub ---
    df_shifted = shift_data_down(df_processed)

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