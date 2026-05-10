# https://github.com/k-off/ft_linear_regression/tree/master
import  sys
import  matplotlib.pyplot as plt
from pathlib import Path
import  os
# Clase linear_regression
from ft_linear_regression import linear_regression
from datetime import datetime

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"  # Reset to default color
ORANGE = "\033[38;5;208m" 

flags = {"scaler": 1, "plot_standardized": 0, "plot_original": 0, "plot_loss":0, "print_error": 0, "loss_function": "MAE", "target_function": "DELTA_LOSS", "target_value":0, "learning_rate":0}
learning_rate = 0

# checking flags and filename of data file
no_flag_cont = 0
scaler_cont = 0
loss_cont = 0
target_cont = 0
data_filename = ""
error = False
help = False
for arg in sys.argv :
    if arg.startswith("-"): # check flags
        check = False
        if arg =="-h":
            error = True
            help = True
            break
        if arg == "-sm" :
            flags["scaler"] = 1
            scaler_cont += 1
            check = True
        if arg == "-ss" :
            flags["scaler"] = 2
            scaler_cont += 1
            check = True
        if arg == "-ps" :
            flags["plot_standardized"] = 1
            check = True
        if arg == "-po" :
            flags["plot_original"] = 1
            check = True
        if arg == "-pl" :
            flags["plot_loss"] = 1
            check = True
        if arg == "-po" :
            flags["plot_original"] = 1
            check = True
        if arg == "-err" :
            flags["print_error"] = 1
            check = True
        if arg == "-mae" :
            flags["loss_function"] = 'MAE'
            loss_cont += 1
            check = True
        if arg == "-mse" :
            flags["loss_function"] = 'MSE'
            loss_cont += 1
            check = True
        if arg == "-dd" :
            flags["target_function"] = 'DELTA_LOSS'
            target_cont += 1
            check = True
        if arg == "-de" :
            flags["target_function"] = 'NUMBER_EPOCHS'
            target_cont += 1
            check = True
        if check == False:
            print (f"{RED}Error: Unknown flag.{RESET}", arg)
            error = True
    
    else:
        no_flag_cont = no_flag_cont + 1
        data_filename = arg

if (scaler_cont>1):
    print (f"{RED}Error: Too many scaler methods choose one. -sm or -ss.{RESET}", scaler_cont)
    error = True
    
if (loss_cont>1):
    print (f"{RED}Error: Too many loss function chosse only one. -mae or -mse.{RESET}", loss_cont)
    error = True
if (target_cont>1):
    print (f"{RED}Error: Too many target function chosse only one. -dd or -de.{RESET}", loss_cont)
    error = True
if no_flag_cont != 2 and not help:
    print (f"{RED}Error: Incorrect number of parameters.{RESET}", no_flag_cont)
    error = True
    
if error:
    print (f"{GREEN}Usage: python train.py <data_file.csv> <flags>{RESET}")
    print (f"{YELLOW}Choose Standarization method flags:")
    print (f"-sm (default) Min Max Feature Scaling")
    print (f"-ss {RESET} Standarization Feature Scaling")
    print (f"{BLUE}Choose Loss function flags:")
    print (f"-mae (default)")
    print (f"-mse{RESET}")
    print (f"{ORANGE}Choose Objective flags:")
    print (f"-dd (default= 0.0000001) : Define delta loss between epochs")
    print (f"-de : Define number of epochs{RESET}")
    print ("Plotting flags:")
    print ("-po : data")
    print ("-ps : Plot standarize data")
    print ("-pl : Plot loss")
    print ("View logs:")
    print ("-err ")
    sys.exit()
if (flags["target_function"]=="NUMBER_EPOCHS"):
    val = input("Number of epochs (1000): ")
    if not val:
       val = 1000
    print(val)
    try:
        flags["target_value"] = int(val)
    except ValueError:
        print(f"{RED}{val} is NOT a valid positive integer number.{RESET}")
    if (flags["target_value"] < 1) :
        print (f"{RED}Error: Number of epochs >0{RESET}")
        sys.exit()
if (flags["target_function"]=="DELTA_LOSS"):
    val = input("Delta Loss ( 0.0000001): ")
    if not val:
        val = 0.0000001
    print(val)
    try:
        flags["target_value"] = float(val)
    except ValueError:
        print(f"{RED}{val} is NOT a valid positive number.{RESET}")

val = input("Input Learning Rate <0.00000001 or >1 (0.1) :")
if not val:
    val = 0.1
try:
    flags["learning_rate"] = float(val)
except ValueError:
    print(f"{RED}{val} is NOT a valid positive number.{RESET}")
if (flags["learning_rate"] < 0.0000001 or flags["learning_rate"] > 1) :
    print (f"{RED}Error: Learning rate <0.00000001 or >1{RESET}")
    sys.exit()

# Check if file is correct
check_file = Path(data_filename)
# check file exist
if check_file.is_file()== False:
    sys.exit("Error: File {:} not exist".format(check_file))
# check file permission
if (os.access(check_file, os.R_OK) == False) :
    sys.exit("Error:    Access denied for " + check_file)
# check extension of the file
if check_file.suffix != '.csv':
    sys.exit("Error: File {:} has to be a csv".format(check_file))

print(flags)
#exit()
print(data_filename)

# Start Trainning
print("Start Training.")
# Create object
data = linear_regression(data_filename)
# Train
data.train_model(flags) 
# Save
filename = "modelo"
fecha_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename_with_date = f"{filename}_{fecha_actual}.csv"
data.save_model("model")
# Show results
if flags["plot_standardized"] == 1 :
    data.plot(std = True, predict = True)
if flags["plot_original"] == 1 :
    data.plot(std = False, predict = True)
if flags["plot_loss"] == 1 :
    data.plot_loss()