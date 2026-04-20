# https://github.com/k-off/ft_linear_regression/tree/master
import  sys
import  matplotlib.pyplot as plt
from pathlib import Path
import  os
# Clase linear_regression
from ft_linear_regression import linear_regression

flags = {"plot_standardized": 0, "plot_original": 0, "print_error": 0}
learning_rate = 0

# checking flags and filename of data file
no_flag_cont = 0
data_filename = ""
for arg in sys.argv :
    if arg.startswith("-"): # check flags
        check = False
        if arg == "-s" :
            flags["plot_standardized"] = 1
            check = True
        if arg == "-o" :
            flags["plot_original"] = 1
            check = True
        if arg == "-err" :
            flags["print_error"] = 1
            check = True
        if arg == "-e" :
            learning_rate = -1.0
            check = True
        if check == False:
            print ("Error: Unknown flag.", arg)
            print ("Flags: -s, -o, -err, -e.")
            sys.exit()
    else:
        no_flag_cont = no_flag_cont + 1
        data_filename = arg
if no_flag_cont > 2:
    print ("Error: Too many no flag parameters.", no_flag_cont)
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

if (learning_rate < 0.0000001 or learning_rate > 1) :
    learning_rate = 0.1

print(flags)
print(data_filename)

data = linear_regression(data_filename)

data.train_model(learning_rate, flags["print_error"])
data.save_model("my_model")

data2 = linear_regression()
data2.load_model("my_model.csv")
data2.print_coeficientes()
print(data2.predict(100000))

if flags["plot_original"] == 1 :
    data.plot_value()
if flags["plot_standardized"] == 1 :
    data.plot_standardized_value()