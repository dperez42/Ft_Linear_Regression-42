import  os
import  sys
from pathlib import Path
from ft_linear_regression import linear_regression


model_file = ""
if len(sys.argv)!=2:
    print ("Error: Too many parameters.")
    print ("Usage: python <model_file>")
    sys.exit()
else:
    model_file=sys.argv[1]

model = linear_regression()
model.load_model(model_file)
# Show coeficientes
model.print_coeficientes()
# Enter mileage and check is valid nummber
check = False
while check == False :
    mileage = input("Enter mileage: ")
    try :
        number_mileage = float(mileage) - 0
        if (number_mileage >= 0) :
            check = True
        else :
            print ("Error: negative mileage? Try again.")
    except :
        print ("Error: not a number, try again.")
pred = int(model.predict(number_mileage))
if pred <= 0:
    print ("Your car is trash. Unable to sold.")
else:
    print ("Your car price is :", int(model.predict(number_mileage)))