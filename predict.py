import  os
from pathlib import Path
from ft_linear_regression import linear_regression

model = linear_regression()

#model.load_model("my_model.csv")
model.print_coeficientes()

# Enter mileage
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
print ("Your car price is :", model.predict(number_mileage))