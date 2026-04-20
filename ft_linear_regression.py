# https://github.com/k-off/ft_linear_regression/tree/master
import  csv
import  sys
import  os.path
import  matplotlib.pyplot as plt
from pathlib import Path

class linear_regression :

    def __init__(self, filename="") :

        # init parameters
        self.size = {0, 0}
        self.data = []
        if filename!="":
            self.load_data(filename)        # load data 
        self.theta_0 = 0.0
        self.theta_1 = 0.0
        self.tmp_theta_0 = 1.0
        self.tmp_theta_1 = 1.0
        self.prev_mse = 0.0
        self.cur_mse = self.mean_square_error()
        self.delta_mse = self.cur_mse        

    def print_data(self) :
        for row in self.data:
            for column in row:
                print (column, end="\t")
            print ("")
    
    def print_coeficientes(self) :
        print("theta_0:",self.theta_0)
        print("theta_1:",self.theta_1)
        print ("")
        
    def load_data(self, filename) :
        with open(filename, 'r') as csv_file:
            try:
                dict_val = csv.reader(csv_file, delimiter = ",")
                for row in dict_val:
                    self.data.append(row)
            except:
                sys.exit("Error: File {:} cannot be read".format(csv_file))
        print("Raw data:")
        self.print_data()

    def mean_square_error(self) :

        i = 0
        tmp_summ = 0

        for line in self.data :
            if (i > 0) :
                tmp_diff = self.predict_tmp(line[0]) - float(line[1])
                tmp_diff *= tmp_diff
                tmp_summ += tmp_diff
            i += 1

        return (tmp_summ / (i - 1))

    def get_gradient0(self) :
        i = 0
        tmp_summ = 0.0

        for line in self.data :
            if (i > 0) :
                tmp_summ += (self.predict_tmp(line[0]) - float(line[1]))
            i += 1
        
        return (self.learning_rate * (tmp_summ / (i - 1)))

    def get_gradient1(self) :
        i = 0
        tmp_summ = 0.0

        for line in self.data :
            if (i > 0) :
                tmp_summ += (self.predict_tmp(line[0]) - float(line[1])) \
                    * float(line[0])
            i += 1
        
        return (self.learning_rate * (tmp_summ / (i - 1)))

    def set_min_max(self) :
        i = 0
        self.min_x = 2 ** 32 / 1.0
        self.max_x = 2 ** 32 / -1.0
        self.min_y = 2 ** 32 / 1.0
        self.max_y = 2 ** 32 / -1.0
        for line in self.data :
            if (i > 0) :
                if float(line[0]) < self.min_x :
                    self.min_x = float(line[0])
                if float(line[0]) > self.max_x :
                    self.max_x = float(line[0])
                if float(line[1]) < self.min_y :
                    self.min_y = float(line[1])
                if float(line[1]) > self.max_y :
                    self.max_y = float(line[1])
            i += 1
    
    def standardize(self) :
        i = 0
        self.set_min_max()
        for line in self.data :
            if (i > 0) :
                line[0] = (float(line[0]) - self.min_x) / \
                    (self.max_x - self.min_x)
                line[1] = (float(line[1]) - self.min_y) / \
                    (self.max_y - self.min_y)
            i += 1
        print("Standardize data:")
        self.print_data()

    def plot_value(self) :
        tmp_val = self.data
        tmp_val.pop(0)

        tmp_theta0 = self.tmp_theta_0
        tmp_theta1 = self.tmp_theta_1

        self.tmp_theta_0 = self.theta_0
        self.tmp_theta_1 = self.theta_1
        print (self.min_y, self.max_y)
        tmp = list(zip(*tmp_val))
        tmp = [list(tmp[0]), list(tmp[1])]
        plot_val = [[], []]
        for i in tmp[0] :
            i = self.min_x + (self.max_x - self.min_x) * i
            plot_val[0].append(i)
        for i in tmp[1] :
            i = self.min_y + (self.max_y - self.min_y) * i
            plot_val[1].append(i)
        plt.title('Real values')
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.plot(plot_val[0], plot_val[1], 'ro')
        plt.plot([self.min_x, self.max_x], [self.estimatePrice(self.min_x), \
            self.estimatePrice(self.max_x)])
        plt.axis([self.min_x - abs(self.max_x * 0.1), self.max_x + \
            abs(self.max_x * 0.1), self.min_y - abs(self.max_y * 0.1), \
            self.max_y + abs(self.max_y * 0.1)])
        plt.show()
        self.tmp_theta_0 = tmp_theta0
        self.tmp_theta_1 = tmp_theta1

    def plot_standardized_value(self) :
        tmp_val = self.data
        tmp_val.pop(0)

        tmp_val = list(zip(*tmp_val))
        tmp_val = [list(tmp_val[0]), list(tmp_val[1])]
        plt.title('Standardized values')
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.plot(tmp_val[0], tmp_val[1], 'ro')
        plt.plot([0, 1], [self.estimatePrice(0), self.estimatePrice(1)])
        plt.show()

    def save_model(self,filename) :
        f = open(filename+".csv", "w+")
        f.write("%f, %f" %(self.theta_0, self.theta_1))
        f.close()
    
    def load_model(self, filename) :
        with open(filename, 'r') as csv_file:
            try:
                dict_val = csv.reader(csv_file, delimiter = ",")
                for row in dict_val:
                    self.theta_0=float(row[0])
                    self.theta_1=float(row[1])
            except:
                sys.exit("Error: File {:} cannot be read".format(csv_file))
    
    def train_model(self, learning_rate, print_error) :
        print("Training Model.")
        self.learning_rate = learning_rate
        self.standardize()
        print("max, min", self.min_x, self.max_x, self.min_y, self.max_y)
        while self.delta_mse > 0.0000001 or self.delta_mse < -0.0000001 :
            self.theta_0 = self.tmp_theta_0
            self.theta_1 = self.tmp_theta_1
            self.tmp_theta_0 -= self.get_gradient0()
            self.tmp_theta_1 -= self.get_gradient1()
            self.prev_mse = self.cur_mse
            self.cur_mse = self.mean_square_error()
            if (print_error == 1) :
                print (self.cur_mse)
            self.delta_mse = self.cur_mse - self.prev_mse

        self.theta_1 = (self.max_y - self.min_y) * self.theta_1 / \
            (self.max_x - self.min_x)
        self.theta_0 = self.min_y + ((self.max_y - self.min_y) * \
            self.theta_0) + self.theta_1 * (1 - self.min_x)

    def predict_tmp(self, value) :
        return ((self.tmp_theta_0 + (self.tmp_theta_1 * float(value))))
    
    def predict(self, value) :
        if (self.theta_0==0.0 and self.theta_0==0.0):
            sys.exit("Error: No model load.")
        value0 = float(value) - 0
        return ((self.theta_0 + (self.theta_1 * float(value0) )))
        