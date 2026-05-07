# https://github.com/k-off/ft_linear_regression/tree/master
import  csv
import  sys
import  os.path
import  matplotlib.pyplot as plt
from pathlib import Path

class linear_regression :

    def __init__(self, filename="") :

        # init parameters
        self.flags = {}
        self.data = []
        self.data_scaler = []
        self.scaler = 1
        self.loss_function = 1
        if filename!="":
            self.load_data(filename)        # load data 
        self.theta_0 = 0.0
        self.theta_1 = 0.0
        self.tmp_theta_0 = 1.0
        self.tmp_theta_1 = 1.0
        self.prev_mse = 0.0
        self.cur_mse = 0.0
        self.delta_mse = self.cur_mse        
        self.loss_acc = []

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
        try:
            with open(filename, 'r') as csv_file:
                try:
                    dict_val = csv.reader(csv_file, delimiter = ",")
                    for row in dict_val:
                        self.data.append(row)
                except:
                    sys.exit("Error: File {:} cannot be read".format(csv_file))
        except:
            sys.exit("Error: File {:} not found".format(filename))

        #print("Raw data:")
        #self.print_data()

    def errors(self):
        """
        Calcula métricas de error para el modelo:
        MAE, MSE, RMSE, Loss promedio y R².
        """
        # Ignorar la primera fila (cabecera)
        data = self.data_scaler[1:]

        n = len(data)
        if n == 0:
            return None  # Evitar división por cero

        mae_sum = 0
        mse_sum = 0
        loss_sum = 0

        y_real = []
        y_pred = []

        for x, y in data:
            y_real_val = float(y)
            y_pred_val = self.predict_tmp(x)

            diff = y_pred_val - y_real_val

            mae_sum += abs(diff)
            mse_sum += diff ** 2
            loss_sum += diff

            y_real.append(y_real_val)
            y_pred.append(y_pred_val)

        # MAE: Es el promedio de la suma de los valores absolutos de la diferencia entre los valores predichos y los valores reales
        mae = mae_sum / n
        # MSE: Es el promedio de la suma de la diferencia al cuadrado entre los valores predichos y los valores reales.
        mse = mse_sum / n
        # RMSE: Es la raíz cuadrada del error cuadrático medio (ECM).
        rmse = mse ** 0.5
        # R²
        mean_y = sum(y_real) / n
        ss_res = sum((y_real[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y_real[i] - mean_y) ** 2 for i in range(n))
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return mae, mse, rmse, r2

    def get_gradientBiass(self) :
        i = 0
        tmp_summ = 0.0
        for line in self.data_scaler :
            if (i > 0) :
                if (self.loss_function=="MAE"):
                    tmp_summ += (self.predict_tmp(line[0]) - float(line[1]))
                if (self.loss_function=="MSE"):
                    tmp_summ += (self.predict_tmp(line[0]) - float(line[1]))*2
            i += 1
        
        return (tmp_summ / (i - 1))
    
    def get_gradientWeight(self) :
        i = 0
        tmp_summ = 0.0

        for line in self.data_scaler :
            if (i > 0) :
                if (self.loss_function=="MAE"):
                    tmp_summ += (self.predict_tmp(line[0]) - float(line[1])) * float(line[0])
                if (self.loss_function=="MSE"):
                    tmp_summ += (self.predict_tmp(line[0]) - float(line[1])) * 2 * float(line[0])
            i += 1
        
        return (tmp_summ / (i - 1))
    
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
        print(f"Max X: {self.max_x}, Min X: {self.min_x}")
        print(f"Max Y: {self.max_y}, Min Y: {self.min_y}")
    
    def set_mean_stddev(self) :
        # Skip the first row (header)
        data_rows = self.data[1:]
        n = len(data_rows)
        x_values = [float(row[0]) for row in data_rows]
        y_values = [float(row[1]) for row in data_rows]
        self.mean_x = 0
        self.mean_y = 0
        self.std_dev_x = 0
        self.std_dev_y = 0
        # --- Mean calculation --
        self.mean_x = sum(x_values) / n
        self.mean_y = sum(y_values) / n
        # --- Standard deviation calculation (sample std dev, ddof=1) ---
        if n > 1:
            variance_x = sum((x - self.mean_x) ** 2 for x in x_values) / (n - 1)
            variance_y = sum((y - self.mean_y) ** 2 for y in y_values) / (n - 1)
            self.std_dev_x = variance_x ** 0.5
            self.std_dev_y = variance_y ** 0.5
        else:
            self.std_dev_x = 0
            self.std_dev_y = 0
         # Debug output
        print(f"Mean X: {self.mean_x}, StdDev X: {self.std_dev_x}")
        print(f"Mean Y: {self.mean_y}, StdDev Y: {self.std_dev_y}")
    
    def ScalerMinMax(self) :
        print("Scaling MinMax:")
        # valores entre 0 -1
        i = 0
        self.set_min_max()
        self.data_scaler = [row[:] for row in self.data]
        for line in self.data_scaler :
            if (i > 0) :
                line[0] = (float(line[0]) - self.min_x) / (self.max_x - self.min_x)
                line[1] = (float(line[1]) - self.min_y) / (self.max_y - self.min_y)
            i += 1    
    
    def ScalerStandardization(self) :
        print("Scaling Standardization:")
        # valores con media en 0 y su desviación
        i = 0
        self.data_scaler = [row[:] for row in self.data]
        for line in self.data_scaler :
            if (i > 0) :
                line[0] = (float(line[0]) - self.mean_x) / (self.std_dev_x)
                line[1] = (float(line[1]) - self.mean_y) / (self.std_dev_y)
            i += 1

    def plot_standardized(self, title = "None") :
        tmp_val = self.data_scaler.copy()
        tmp_val.pop(0)
        # Convert to numeric if needed
        tmp_val = [(float(x), float(y)) for x, y in tmp_val]
        # Separate into X and Y lists
        x_vals = [row[0] for row in tmp_val]
        y_vals = [row[1] for row in tmp_val]
        plt.title('Standardized values: '+ title)
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.plot(x_vals, y_vals, 'ro')
        plt.grid(True)
        plt.show()
    
    def plot_predict_standardized(self, title = "None") :
        title = ""
        if (self.scaler  == 1):
            title =  "MinMax" 
        else:
            title = "Standardized"
        tmp_val = self.data_scaler.copy()
        tmp_val.pop(0)
        # Convert to numeric if needed
        tmp_val = [(float(x), float(y)) for x, y in tmp_val]
        # Separate into X and Y lists
        x_vals = [row[0] for row in tmp_val]
        y_vals = [row[1] for row in tmp_val]
        # Prepare predicted values in the same X order
        plot_val = [x_vals, [self.predict_tmp(x) for x in x_vals]]

        #tmp_val = list(zip(*tmp_val))
        #tmp_val = [list(tmp_val[0]), list(tmp_val[1])]
        #plot_val = [[], []]
        #for i in tmp_val[0] :
        #    plot_val[0].append(i)
        #for i in tmp_val[0] :
        #    plot_val[1].append(self.predict_tmp(i))
        plt.title('Standardized values: '+ title)
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.plot(x_vals, y_vals, 'ro')
        plt.plot(plot_val[0], plot_val[1], color="#1C7B9B", linestyle='-')
        #plt.plot(tmp_val[0], tmp_val[1], 'ro')
        #plt.plot(plot_val[0], plot_val[1], color="#1C7B9B", linestyle='-')
        plt.grid(True)
        plt.show()

    def plot(self, title = "None") :
        tmp_val = self.data.copy()
        tmp_val.pop(0)
        # Convert to numeric if needed
        tmp_val = [(float(x), float(y)) for x, y in tmp_val]
        # Sort by X value
        tmp_val.sort(key=lambda row: row[0])
        # Separate into X and Y lists
        x_vals = [row[0] for row in tmp_val]
        y_vals = [row[1] for row in tmp_val]
        # Prepare predicted values in the same X order
        plot_val = [x_vals, [self.predict(x) for x in x_vals]]

        plt.title('Real values: '+title)
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.plot(x_vals, y_vals, 'ro')
        plt.plot(plot_val[0], plot_val[1], color="#1C7B9B", linestyle='-')
        plt.grid(True)
        plt.show()

    def plot_loss(self):
        plt.title('loss function')
        plt.xlabel('Epochs')
        plt.ylabel(self.loss_function)
        plt.plot(self.loss_acc, color="#1C7B9B", linestyle='-')
        plt.grid(True)
        plt.show()

    def save_model(self,filename) :
        print("Saving Model:")
        mae, mse, rmse, r2 = self.errors()
        scaler_type = ""
        if (self.scaler  == 1):
            scaler_type =  "MinMax" 
        else:
            scaler_type = "Standardized"
        print("theta_0: "+str(self.theta_0)+",theta_1: "+str(self.theta_1)+",scaler: "+scaler_type+",loss_function: "+self.loss_function)
        f = open(filename+".csv", "w+")
        f.write("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" %("BIAS", "WEIGHT", "MAE", "MSE", "RMSE", "R2", "SCALER", "LOSS_FUNCTION", "OBJECTIVE", "OBJECTIVE_VALUE", "LEARNING_RATE"))
        f.write("%f, %f, %s, %s, %s, %s, %s, %s, %s, %s, %s" %(self.theta_0, self.theta_1, mae, mse, rmse, r2, scaler_type, self.loss_function, self.target, self.flags["target_value"], self.learning_rate))
        f.close()
    
    def load_model(self, filename) :
        try:
            with open(filename, 'r') as csv_file:
                try:
                    dict_val = csv.reader(csv_file, delimiter = ",")
                    # Saltar la primera línea (encabezado)
                    next(dict_val, None)
                    for row in dict_val:
                        self.theta_0=float(row[0])
                        self.theta_1=float(row[1])
                except:
                    sys.exit("Error: File {:} cannot be read or have incorrect data".format(csv_file))
        except:
            sys.exit("Error: File {:} not found".format(filename))
    
    def denormalize_coef(self):
        # get min and max of X
        x_col = [row[0] for row in self.data_scaler]
        x_col.pop(0)
        min_x = min(x_col)
        min_x_pred = self.predict_tmp(min_x)
        max_x = max(x_col)
        max_x_pred = self.predict_tmp(max_x)
        #exit()
        if (self.scaler==1):
            min_x = min_x * (self.max_x-self.min_x)+self.min_x
            max_x = max_x * (self.max_x-self.min_x)+self.min_x
            min_x_pred = min_x_pred * (self.max_y-self.min_y)+self.min_y
            max_x_pred = max_x_pred * (self.max_y-self.min_y)+self.min_y            
            self.theta_1 = (max_x_pred - min_x_pred) / (max_x - min_x)
            self.theta_0 = min_x_pred - self.theta_1 *  min_x      
        if (self.scaler==2):
            min_x = min_x * self.std_dev_x + self.mean_x
            max_x =  max_x * self.std_dev_x + self.mean_x
            min_x_pred = min_x_pred* self.std_dev_y + self.mean_y
            max_x_pred = max_x_pred * self.std_dev_y + self.mean_y
            #print(str(min_x)+" - "+str(max_x))
            #print(str(min_x_pred)+" - "+str(max_x_pred))
            #exit()
            self.theta_1 = (max_x_pred - min_x_pred) / (max_x - min_x)
            self.theta_0 = min_x_pred - self.theta_1 *  min_x     
    
    def train_model(self, flags): 
        print("Training Model....")
        self.flags = flags
        self.learning_rate = flags["learning_rate"]
        self.scaler =  flags["scaler"]
        self.loss_function = flags["loss_function"]
        self.loss_acc = []
        self.target = flags["target_function"]
        self.epochs = 100000
        self.delta = 0.00000001
        if (self.target=='NUMBER_EPOCHS'):
            self.epochs = flags["target_value"]
        if (self.target=='DELTA_LOSS'):
            self.delta = flags["target_value"]
              
        # Standarized data (feature scaling)
        self.set_mean_stddev()
        self.set_min_max()
        if (self.scaler  == 1):
            self.ScalerMinMax()
        if (self.scaler  == 2):
            self.ScalerStandardization()
        
        epoch  = 0 
        # you can set the number of epochs or check the delta mse before and current
        while epoch < self.epochs :
            self.theta_0 = self.tmp_theta_0
            self.theta_1 = self.tmp_theta_1
            self.tmp_theta_0 -= self.learning_rate * self.get_gradientBiass()
            self.tmp_theta_1 -= self.learning_rate * self.get_gradientWeight()     
            mae, mse, rmse, r2 = self.errors()
            if (self.loss_function=="MAE"):
                self.loss_acc.append(mae)
            if (self.loss_function=="MSE"):
                self.loss_acc.append(mse)
            #self.prev_mse = self.cur_mse
            #self.cur_mse = self.mean_square_error()
            #self.loss_acc.append(self.cur_mse)
            if (flags["print_error"] == 1) :
                print(
                    "Epoch:{}\t mae:{}\t mse:{} \t rmse:{} \t r2:{}".format(
                        epoch,mae,mse, rmse, r2
                    )
                )
            if (epoch>0):
                delta_loss = self.loss_acc[-1] - self.loss_acc[-2]
                if abs(delta_loss)<self.delta and self.target=='DELTA_LOSS':
                    print("number of epochs", epoch)
                    break
            epoch = epoch + 1
        
        self.denormalize_coef()
        
        
    def predict_tmp(self, value) :
        if (self.tmp_theta_0==0.0 and self.tmp_theta_0==0.0):
            sys.exit("Error: No model load.")
        value0 = float(value) - 0
        return ((self.tmp_theta_0 + (self.tmp_theta_1 * float(value0))))
    
    def predict(self, value) :
        if (self.theta_0==0.0 and self.theta_0==0.0):
            sys.exit("Error: No model load.")
        value0 = float(value) - 0
        return ((self.theta_0 + (self.theta_1 * float(value0) )))
        