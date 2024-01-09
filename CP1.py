import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def main():
    ## FUNCTIONS CORRESPOND TO QUESTIONS 1.3 - 1.6
    basic()
    data_plot()
    regression()
    task_1()
    task_2()


## 1.3 BASIC DATA ANALYSIS  
def basic():
    with open("Activity1Data.txt", "r") as infile:
        header = infile.readline()
        data = np.zeros(60)
        i = 0
        for line in infile:
            line = line.strip()
            columns = line.split()
            data[i] = float(columns[1])
            i += 1

        print("|----------1.3 BASIC DATA ANALYSIS------------|")     
        # Numpy Analysis
        m2 = np.mean(data)
        u_m2 = np.sqrt(np.var(data)) / np.sqrt(np.size(data))
        print(f"Using Numpy, we measure:\nx = {m2:.2f} +/- {u_m2:.2f} | variance = {np.var(data):.3f}\n")
        
        # Scipy Analysis
        n,(xmin ,xmax), m3, v, s, k = stats.describe(data)
        u_m3 = np.sqrt(v) / np.sqrt(n)
        print(f"Using Scipy, we measure:\nx = {m3:.2f} +/- {u_m3:.2f} | variance = {v:.3f}")
        print("*We assume the data we observe is sampled from a normal distribution")
        print("|---------------------------------------------|\n")


## 1.4 PLOTTING WITH MATPLOTLIB           
def data_plot():
    #Format:                x, u(x),    y, u(y)
    reg_data = np.array([[1.21, 0.16, 10.2, 2.1],
                         [2.04, 0.10, 15.3, 3.2],
                         [3.57, 0.13, 19.8, 2.6]])
    
    x = reg_data[:,0]
    u_x = reg_data[:,1]
    y = reg_data[:,2]
    u_y = reg_data[:,3]
    
    fig, ax = plt.subplots(1)
    ax.errorbar(x, y, u_y, u_x, label="PHY2004W Data",fmt='s', color="red", ecolor="black")
    
    # Formatting axes
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 30)
    ax.set_title("Comparison of Experimental Data with Theoretical Prediction")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    #fig.savefig("figure1.eps")
    fig.show()


## 1.5 FITTING A MODEL TO DATA
def regression():
    tb = np.loadtxt("LinearNoErrors.txt", skiprows=1)
    N = tb.shape[0]
    x_i = tb[:,0]
    y_i = tb[:,1]

    # Slope and Intercept
    denominator = (N * np.sum(x_i**2) - np.sum(x_i)**2)
    m = (N * np.sum(x_i * y_i) - np.sum(x_i) * np.sum(y_i)) / denominator
    c = (np.sum(x_i**2) * np.sum(y_i) - np.sum(x_i * y_i) * np.sum(x_i)) / denominator

    # Deviations
    d_i = y_i - (m * x_i + c)

    # Uncertainties
    u_m = np.sqrt((np.sum(d_i**2) * (N / (N - 2))) / denominator)
    u_c = np.sqrt((np.sum(d_i**2) * np.sum(x_i**2)) * (N / (N - 2)) / (N * denominator))
    print("|----------1.5 FITTING A MODEL TO DATA------------|")
    print(f"m = {m:.2f} +- {u_m:.2f}\nc = {c:.0f} +/- {u_c:.1f}")
    print("|-------------------------------------------------|\n")


## 1.6 FOR SUBMISSION
def task_1():
    # 1)
    with open("Activity1Data.txt", "r") as infile:
        header = infile.readline()
        data = np.zeros(60)
        i = 0
        for line in infile:
            line = line.strip()
            columns = line.split()
            data[i] = float(columns[1])
            #print(i, data[i])
            i += 1
        print("|-------------------Task 1--------------------|")
        # By definition (Sample Variance)
        m1 = sum(data) / len(data)
        SS = 0
        for i in range(len(data)):
            SS += (data[i] - m1)**2
        var = SS / (len(data) - 1)
        u_m1 = np.sqrt(var) / np.sqrt(len(data))
        print(f"Using the definition, we measure:\nx = {m1:.2f} +/- {u_m1:.2f} | variance = {var:.2f}\n")
                
        # Numpy Analysis
        m2 = np.mean(data)
        u_m2 = np.sqrt(np.var(data)) / np.sqrt(np.size(data))
        print(f"Using Numpy, we measure:\nx = {m2:.2f} +/- {u_m2:.2f} | variance = {np.var(data):.2f}\n")
                
        # Scipy Analysis
        n,(xmin ,xmax), m3, v, s, k = stats.describe(data)
        u_m3 = np.sqrt(v) / np.sqrt(n)
        print(f"Using Scipy, we measure:\nx = {m3:.2f} +/- {u_m3:.2f} | variance = {v:.2f}")
        print("|---------------------------------------------|\n")


def task_2():
    #Format:                x, u(x),    y, u(y)
    reg_data = np.array([[1.21, 0.16, 10.2, 2.1],
                         [2.04, 0.10, 15.3, 3.2],
                         [3.57, 0.13, 19.8, 2.6]])


    x = reg_data[:,0]
    u_x = reg_data[:,1]
    y = reg_data[:,2]
    u_y = reg_data[:,3]
    
    # Best fit
    N = reg_data.shape[0]
    x_i = reg_data[:,0]
    y_i = reg_data[:,2]
    # Slope and Intercept
    denominator = (N * np.sum(x_i**2) - np.sum(x_i)**2)
    m = (N * np.sum(x_i * y_i) - np.sum(x_i) * np.sum(y_i)) / denominator
    c = (np.sum(x_i**2) * np.sum(y_i) - np.sum(x_i * y_i) * np.sum(x_i)) / denominator

    # Deviations
    d_i = y_i - (m * x_i + c)

    # Uncertainties
    u_m = np.sqrt((np.sum(d_i**2) * (N / (N - 2))) / denominator)
    u_c = np.sqrt((np.sum(d_i**2) * np.sum(x_i**2)) * (N / (N - 2)) / (N * denominator))

    # Line Equation f(t) = mt + c
    t = np.linspace(0.5, 4)
    f =  m * t + c

    # Error-bar plot
    fig, ax = plt.subplots(1)
    ax.errorbar(x, y, u_y, u_x, label="PHY2004W Data",fmt='s', color="red", ecolor="black")
    ax.plot(t, f, color="b", label=f"Best Fit\nm = {m:.2f} +/- {u_m:.2f}\nc = {c:.0f}m +/- {u_c:.0f}m")
    
    # Formatting axes
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 30)
    ax.set_title("Comparison of Experimental Data with Theoretical Prediction")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    #fig.savefig("figure2.eps")
    fig.show()

    
if __name__ == "__main__":
    main()



























