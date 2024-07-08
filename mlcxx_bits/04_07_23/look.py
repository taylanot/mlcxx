import numpy as np
import matplotlib.pyplot as plt

if False:
    root = "comp_class_overlap/"
    models = ["ldc","logit","nnc","qdc"]
    ns = np.genfromtxt(root+"ns.csv",delimiter=",")
    for model in models:
        data = np.genfromtxt(root+model+".csv",delimiter=",")
        plt.plot(ns,np.mean(data,0),label=model)
    plt.legend()
    plt.ylabel("error rate")
    plt.xlabel("sample size")
    plt.show()


if True:
    root = "comp_class_overlap/"
    models = ["ldc","logit","nnc","qdc"]
    alphas = [1.,0.8,0.6,0.4,0.2]
    Ns = [0, 50, 99]
    fig,axs = plt.subplots(2,len(Ns),sharey=True)
    for j,N in enumerate(Ns):
        for i,model in enumerate(models):
            data = np.genfromtxt(root+model+".csv",delimiter=",")
            axs[0,j].hist(data[:,Ns[j]],bins=100,label=model,alpha=alphas[i])

    root = "05_07_23/comp_class_overlap/"
    for j,N in enumerate(Ns):
        for i,model in enumerate(models):
            data = np.genfromtxt(root+model+".csv",delimiter=",")
            axs[1,j].hist(data[:,Ns[j]],bins=100,label=model,alpha=alphas[i])
    # Set shared labels
    fig.text(0.5, 0.04, 'error rate', ha='center')
    fig.text(0.04, 0.5, 'frequency', va='center', rotation='vertical')
    fig.text(0.5, 0.93, 'NotStratified', ha='center', fontsize=16)
    # Add a custom title for the second row
    fig.text(0.5, 0.47, 'Stratified', ha='center', fontsize=16)
    plt.legend()
    plt.show()

if False:
    root = "kaczmarz_pinv/"
    models = ["kaczmarz","pinv"]
    ns = [1,100,200,500,600,700,800,1000];
    for model in models:
        data = np.genfromtxt(root+model+".csv",delimiter=",")
        plt.plot(ns,np.mean(data,1),label=model)
        # if model == "kaczmarz":
        #     plt.plot(ns,data[:,:],label=model,color="k")
        # else:
        #     plt.plot(ns,data[:,:],label=model,color="b")
            
    plt.legend()
    plt.ylabel("mse")
    plt.xlabel("sample size")
    plt.show()



