#
# @file plot.py
# @author Ozgur Taylan Turan
#
# Main Plotting Interface for mlcxx results
#


# General Stuff
import os
import argparse
# Plotting Stuff
from read import * 
from plotter import * 
# Define your plot cycle color

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", nargs='+')
parser.add_argument("-o", "--order", default="row")
parser.add_argument("-t", "--type", default="llc")
parser.add_argument("-di", "--dinput", default=1)
parser.add_argument("-do", "--doutput", default=1)
parser.add_argument("-l", "--legend", default=str(False))
parser.add_argument("-s", "--save", default='.tex')
parser.add_argument("-sn", "--savename", default='plot')
parser.add_argument("-v", "--visualize", default=bool(True))
parser.add_argument("-sz", "--size", default=(8,6))
parser.add_argument("-xax", "--xaxis", default="normal")
parser.add_argument("-yax", "--yaxis", default="normal")

args = parser.parse_args()




if args.type == "llc":
    assert (len(args.filename)<2), "Need only one file for the learning curve!"
    filename, filext = os.path.splitext(args.filename[0])
    if  filext == ".csv":
        data = load_csv(args.filename[0], args.order)
        fig = plt.figure(figsize=args.size)
        ax = fig.add_subplot(111) 
        lcurve_llc(ax,data,False,False)

    if  filext == "":
        paths = [ os.path.join(dp, f) for dp, dn, filenames in \
                 os.walk(filename) for f in filenames \
                 if os.path.splitext(f)[1] == '.csv' ]
        datas = list()
        for path in paths:
            datas.append(load_csv(path, args.order))
        fig = plt.figure(figsize=args.size)
        ax = fig.add_subplot(111) 
        [lcurve_llc(ax,data,dots=False,mean=False) for data in datas]
    plt.xlabel('N')
    plt.ylabel('Error')

if args.type == "llc-hist":
    assert (len(args.filename)<2), "Need only one file for the learning curve!"
    filename, filext = os.path.splitext(args.filename[0])
    if  filext == ".csv":
        data = load_csv(args.filename[0], args.order)
        fig = plt.figure(figsize=args.size)
        for i in range(50,data.shape[1]):
            plt.hist(data[i,1:])
            plt.show()

if args.type == "learning":
    assert (len(args.filename)<2), "Need only one file for the learning curve!"
    filename, filext = os.path.splitext(args.filename[0])
    if  filext == ".csv":
        data = load_csv(args.filename[0], args.order)
        fig = plt.figure(figsize=args.size)
        ax = fig.add_subplot(111) 
        lcurve(ax,data)

    if  filext == "":
        file_ids = [ f for f in os.listdir(filename)\
                     if os.path.isfile(os.path.join(filename,f)) ]
        datas = [load_csv(os.path.join(filename,file_id), args.order)\
                for file_id in file_ids]
        fig = plt.figure(figsize=args.size)
        ax = fig.add_subplot(111) 
        [lcurve(ax,data,error=False,dots=False) for data in datas]
        plt.xlabel('N')
        #plt.ylim([0,20])
        plt.ylabel('Error')

if args.type == "prediction":
    datas = []
    file_info = [os.path.splitext(name) for name in args.filename]
    for i, file_ in enumerate(file_info):
        if file_[1] == ".csv":
            datas.append(load_csv(args.filename[i],args.order))

    trainset = datas[0]
    predset = datas[1]
    fig = plt.figure(figsize=args.size)
    ax = fig.add_subplot(111) 
    pred(ax,predset, args.dinput, args.doutput)
    data(ax,trainset, args.dinput, args.doutput, "Train")
    d = np.linspace(-3,3,1000)
    ax.plot(d, np.sin(d))


    if len(args.filename) == 3:
        testset = datas[2]
        data(ax,testset, args.dinput, args.doutput, "Test")

if args.type == "xys":
    filename, filext = os.path.splitext(args.filename[0])
    if  filext == ".csv":
        data = load_csv(args.filename[0], args.order)
    fig = plt.figure(figsize=args.size)
    ax = fig.add_subplot(111) 
    ax.plot(data[:,0], data[:,1:int(args.doutput)+1])

if args.type == "xsys":
    filename, filext = os.path.splitext(args.filename[0])
    if  filext == ".csv":
        data = load_csv(args.filename[0], args.order)
    fig = plt.figure(figsize=args.size)
    ax = fig.add_subplot(111) 
    din = int(args.dinput)
    dout = int(args.doutput)
    [ax.plot(data[:,i], data[:,i+din]) for i in range(din+dout-1)]

if args.type == "xsys-scatter":
    filename, filext = os.path.splitext(args.filename[0])
    if  filext == ".csv":
        data = load_csv(args.filename[0], args.order)
    fig = plt.figure(figsize=args.size)
    ax = fig.add_subplot(111) 
    din = int(args.dinput)
    dout = int(args.doutput)
    [ax.scatter(data[:,i], data[:,i+din]) for i in range(din+dout-1)]

if args.legend == str(True):
    plt.legend(frameon=False)

if bool(args.visualize):
    if args.xaxis == 'log':
        plt.xscale('log')
    if args.yaxis == 'log':
        plt.yscale('log')
    plt.show()
else:
    if args.save == ".pdf":
        plt.savefig(args.savename+args.save)
    elif args.save == ".tex":
        tikzplotlib.save(args.savename+'.tex')


