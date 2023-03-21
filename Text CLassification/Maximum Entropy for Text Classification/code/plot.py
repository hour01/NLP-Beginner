import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_loss(loss, name = 'loss', path = './'):
    x = list(range(1,1+len(loss)))
    plt.figure()
    plt.plot(x,loss,'bo-',label = name)
    plt.ylabel(name)
    plt.xlabel('iter_num')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(path+name+'.jpg')

def plot_accuracy(accuracy, name = 'accuracy', path = './'):
    x = list(range(1,1+len(accuracy)))
    plt.figure()
    plt.plot(x,accuracy,'bo-',label = name)
    plt.ylabel(name)
    plt.xlabel('iter_num')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(path+name+'.jpg')

def plot_accuracy_2(accuracy1,accuracy2,name1,name2,path = './'):
    x = list(range(1,1+len(accuracy1)))
    plt.figure()
    plt.plot(x,accuracy1,'bo-',label = name1)
    plt.plot(x,accuracy2,'ro-',label = name2)
    plt.ylabel('accuracy')
    plt.xlabel('iter_num')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(path+name1+'_'+name2+'.jpg')

if __name__ == '__main__':
    plot_accuracy(list(range(10)))