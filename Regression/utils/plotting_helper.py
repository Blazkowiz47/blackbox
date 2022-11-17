import matplotlib.pyplot as plt

def scatter_plot (X,Y,c = 'black',cmap = None, figsize = (2,2),marker = ".",xlabel = "Label Not Given", ylabel = "Label Not Given", title = "Title Not Given" , s = 2,vmin = None , vmax = None, 
titlefontsize = 17, labelfontsize = 15, show_plot = False, label = "Label Not Given"):
    plt.scatter(X,Y,c = c,cmap = cmap, s = s,marker = marker, label = label)
    plt.xlabel(xlabel,fontsize = labelfontsize)
    plt.ylabel(ylabel,fontsize = labelfontsize)
    plt.title(title,fontsize = titlefontsize)
    if show_plot :
        plt.show()

