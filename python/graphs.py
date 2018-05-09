import matplotlib.pyplot
width = 0.333333
centers = [0.166667,0.5,0.833333]
percents = [0.4,0,0.6]
accuracies = [0,0,1]
matplotlib.pyplot.bar(centers,percents,width)
matplotlib.pyplot.xlabel('confidence')
matplotlib.pyplot.ylabel('% samples')
matplotlib.pyplot.savefig('conf_repartition.pdf')
matplotlib.pyplot.clf()
matplotlib.pyplot.bar(centers,accuracies,width)
matplotlib.pyplot.xlabel('confidence')
matplotlib.pyplot.ylabel('accuracy')
matplotlib.pyplot.savefig('conf_accuracy.pdf')
