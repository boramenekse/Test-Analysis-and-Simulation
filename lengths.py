import matplotlib.pyplot as plt
lengthslist = [
0.035
,0.035625
,0.035625
,0.036041667
,0.036041667
,0.036875
,0.037291667
,0.037291667
,0.037708333
,0.03875
,0.039375
,0.039375
,0.039583333
,0.040208333
,0.039166667
,0.039583333
,0.039583333
,0.04
,0.04
,0.04
,0.04
,0.040208333
,0.040208333
,0.040625
,0.040833333
,0.040833333
,0.040833333
,0.040833333
,0.042083333
,0.042291667
,0.042291667
,0.046041667
,0.046666667
,0.05125
,0.051458333
,0.055
,0.056041667
,0.056666667
,0.0575
,0.060833333
,0.061875
,0.065208333
,0.065208333
,0.067083333
,0.069791667
,0.069791667
,0.069791667
,0.073541667
,0.07375
,0.073958333
,0.075208333
,0.075416667
,0.076041667
,0.076041667
,0.077083333
,0.07875
,0.079583333
,0.079583333
,0.080416667
,0.080416667
,0.081875
,0.081875
,0.084791667
,0.085416667
,0.085416667
,0.085625
,0.088958333
,0.090416667
,0.090416667
,0.090625
,0.091041667
,0.091458333
,0.091458333
,0.091875
,0.094375
,0.094583333
,0.094583333
,0.094791667
,0.095625
,0.0975
,0.0975
,0.097916667
,0.099791667
,0.100416667
,0.101458333
,0.1025
,0.103541667
,0.103958333
,0.103958333
,0.105416667
,0.106875
,0.108125
,0.108541667
,0.109791667
,0.110416667
,0.110625
,0.110625
,0.112083333
,0.112291667
,0.112291667
,0.112291667
,0.113125
,0.113541667
,0.113541667
,0.113541667
,0.11375
,0.113958333
,0.113958333
,0.113958333
,0.113958333
,0.114166667
,0.114166667
,0.114166667
,0.114166667
,0.114375
,0.114375
,0.114375
,0.114791667
,0.114791667
,0.115
,0.115
,0.115416667
,0.115625
,0.115625
,0.115625
,0.115833333
,0.116875
,0.119583333
,0.123958333
,0.126041667
,0.12625
,0.127291667
,0.127916667
,0.128333333
,0.129375
,0.129583333
,0.130416667
,0.130416667
,0.130625
,0.130625
,0.13125
,0.131458333
,0.131666667
,0.131666667
,0.131875
,0.131875
,0.131875
,0.132291667
,0.132291667
,0.132291667
,0.1325
,0.132708333
,0.132708333
,0.132708333
]
print(len(lengthslist))
fileslist=[*range(154)]
plt.figure()
plt.plot(fileslist, lengthslist)
plt.show()



