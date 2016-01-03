# remember to set working directory
setwd("/Users/christian/Documents/OneDrive/Fagprojekt/fagprojekt-kode")

machine = read.csv("data/machine.csv", header = TRUE, sep=";")
netmhc = read.csv("data/netmhc.csv", header = TRUE, sep=";")
smmpmbec = read.csv("data/smmpmbec.csv", header = TRUE, sep=";", stringsAsFactors = FALSE)
syfpeithi = read.csv("data/syfpeithi.csv", header = TRUE, sep=";")
experiments = read.csv("data/hivEpitopes.csv", header = TRUE, sep=";")
# not needed since the same is in smmpmbec and syfpeithi
# also, for some reason this is shorter than the other. Don't use it
#hivorf = read.csv("data/hivorf.txt", header = FALSE, sep="\n")

methodNames = c("Experimental", "Machine", "NetMHC", "SMMPMBEC", "SYFPEITHI")
numOfMethods = length(methodNames)

protNames = c("gag", "pol", "vif", "vpr", "tat", "rev", "vpu", "env", "nef")
numOfProt = length(protNames)
protLengths = rep(0, numOfProt)
for(i in 1:numOfProt) {
  # smmpmbec is just a random file used where alle the orfs are present
  # with connection to their protein
  protLengths[i] = sum(smmpmbec$protein == protNames[i]) + 8
}

# make sure that everything from experimental is actually from the same peptides
experiments = intersect(experiments$sequence, smmpmbec$sequence)
totalExperiments = length(experiments)
totalExperiments
experimentPos = rep(0, totalExperiments)
experimentProt = rep('', totalExperiments)
for(i in 1:totalExperiments) {
  index = which(experiments[i] == smmpmbec$sequence)
  experimentPos[i] = smmpmbec$position[index]
  experimentProt[i] = smmpmbec$protein[index]
}

# remove everything that isn't epitopes
limit = 500
netmhc = netmhc[netmhc$affinity <= limit,]
smmpmbec = smmpmbec[smmpmbec$affinity <= limit,]

# total number of epitopes
totalMachine = dim(machine)[1]
totalMachine
totalNetmhc = dim(netmhc)[1]
totalNetmhc
totalSmmpmbec = dim(smmpmbec)[1]
totalSmmpmbec

avgEpitopes = mean(totalNetmhc, totalSmmpmbec)
scoreLimit = sort(syfpeithi$score, decreasing = TRUE)[avgEpitopes]
syfpeithi = syfpeithi[syfpeithi$score >= scoreLimit,]
totalSyfpeithi = dim(syfpeithi)[1]
totalSyfpeithi


# bar plot of counts of predictions
sumExperiments = rep(0, numOfProt)
sumMachine = rep(0, numOfProt)
sumNetmhc = rep(0, numOfProt)
sumSmmpmbec = rep(0, numOfProt)
sumSyfpeithi = rep(0, numOfProt)

for(prot in 1:numOfProt) {
  sumExperiments[prot] = sum(experimentProt == protNames[prot])
  sumMachine[prot] = sum(machine$protein == protNames[prot])
  sumNetmhc[prot] = sum(netmhc$protein == protNames[prot])
  sumSmmpmbec[prot] = sum(smmpmbec$protein == protNames[prot])
  sumSyfpeithi[prot] = sum(syfpeithi$protein == protNames[prot])
}

# collect data in data frame for plotting
methods = c(rep(methodNames[1], numOfProt),
           rep(methodNames[2], numOfProt),
           rep(methodNames[3], numOfProt),
           rep(methodNames[4], numOfProt),
           rep(methodNames[5], numOfProt))
proteins = ordered(rep(protNames, numOfMethods), levels = protNames)
epitopes = c(sumExperiments, sumMachine, sumNetmhc, sumSmmpmbec, sumSyfpeithi)
all = data.frame(methods, proteins, epitopes)

# plot prediction count
require(lattice)
barchart(epitopes~proteins, data = all, groups = methods,
         origin = 0, auto.key = list(corner = c(0.5, 0.95)))







# find prediction position on x axis of upcoming plot
xExperiments = experimentPos
xMachine = machine$position
xNetmhc = netmhc$position
xSmmpmbec = smmpmbec$position
xSyfpeithi = syfpeithi$position
ticks = c(0, cumsum(protLengths))
# start from 2 since zero is added when dealing with gag
for(i in 2:numOfProt) {
  index = experimentProt == protNames[i]
  xExperiments[index] = xExperiments[index] + ticks[i]
  index = machine$protein == protNames[i]
  xMachine[index] = xMachine[index] + ticks[i]
  index = netmhc$protein == protNames[i]
  xNetmhc[index] = xNetmhc[index] + ticks[i]
  index = smmpmbec$protein == protNames[i]
  xSmmpmbec[index] = xSmmpmbec[index] + ticks[i]
  index = syfpeithi$protein == protNames[i]
  xSyfpeithi[index] = xSyfpeithi[index] + ticks[i]
}


xlimit = c(0, ticks[numOfProt+1])
ylimit = c(0.05,1.15)
plot(NULL, ylab = '', xlab = "Position", xlim = xlimit, ylim = ylimit, type = 'p', yaxt = 'n', xaxt = 'n')

# grid
for(i in 1:length(ticks)) {
  lines(rep(ticks[i],2), c(-10,10), col = "gray")
}
colors = c('#408080', '#804080', '#40804f', '#806140', '#400080')
points(xExperiments, rep(0.5,length(xExperiments)), pch = '|', col = colors[1])
points(xMachine, rep(0.4,length(xMachine)), pch = '|', col = colors[2])
points(xNetmhc, rep(0.3,length(xNetmhc)), pch = '|', col = colors[3])
points(xSmmpmbec, rep(0.2,length(xSmmpmbec)), pch = '|', col = colors[4])
points(xSyfpeithi, rep(0.1,length(xSyfpeithi)), pch = '|', col = colors[5])

axis(side=1, at = ticks, labels = FALSE)
text(ticks, par("usr")[3]-0.1, labels = c(protNames, ''), srt = 90, pos = 1, xpd = TRUE)

legend("top", methodNames, lty = 1, lwd = 10, col = colors)








intersect1 = length(intersect(xMachine, xNetmhc))
intersect2 = length(intersect(xMachine, xExperiments))
intersect3 = length(intersect(xNetmhc, xExperiments))
intersect4 = length(intersect(intersect(xMachine, xNetmhc), xExperiments))

require(venneuler)
venn = venneuler(c(Machine=totalMachine, NetMHC=totalNetmhc, Experimental=totalExperiments,
                "Machine&NetMHC" = intersect1,
                "Machine&Experimental" = intersect2,
                "NetMHC&Experimental" = intersect3,
                "Machine&NetMHC&Experimental" = intersect4))
plot(venn)






# see what we found that the other methods didn't find.

found = setdiff(intersect(xMachine, xExperiments),
        intersect(xNetmhc, xExperiments))
found
prot = experimentProt[found == xExperiments]
prot
posInProt = experimentPos[found == xExperiments]
posInProt
# reload original data
netmhc = read.csv("data/netmhc.csv", header = TRUE, sep=";")
smmpmbec = read.csv("data/smmpmbec.csv", header = TRUE, sep=";", stringsAsFactors = FALSE)
syfpeithi = read.csv("data/syfpeithi.csv", header = TRUE, sep=";")
found = smmpmbec[smmpmbec$position == posInProt,]
found
found = found[found$protein == prot,]
found

# see how well different methods predicted on this
a = netmhc[netmhc$position == posInProt,]
a[a$protein == prot,]$affinity
a = smmpmbec[smmpmbec$position == posInProt,]
a[a$protein == prot,]$affinity
a = syfpeithi[syfpeithi$position == posInProt,]
a[a$protein == prot,]$score





