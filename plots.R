###Method 1 (NetMHC)
netMHC <- read.csv("hivprotein.csv",header = TRUE, sep=";")

names=c("Rev", "Vpu", "Vpr", "Tat", "Pol", "Vif", "Gag", "Env", "Nef")

par(mfrow = c(1,1))

for(i in c(1:length(names))){
  affin <- subset(netMHC$affin, netMHC$Protein == names[i])
  pos <- subset(netMHC$Position, netMHC$Protein == names[i])
  print(pos)
  x=which(affin<500)
  #print(x)
  if(length(x)>0) {
    plot(x, rep(1,length(x)), type = 'p', pch = "*", main = names[i], xlab = "Position")
  }
}

pos <- subset(netMHC$Position, netMHC$Protein == names[i])

#Only one protein
#Subsetting values. We want values for the Gag protein.
#affin <- subset(Data$affin, Data$Protein == "Gag")
#pos <- subset(Data$Position, Data$Protein == "Gag")

#Transforming values. <500 = 1 = epitope and >500 = 0 = no epitope
#affin[affin<500]=1
#affin[affin>=500]=0

#Plot of epitopes for Gag protein
#plot(pos, affin,type = 'p',pch="-", main=paste("Gag protein binding affinity"))



############


#Doing the same thing for all proteins
Position <- netMHC$Position
Position
Binding_affinity <- netMHC$affin

Binding_affinity[Binding_affinity<500]=1
Binding_affinity[Binding_affinity>=500]=0

sum(Binding_affinity[Binding_affinity<500])


plot(Binding_affinity, type = 'p', pch="-", xaxt="n", 
     main=paste("Protein binding affinity"))

names=c("Rev","Vpu","Vpr","Tat","Pol","Vif","Gag","Env","Nef")

axis(1, at = which(Position==0), labels = FALSE)
text(which(Position==0), par("usr")[3]-0.2, labels = names,srt = 90, pos = 1, xpd = TRUE)

###Method 2 (SSM)

## Read data
SMMData <- read.table("smm hivprotein.csv", sep=",", header=TRUE)

## Converts numbers to protein names

names=c("Gag","Pol","Vif","Vpr","Tat","Rev","Vpu","Env","Nef")
SMMData$seq_num = names[SMMData$seq_num]

## We set a limit for how many epitopes we want

numOfEpitopes <- 89

limit <- sort(SMMData$rank)[numOfEpitopes]

epitopes <- SMMData[SMMData$rank <= limit,]


## We want to plot these 89 epitopes
par(mfrow=c(3,3))

for(i in c(1:length(names))){
  x <- epitopes$start[epitopes$seq_num == names[i]]
  if(length(x)>0) {
    plot(x, rep(1,length(x)), type = 'p', pch = "*", col = i, main = names[i], xlab = "Position")
  }
}

## Plot all proteins

epitopes = epitopes[order(epitopes$start),]
epitopes = epitopes[order(epitopes$seq_num),]

position = epitopes$start


plot(position, type = 'p', pch = "-", xaxt = "n", 
     main=paste("Protein binding affinity"))
  
axis(1, at = which(Position == 0), labels = FALSE)
text(which(Position==0), par("usr")[3]-0.2, labels = names,srt = 90, pos = 1, xpd = TRUE)




