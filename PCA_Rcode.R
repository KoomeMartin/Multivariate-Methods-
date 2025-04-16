#=======================================
#EXAMPLE 1
#FOOD SCIENCE EXAMPLE
#========================================
banana<-read.table('banana.rda',header=TRUE, sep=",")
banana<-load('banana.rda')

# save(banana,file="banana.rda")
#load the rda file
load(file = "banana.rda")
load("banana.rda")
head(banana, 4)

#ls()

#write.csv(banana, file = "banana.csv", row.names = FALSE)




banana$Group=as.factor(banana$Group)
View(banana)

#compute variances
Var_banana<-round(sapply(banana[,2:14], var), digits=5)
#Some variances  are much larger than others hence the need for scaling
barplot(Var_banana, main='Variances')
#View the scatter plots and correlation matrix
library(psych)
#construct pairwise scatter plot for first 4 variables
pairs(banana[,2:4],col = c(rep ('gray',12), rep ('darkkhaki',12),
                                rep ('orange',12), rep ('blue',12)))
hist(banana$TSS)
#Get scatter plots, histograms 
#and correlations in one plot
pairs.panels(banana[,2:5],pch=21, bg=c("red","green","blue","grey")[banana$Group],
             smooth=FALSE, main="Scatterplot Matrix",ellipses=FALSE)
#compute correlation matrix
r1<-corr.test(banana[,2:14])$r
r1
#Visualize correlation maxtrix
library(corrplot)
corrplot.mixed(round(r1,1),lower.col = "black", tl.cex = .8)
#Observe that there are some high correlations
#conduct PCA
pca<-princomp(banana[2:14], cor=TRUE)
summary(pca)
#Draw Scree plot to determine how many components to keep.
screeplot(pca, npcs=13, type ="barplot", xaxt="n", yaxt = "n", main="Scree Plot")

#Obtain proportion of variance explained by the PCs "manually" 
#and plot the cumulative variance
#extract the standard deviations of the principal components
svalues<-pca$sdev
#compute the eigen values
# These are variances of the PCs
PC.Variances<-svalues^2
#compute cummulative proportion of variance 
Prop.Variance.PCs<-cumsum(PC.Variances)/sum(PC.Variances)
Prop.Variance.PCs
#plot the cummulative proportion of variance
par(mfrow=c(1,1))
barplot(Prop.Variance.PCs, main="cummulative proportion of variance",las=2)
plot(Prop.Variance.PCs, type="b",main="cummulative proportion of variance")
dev.off()
#display loadings (the matrix V)
Load_banana<-unclass(pca$loadings)
round (unclass (Load_banana), digits=2)

#display the scores (The matrix T in the slides)
T<-round(pca$scores, digits=2 )
#plot the scores of the first PC
barplot (T[,1], xaxt="n", yaxt="n", ylim=c(-5, 5),
         col = c(rep ('gray',12), rep ('darkkhaki',12),
                 rep ('orange',12), rep ('blue',12)))
#Plot the scores of the first two PCs 
plot(T[,1],T[,2], col = c(rep ('gray',12), rep ('darkkhaki',12),
                        rep ('orange',12), rep ('blue',12)),
                         main='Plot of the first 2 PCs',xlab='PC1',ylab='PC2')
text(-2,2, "Ripe pulp",col="blue")
text(-2,-1, "Ripe peel", col='orange')
text(0,-1,"Green Peel", col="gray")
text(3.3,-0.5,"Green Pulp",col="darkkhaki")
#scatterplot of the scores with legend
#using ggplot
library(ggplot2)
T<-as.data.frame(T)
T$Group<-banana$Group
ggplot(T,aes(x=Comp.1,y=Comp.2,color=Group))+
  geom_point()
  
#Generate some new scaled data
#Scaled.Banana<-scale(banana[,2:14])
newdata<-c(-1.51,-0.68,-0.71,1.12,1.53,0.58,0.23,-0.34,-0.6,-0.6,0.2,-0.45, 0.62)
names(newdata)<-names(banana[,2:14])
newdata
#Extract the first two PCs
P.components<-pca$loadings[,1:2]
#Project the new data onto the first two PCs.
projection<-newdata%*%P.components
#Plot the scores of the first two PCs 
plot(T[,1],T[,2], col = c(rep ('gray',12), rep ('darkkhaki',12),
                          rep ('orange',12), rep ('blue',12)),
     main='Prediction for unidentified flour',xlab='PC1',ylab='PC2')
text(-2,2, "Ripe pulp",col="blue")
text(-2,-1, "Ripe peel", col='orange')
text(0,-1,"Green Peel", col="gray")
text(3.3,-0.5,"Green Pulp",col="darkkhaki")
#add new point to the plot for classification
points(x=projection[1],y=projection[2], col="red",pch=18, cex=2)

#Generate the biplot
biplot(pca,col=c("red","blue"), main="Biplot")

#More PCA work
#if (!require("BiocManager", quietly = TRUE))
#install.packages("BiocManager")
#BiocManager::install(version = "3.15")
#BiocManager::install("pcaMethods")
#========================================================
#EXAMPLE 2 
# POLITICAL SCIENCE
#======================================================

#==============================================================
# The Latin American Public Opinion Project (LAPOP) - coordinated from Vanderbilt
# University - specializes in conducting impact assessment 
# studies and producing reports This data contains 12 questions from the 
#LAPOP survey, carried out from a selection of about
#7000 people in 10 Latin American countries on the attitudes, 
# evaluations and experiences of individuals in Latin American countries.
# This data provides researchers with different questions that, together, could help
# us to approximate how much confidence there is in the region regarding democratic
# institutions in the country corresponding to each individual.
# https://www.vanderbilt.edu/lapop/ab2016/AB2017-v18.0-Spa-170202_W.pdf
#========================================================================================
#========================================================================================
load(file="lapop.rda")
library(tidyverse)
#Get a sample from the data for better visualization
set.seed(1234)
Pop_Sample<-lapop%>%
  group_by(country_name)%>%
  sample_n(size=100)
#get the average of trust_elections by country
lapop2 <- Pop_Sample %>%
  group_by(country_name) %>%
  mutate(trust_elections_prom = mean(trust_elections)) %>%
  ungroup()
#visualize trust in elections by country
ggplot(lapop2, aes(x = trust_elections)) +
  geom_histogram() +
  scale_x_continuous(breaks = 1:7) +
  labs(title = "Trust in elections",
       x = "The national average is expressed as a dashed line.",
       y = "Frequency")+
  geom_vline(aes(xintercept = trust_elections_prom),
             color = "black", linetype = "dashed", size = 1)+
  facet_wrap(~country_name) 
#prepare the data for PCA
#Exclude country variable from the data set to retain only numerical variables
lapop_num <- lapop2 %>%
  dplyr::select(justify_coup, justify_cong_shutdown, trust_institutions,trust_courts, trust_congress, trust_president, trust_parties,
         trust_media, trust_elections, satisfied_dem, vote_opposers)
# scale the variables and omit cases with missing data
lapop_num <-lapop_num %>%
  scale()%>%
  na.omit()
#   devtools::install_github("hadley/devtools")
install.packages("devtools", dependencies=TRUE)
# #Check correlations
install.packages("ggcorrplot")
install.packages("ggcorrplot")
library(ggcorrplot)

#compute pairwise correlations
corr_lapop <- lapop_num %>%
  # calculate correlation matrix and round to 1 decimal place:
  cor(use = "pairwise") %>%
  round(1)
#visualize the correlation
library(corrplot)
corrplot.mixed(corr_lapop,lower.col = "black", tl.cex = .7)
#conduct PCA on the numerical variables
library(stats)
  pca2 <- princomp(lapop_num)
  
  summary(pca2, loadings = TRUE, cutoff = 0.3)
#======================================================  
#Extract the eigenvalues
#======================================================
  # install.packages("factoextra")
library(factoextra)
eig_val <- get_eigenvalue(pca2)
eig_val
#visualise the eigen values
fviz_eig(pca2, addlabels = TRUE, ylim = c(0, 35)) 
#display the actual eigen values on the plot
fviz_eig(pca2, choice = c("eigenvalue"), addlabels = TRUE, ylim = c(0, 4))
#change some plot features
fviz_eig(pca2, choice = c("eigenvalue"), addlabels = TRUE, ylim = c(0, 4),
        barfill = "darkgray", barcolor = "darkgray")

#Plot the biplot
fviz_pca_biplot(pca2, repel = FALSE, col.var ="black", col.ind = "gray")
# explore the different dimensions that will 
# compose the concept we want to measure
fviz_contrib(pca2, choice = "var", axes = 1, top = 10)
fviz_contrib(pca2, choice = "var", axes = 2, top = 10)
fviz_contrib(pca2, choice = "var", axes = 3, top = 10)
fviz_contrib(pca2, choice = "var", axes = 4, top = 10)

# the first component is the most diverse but 
# it is largely fed by confidence variables and represents trust
#=============================================================

#===========================================================
#=============================================================
# The second dimension
# is composed of variables that represent the propensity to
# justify coups d’état or closures of Congress. Our
# concept will then have components that measure how 
# individual opinions of democratic breaks are formed, 
# measuring how fragile formal institutions are within public opinion
# in Latin America.
#==============================================================
# The third component leans towards
#  vote_opposers and represents level of
#political tolerance.
#====================================================
#We can use FactomineR to visualise eigen values
library(FactoMineR)
pca_1 <- PCA(lapop_num, graph = FALSE)
fviz_eig(pca_1, choice = "eigenvalue", addlabels = TRUE, ylim = c(0, 3.5))
eig <- get_eig(pca_1)
# add up these four components,
# but weighting each one by the percentage of the variance 
# they represent. We do it the following way:
#Construct an index as a democracy indicator
data_pca <- pca_1$ind$coord%>%
  as_tibble() %>%
  mutate(pca_01 = (Dim.1 * 28.7 + Dim.2 * 12.3 + Dim.3 * 10.3 +
                     Dim.4 * 7.9) / 60)
lapop2 <- bind_cols(lapop2, data_pca %>% dplyr::select(pca_01))
#Rescale the index to 100
#install.packages('GGally')
library(GGally)
lapop2 <- lapop2 %>%
  mutate(democracy_index = GGally::rescale01(pca_01) * 100)%>%
  dplyr::select(democracy_index, everything())

#Plot the density plot for the index
index_density <- ggplot(data = lapop2,
                        mapping = aes(x = democracy_index)) +
  labs(x = "Index of trust in democracy", y = "Density") +
  geom_density()
index_density
#density plot of index by country
lapop2 <- lapop2 %>%
  group_by(country_name) %>%
  mutate(democracy_avg = mean(democracy_index)) %>%
  ungroup()
ggplot(lapop2, aes(x = democracy_index)) +
geom_density() +
  labs(title = "Trust in democracy in Latin America (N = 7000)",
       x = "Trust in democracy",
       y = "Density") +
  facet_wrap(~country_name) +
  geom_vline(aes(xintercept = democracy_avg),
             color = "black", linetype = "dashed", size = 1)
#Extract the scores and plot for the first 2 PCs
#and store them in the variable DemoTrust
DemoTrust<-pca2$scores
head(DemoTrust)
dev.off()
plot(DemoTrust[,1],DemoTrust[,2])

#visualise the loadings on a bar plot.
Demoloadings<-pca2$loadings
op<-par(mfrow=c(1,3),mar=c(15,4,4,2))
barplot(Demoloadings[,1],las=2,main="PC1")
barplot(Demoloadings[,2],las=2,main="PC2")
barplot(Demoloadings[,3],las=2,main="PC3")
par(op)

#Find  clusters with Kmeans
X<-scale(lapop_num)
cluster.lapop<-kmeans(X,centers=2)
DemTrust<-as.data.frame(DemoTrust)
ggplot(DemTrust, aes(DemTrust[,1],DemTrust[,2], color=factor(cluster.lapop$cluster)))+
  geom_point()
#==========================================================
#Cluster using  Linear Discriminant Analysis
#to find the clustering imposed by PC2
#================================================
#construct a variable for the clusters
DemClusters<-rep(1,nrow(DemTrust))
t1<-DemTrust[,2]>1.5
t3<-(DemTrust[,2]>0 & DemTrust[,2]<1.5)
DemClusters[t3]<-2
t2<-DemTrust[,2]<0
DemClusters[t2]<-3
DemClusters<-factor(DemClusters)
#add a new variable to X to store the cluster information
X2<-cbind(DemClusters,X)
X2<-as.data.frame(X2)

#Use linear discriminant analysis to classify the
#data
library(MASS)
linear <-lda(X2$DemClusters~., X2)
p <- predict(linear, X2)
tab2<-table(predicted=p$class,observed=X2$DemClusters)
#create confusion matrix
tab2
#check accuracy of the predictions
sum(diag(tab2))/sum(tab2)
#Plot histograms 
ldahist(data = p$x[,1], g = X2$DemClusters)
ggplot(DemTrust, aes(DemTrust[,1],DemTrust[,2], color=p$class))+
  geom_point()+
  labs(x="PC1",y="PC2",title="PC1 vs PC2 for political data", color="attitude")
#facet_wrap(~Pop_Sample$country_name)

#Check to see if LDA can find the Kmeans clusters 
Km.Cluster<-cluster.lapop$cluster
X3<-cbind(Km.Cluster,X)
X3<-as.data.frame(X3)
linear <-lda(X3$Km.Cluster~., X3)
p2 <- predict(linear, X3)
table(p2$class,X3$Km.Cluster)
ldahist(data = p2$x[,1], g = X3$Km.Cluster)
ggplot(DemTrust, aes(DemTrust[,1],DemTrust[,2], color=p2$class))+
  geom_point()
#We can study interaction between the two classifications

tab<-table(LDA=p$class,Kmeans=p2$class)
tab
library(vcd)
tile(tab)

#==================================================================
#EXAMPLE 3 

#Diagnosis of cancerous tumour
#Features are computed from a digitized 
# image of a fine needle aspirate (FNA) of a breast mass.
# They describe characteristics of the cell nuclei present in the image
#https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
#========================================================================
library(readr)
data1 <- read_csv("data.csv")
#convert diagnosis to a factor
data1$diagnosis<-as.factor(data1$diagnosis)
#Remove the unused variables
data2=data1[,-c(1,33)]
str(data2)
#View(data2)
# install.packages('psych')
library(psych)
pairs.panels(data2[,2:8],
             gap = 0,
             bg = c("red", "green")[data2$diagnosis],
             pch = 21)
#Box plot for radius_mean
boxplot(data2$radius_mean~data2$diagnosis, main="Radius Mean")

library(stats)
library(tidyverse)
#Scale data and omit cases with missing values
data3<-data2[,-1]%>%
  scale()#%>%
  #na.omit()
#Run PCA
library(stats)
pca3 <- princomp(data3,cor = TRUE)
summary(pca3, loadings = TRUE, cutoff = 0.3)
library(factoextra)
#visualise the eigen values on the scree plot
fviz_eig(pca3, addlabels = TRUE, ylim = c(0, 50)) 
#display the actual eigen values on the plot
fviz_eig(pca3, choice = c("eigenvalue"), addlabels = TRUE, ylim = c(0,14))
#change some plot features
fviz_eig(pca3, choice = c("eigenvalue"), addlabels = TRUE, ylim = c(0, 14),
         barfill = "darkgray", barcolor = "darkgray")
#extract loadings
Canc_loadings<-pca3$loadings
Canc_loadings
#Extract scores
Canc_Scores<-pca3$scores
#create a logical variable for identifying
#the tumour type in the plots
tumour<-data2$diagnosis=="B"
tumour=as.numeric(tumour)+1
plot(Canc_Scores[,1],Canc_Scores[,2],col=tumour,xlab="PC1",
     ylab="PC2" ,main="PC1 vs PC2 for Cancer Data")
text(0.2,-10 ,"Benign" ,col="red")
text(13,5,"Malignant")

#===========================================
#EXAMPLE 4 Outlier Detection
#Applied to images
#======================================================

library(jpeg)
t <- tempfile()
download.file('http://bit.ly/nasa-img', t)
#img <- readJPEG("fracture.jpg")
img <- readJPEG(t)
#Display the image
plot(1:10, ty='n',axes=FALSE)
rasterImage(img, 1,1,6,10)
str(img)
h <- dim(img)[1]
w <- dim(img)[2]

#Reshape each layer into a vector
m <- matrix(img, h*w)
str(m)
pca <- prcomp(m)
summary(pca)
#create a helper function to extract colors
extractColors <- function(x)
  rgb(x[1], x[2], x[3])
(colors <- apply(abs(pca$rotation), 2, extractColors))
#Plot the colours in the image
pie(pca$sdev, col = colors, labels = colors)

#Reshape the rows back to matrices and display the Scores.
par(mfrow = c(1, 2), mar = rep(0, 4))
image(matrix(pca$x[, 1], h), col = gray.colors(100))
image(matrix(pca$x[, 2], h), col = gray.colors(100), yaxt = 'n')
#More PCA

#========================================================================
#EXAMPLE 5
#Geochemical Example
#========================================================================
# Here, we will use the Nashville carbonates geochemistry 
# data set, a set of geochemical measurements on Upper 
# Ordovician limestone beds from central Tennessee (Theiling et al. 2007).
# The first column contains the stratigraphic position, 
# which we’ll use as an external variable to interpret our ordination

#from http://strata.uga.edu/8370/lecturenotes/principalComponents.html
nashville <-read.table('NashvilleCarbonates.csv', header=TRUE, row.names=1, sep=',')
View(nashville)
#row.names(nashville)<-NULL 
#Pick the numerical variables
geochem <- nashville[ , 2:9]
#log transform the percentages
geochem[ , 3:8] <- log10(geochem[ , 3:8])
#view(geochem)
Var.geo<-sapply(nashville[,2:9],var)
barplot(Var.geo)
#Run the PCA
pca5<-prcomp(geochem, scale. = TRUE)
#Extract the scores
scores.geo<-pca5$x
#view the scores
head(scores.geo)
#Draw the biplot
biplot(pca5)
#Plot the first two PCs
plot(scores.geo[,1],scores.geo[,2 ])

#Split stratposition variable into two categories
#carters formation
Carters <- nashville$StratPosition < 34.2
#Hermitage formation
Hermitage <- nashville$StratPosition >= 34.2

#Build the plot by adding points successively from the two categories
plot(scores.geo[,1],scores.geo[,2], type='n',
     main="PC2 vs PC1 for geochemical data",xlab="PC1",ylab="PC2")
points(scores.geo[Carters, 1], scores.geo[Carters, 2], pch=16, cex=0.7, 
       col='blue')
points(scores.geo[Hermitage, 1], scores.geo[Hermitage, 2], 
       pch=16, cex=0.7, col='red', )

text(1, 3, 'Carters', col='blue')
text(-1, -4, 'Hermitage', col='red')

#some further classification based on the loadings
library(stats)
pca6 <- princomp(scale(geochem))
#Show high loadings
summary(pca6, loadings = TRUE, cutoff = 0.3)

text(-1, -5, 'non-dolomitized', pos=3, col='black')
text(-1, 3, 'dolomitized', pos=3, col='black')
text(2, -1.8, 'clean limestone', pos=3, col='black')
text(-4.5, -1.8, 'clay-rich limestone', pos=3, col='black')

# Axis 1 has a strong positive loading for calcium, 
# and strong negative loadings for silicon, iron, and aluminum. 
# As these are analyses of limestones, this likely reflects 
# relatively pure limestone at positive axis 1 scores and increasing 
# clay content at negative axis 1 scores
# Axis 2 has strong positive loadings for d18O and magnesium,
# which may reflect dolomitization.

