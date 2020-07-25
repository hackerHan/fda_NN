library(splines2)
library(graphics)

source('R/init.R')
source('R/model2/NN_model.R')

knot_pair <- apply(ref_amp,1,function(x) c(which(x == min(x)),which(x == max(x))))
knot_pair <- sort(unique(unlist(knot_pair)))    
x <- fd.output[,1]
bsMat <- bSpline(x, knots = knot_pair[35], degree = 1, intercept = T)

matplot(x, bsMat, type = "l", ylab = "Piecewise constant B-spline bases")
abline(v = knot_pair, lty = 2, col = "gray")


time <- Sys.time()
curve <- basis_MLP(test = fd.input,input = fd.input,output = ref_amp,
                         learning.rate = 0.01,basis = bsMat,step = 20,seed = 1)
elapse_time <- Sys.time() - time
