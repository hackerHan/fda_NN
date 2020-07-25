source('R/init.R')

time <- Sys.time()
curve100 <- f_rbf_bp(input_sample = fd.input,train_input = fd.input, train_output = ref_amp, 
                  cluster_source = 'feature space', centroid_method ='mean', 
                  ncluster = 10, seed = 2, sigma_option = "totss",
                  sigma_value = 1, step = 100,alpha = 0.01,gamma_min = 0,gamma_max = 1)

elapse_time <- Sys.time() - time




time <- Sys.time()
curve1000 <- f_rbf_bp(input_sample = fd.input,train_input = fd.input, train_output = ref_amp, 
                     cluster_source = 'feature space', centroid_method ='mean', 
                     ncluster = 10, seed = 2, sigma_option = "totss",
                     sigma_value = 1, step = 1000,alpha = 0.01,gamma_min = 0,gamma_max = 1)

elapse_time1 <- Sys.time() - time


for(i in 1:125)
{
  plot(ref_amp[i,]-curve1000[[i]],ylim = c(-0.2,0.2),ylab = paste('residual'),
       xlab = 'frequency', main = paste('test set ',i,' residual plot',sep = ""))
}