# basis function
# each hidden layer neuron is a basis function
# number of hidden layer neurons is number of clusters
# @ train_data is input data to generate basis involving similarity values
# @ feature_centroid is feature centroids involved in hidden neurons
# @ curve_centroid is curve centroids involved in hidden neurons
# @ sigma_value is sigma value involved in activation function

Act_basis <- function(train_data = fd.input, feature_centroid, 
                      curve_centroid, sigma_value)
{
  # Initialization
  ninput <- ifelse(is.null(dim(train_data)[1L]),1,dim(train_data)[1L]) # number of training samples
  
  namp <- dim(curve_centroid)[1L] # dimension of curve space
  
  ncluster <- dim(curve_centroid)[2L] # number of clusters
  
  # initialize basis as a list
  basis <- rep(list(matrix(NA, nrow = namp,ncol = ncluster)), times = ninput)
  
  
  if(length(sigma_value) == 1) # use total variance or user-defined variance
  {
    for(i in 1:length(basis)) # execute for each training sample
    {
      basis[[i]] <- curve_centroid %*%
        diag(apply(feature_centroid,2,
                   function(x) act_func(object = as.matrix(train_data)[i,],
                                        target = x, sigma = sigma_value)))
    }
  }
  
  else if(length(sigma_value) > 1) # use within-cluster variance
  {
    for(i in 1:length(basis)) # execute for each training sample
    {
      for(j in 1:ncluster)
      {
        basis[[i]][,j] <- curve_centroid[,j] * act_func(object = as.matrix(train_data)[i,],
                                                        target = feature_centroid[,j],sigma = sigma_value[j])
      }
    }
  }
  else
  {
    stop('sigma value doesn\'t make sense')
  }
  
  
  
  return(basis)
}



