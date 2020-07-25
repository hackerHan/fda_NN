# rbf neural network for functional output data
# below is the final model to predict the functional output (curve) given any known feature vector
# @ if sigma_option is 'within-cluster', then sigma value in activation function will be variance within each cluster
# @ input_sample is sample to be predicted
# @ train_input is sample to train feature centroid
# @ train_output is sample to train curve centroid
# @ cluster_source == 'feature space', then input feature will be used to generate clusters
# @ cluster_source == 'curve space', then output curve will be used to generate clusters
# @ centroid method is the way to get centroid; 
# @ If centroid_method = 'mean', obtain centroid using average value;
# @ If centroid_method = 'nearest neighbor', obtain centroid using observation which is closest to the cluster mean
# @ ncluster is number of clusters as well as number of hidden neurons
# @ seed is used to set.seed(seed)
# @ if sigma_option is 'totss', then sigma value in activation function will be the same with total variance
# @ if sigma_option is 'within-cluster', then sigma value in activation function will be assigned with variance within cluster
# @ if sigma_option is 'user-defined', then sigma value in activation function will be defined at random
# @ sigma_value is defaulted value in activation function when sigma_option is set to be 'user-defined'
# @ if multiplier is 'sigma', the reciprocal of within cluster variance will be multiplied to corresponding basis
f_rbf <- function(input_sample,train_input = fd.input, train_output = ref_amp, 
                  cluster_source = 'feature space', centroid_method ='mean', 
                  ncluster = 10, seed = 1, sigma_option = "within-cluster",
                  sigma_value = 1)
{
  
  # calculate hidden neuron centroid
  centroid <- centroid(input = train_input, output = train_output, 
                           cluster_source = cluster_source, 
                       centroid_method =centroid_method, 
                       ncluster = ncluster, seed = seed)
  
  feature.centroid <- centroid$feature_centroid # feature centroid
  
  curve.centroid <- centroid$curve_centroid # curve centroid
  
  WCSS <- centroid$WCSS # within-cluster sum of squares
  
  size <- centroid$size # the number of points in each cluster
  
  # calculate sigma value for activation function
  if(sigma_option == 'totss')
  {
    # total variance
    sigma.value <-centroid$totss/nrow(train_input)
  }
  
  else if(sigma_option == 'within-cluster')
  {
    sigma.value <- WCSS/size # within-cluster variance
  }
  
  else if(sigma_option == 'user-defined')
  {
    sigma.value <- sigma_value
  }
  
  else
  {
    stop("sigma_option doesn\'t match the record")
  }
  
  # hidden neuron basis for a given training set
  rbf.basis <- Act_basis(train_data = train_input,
                         feature_centroid = feature.centroid,
                         curve_centroid = curve.centroid,
                         sigma_value = sigma.value)
  
  # weight connecting hidden neuron basis and output layer
  # correlation function of the hidden unit outputs
  R <- 0
  
  # cross correlation vector between the desired reponse at the output of the RBF network
  # and the hidden unit outputs
  r <- 0
  
  for(i in 1:length(rbf.basis))
  {
    R <- t(rbf.basis[[i]]) %*% rbf.basis[[i]] + R
    r <- t(rbf.basis[[i]]) %*% train_output[i,] +r
  }
  
  # model weight
  rbf.w <- solve(R,tol = 1e-17) %*% r
  
  
  # predicted basis
  # saved as a list
  pred.basis <- Act_basis(train_data = input_sample,feature_centroid = feature.centroid,
                          curve_centroid = curve.centroid,
                          sigma_value = sigma.value)
  
  # predicted output
  # saved as a list
  f.curve <- lapply(pred.basis,function(x) x %*% rbf.w)
  return(f.curve)
}
