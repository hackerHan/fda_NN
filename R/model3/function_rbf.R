# backpropagation algorithm involved
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
# @ if sigma_option is 'user-defined', then sigma value in activation function will be defined at random
# @ sigma_value is defaulted value in activation function when sigma_option is set to be 'user-defined'
# @ if multiplier is 'sigma', the reciprocal of within cluster variance will be multiplied to corresponding basis
# @ step is backpropagation step; default value is 10
# @ alpha is the learning rate; default value is 0.0001
# @ gamma_min and gamma_max are range of initial value of gamma
f_rbf_bp <- function(input_sample,train_input = fd.input, train_output = ref_amp, 
                  cluster_source = 'feature space', centroid_method ='mean', 
                  ncluster = 10, seed = 1, sigma_option = "totss",
                  sigma_value = 1, step = 10,alpha = 0.0001,gamma_min = 0, gamma_max = 1)
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
  
  else if(sigma_option == 'user-defined')
  {
    sigma.value <- sigma_value
  }
  
  else
  {
    stop("sigma_option doesn\'t make sense")
  }
  
# ========================== Backpropagation part ===========================#
  
  # initialize sources
  a <- matrix(NA, nrow = ncluster, ncol = nrow(train_input))
  gamma <- runif(ncluster,min = gamma_min,max = gamma_max)
  rbf.basis <- rep(list(matrix(NA, nrow = ncol(train_output),ncol = ncluster)), 
                   times = nrow(train_input))
  grad_phi <- rep(list(matrix(NA, nrow = ncol(train_output),ncol = ncluster)), 
                  times = nrow(train_input))
  grad_b <- matrix(NA,nrow = ncluster, ncol = nrow(train_input))
  grad_g <- matrix(NA, nrow = ncluster, ncol = nrow(train_input))
  grad_a <- matrix(NA, nrow = ncluster, ncol = nrow(train_input))
  grad_curve <- rep(list(matrix(NA, nrow = ncol(train_output),ncol = ncluster)), 
                    times = nrow(train_input))
  grad_sigma <- matrix(NA, nrow = nrow(train_input))
  
  for(iter in 1:step)
  {
    for(i in 1:nrow(train_input))
    {
      a[,i] <- apply(feature.centroid,2,
                     function(x) act_func(object = train_input[i,],
                                          target = x, sigma = sigma.value))
    }
    b <- a * gamma
    # hidden neuron basis for a given training set
    for(i in 1: nrow(train_input))
    {
      rbf.basis[[i]] <- (matrix(1,ncol(train_output),1) %*%
                           matrix(exp(b[,i]),nrow = 1)) * curve.centroid
    }
    
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
    
    # Loss function
    Loss <- mean(apply(matrix(c(1:nrow(train_input))),1,
                       function(x) 0.5*norm(train_output[x,]-rbf.basis[[x]] %*% 
                                              rbf.w,"2")^2))
    
    message(paste('step',iter,sep = ''),' ','loss',' ',Loss)
      
      
    # gradient
    for(i in 1:nrow(train_input))
    {
      grad_phi[[i]] <- (rbf.basis[[i]]%*%
                          rbf.w-matrix(train_output[i,],ncol = 1)) %*% 
        t(rbf.w)/nrow(train_input)
      grad_curve[[i]] <- grad_phi[[i]] * 
        (matrix(1,nrow = ncol(train_output)) %*% 
           t(exp(b[,i]))) # part of gradient for curve
      
      grad_b[,i] <- exp(b[,i]) * (t((grad_phi[[i]])*curve.centroid) %*%
                                    matrix(1,nrow = ncol(train_output)))
      grad_g[,i] <- grad_b[,i] * a[,i] # part of gradient for gamma
      grad_a[,i] <- grad_b[,i] * gamma 
      grad_sigma[i] <- t(a[,i]) %*% grad_a[,i]/(-sigma.value) # part of gradient for sigma
      
    }
    # compute gradient for feature.centroid
    for(j in 1:ncluster)
    {
      grad_mu <- matrix(0, nrow = ncol(train_input), ncol = ncluster)
      for(i in 1:nrow(grad_a))
      {
        grad_mu[,j] <- grad_mu[,j] + grad_a[j,i] * (train_input[i,]-feature.centroid[,j])/sigma.value
      }
    }
    gamma <- gamma - alpha * rowSums(grad_g)
    feature.centroid <- feature.centroid - alpha * grad_mu
    sigma.value <- sigma.value - alpha * sum(grad_sigma)
    curve.centroid <- curve.centroid - alpha * Reduce('+',grad_curve)
    
  }

# ========================== Final model ==================================== #
  for(i in 1:nrow(train_input))
  {
    a[,i] <- apply(feature.centroid,2,
                   function(x) act_func(object = train_input[i,],
                                        target = x, sigma = sigma.value))
  }
  b <- a * gamma
  # hidden neuron basis for a given training set
  for(i in 1: nrow(train_input))
  {
    rbf.basis[[i]] <- (matrix(1,ncol(train_output),1) %*%
                         matrix(exp(b[,i]),nrow = 1)) * curve.centroid
  }
  
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
  
  # Loss function
  Loss <- mean(apply(matrix(c(1:nrow(train_input))),1,
                     function(x) 0.5*norm(train_output[x,]-rbf.basis[[x]] %*% 
                                            rbf.w,"2")^2))
  
  message(paste('final model',sep = ''),' ','loss',' ',Loss)
  
# ========================== Prediction part ================================ #
  
  # predicted basis
  # saved as a list
  pred.basis <-  rep(list(matrix(NA, nrow = ncol(train_output),ncol = ncluster)), 
                     times = ifelse(is.null(dim(input_sample)[1L]),1,
                                    dim(input_sample)[1L]))
  for(i in 1: length(pred.basis))
  {
    pred.basis[[i]] <- (matrix(1,nrow = ncol(train_output)) %*% 
      t(exp(matrix(gamma,ncol = 1) * 
              matrix(apply(feature.centroid,2,
                           function(x) act_func(object = input_sample[i,],
                                                target = x, 
                                                sigma = sigma.value)),ncol = 1)))) * 
      curve.centroid
  }
  
  # predicted output
  # saved as a list
  f.curve <- lapply(pred.basis,function(x) x %*% rbf.w)
  return(f.curve) # output predicted curve
}