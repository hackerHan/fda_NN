act_func <- function(object, target)
{
  object <- as.numeric(object) # standardize object data
  
  target <- as.numeric(target) # standardize target data
  
  euclidean_distance <- norm(object-target,'2') # 2-norm
  
  act <- (-1*euclidean_distance^2)/2
  
  return(act) # return activation function
}

centroid <- function(input = fd.input, output = ref_amp, 
                     cluster_source = 'feature space', 
                     centroid_method ='mean', 
                     ncluster = 10, seed = 1,
                     cluster_step,
                     nstart,
                     kmeans_algorithm)
{
  # initialization
  input <- data.frame(input)
  
  output <- data.frame(output)
  
  nfeature <- dim(input)[2L] # dimension of feature space
  
  namp <- dim(output)[2L] # dimension of curve space
  
  feature_centroid <- matrix(NA, nrow = nfeature, ncol = ncluster) # initialize feature centroid
  
  curve_centroid <- matrix(NA, nrow = namp, ncol = ncluster) # initialization feature centroid
  
  # use input feature space to generate cluster
  if(cluster_source == 'feature space' & centroid_method == 'mean')
  {
    # set.seed
    set.seed(seed)
    
    # generate cluster object
    input_cluster <- kmeans(input, centers = ncluster,
                            iter.max = cluster_step,
                            nstart = nstart,
                            algorithm = kmeans_algorithm)
    
    # cluster index
    cluster_idx <- input_cluster$cluster
    
    # feature centroid
    feature_centroid <- t(input_cluster$centers) # each centroid is a column vector
    
    size <- input_cluster$size # number of points in each cluster
    
    # use cluster index to calculate curve centroid
    for(i in 1:ncluster)
    {
      curve_centroid[,i] <- apply(output[cluster_idx == i,], 2, mean)
    }
  }
  
  # use output curve space to generate cluster
  else if(cluster_source == 'curve space' & centroid_method == 'mean')
  {
    set.seed(seed)
    
    # generate cluster object
    output_cluster <- kmeans(output, centers = ncluster,
                             iter.max = cluster_step,
                             nstart = nstart,
                             algorithm = kmeans_algorithm)
    
    # cluster index
    cluster_idx <- output_cluster$cluster
    
    # curve centroid
    curve_centroid <- t(output_cluster$centers) # each centroid is a column vector
    
    size <- output_cluster$size # number of points in each cluster
    
    # use cluster index to calculate feature centroid
    for(i in 1:ncluster)
    {
      feature_centroid[,i] <- apply(input[cluster_idx == i,], 2, mean)
    }
  }
  # use input feature space to generate cluster
  else if(cluster_source == 'feature space' & centroid_method == 'nearest neighbor')
  {
    set.seed(seed)
    
    # generate cluster object
    input_cluster <- kmeans(input,centers = ncluster,
                            iter.max = cluster_step,
                            nstart = nstart,
                            algorithm = kmeans_algorithm)
    # cluster index
    cluster_idx <- input_cluster$cluster
    
    size <- input_cluster$size # number of points in each cluster
    
    # euclidean norm
    for(i in 1:ncluster)
    {
      # feature space in each cluster
      feature_cluster <- input[which(cluster_idx == i),]
      
      # curve space in each cluster
      curve_cluster <- output[which(cluster_idx == i),]
      
      # Initialize Euclidean distance 
      distance <- NULL
      for(j in 1:dim(feature_cluster)[1L])
      {
        # calculate euclidean distance for each feature observation in the same cluster
        distance[j] <- norm(feature_cluster[j,]-input_cluster$centers[i,],'2')
      }
      
      # index with minimum euclidean distance
      min_idx <- which(distance == min(distance))
      
      feature_centroid[,i] <- t(feature_cluster[min_idx,])
      
      curve_centroid[,i] <- t(curve_cluster[min_idx,])
    }
  }
  
  # use output curve space to generate cluster
  else
  {
    set.seed(seed)
    
    # generate cluster object
    output_cluster <- kmeans(output,centers = ncluster,
                             iter.max = cluster_step,
                             nstart = nstart,
                             algorithm = kmeans_algorithm)
    
    # cluster index
    cluster_idx <- output_cluster$cluster
    
    size <- output_cluster$size # number of points in each cluster
    
    # euclidean norm
    for(i in 1:ncluster)
    {
      # feature space in each cluster
      feature_cluster <- input[which(cluster_idx == i),]
      
      # curve space in each cluster
      curve_cluster <- output[which(cluster_idx == i),]
      
      # Initialize Euclidean distance 
      distance <- NULL
      for(j in 1:dim(curve_cluster)[1L])
      {
        # calculate euclidean distance for each curve observation in the same cluster
        distance[j] <- norm(curve_cluster[j,]-output_cluster$centers[i,],'2')
      }
      
      # index with minimum euclidean distance
      min_idx <- which(distance == min(distance))
      
      feature_centroid[,i] <- t(feature_cluster[min_idx,])
      
      curve_centroid[,i] <- t(curve_cluster[min_idx,])
    }
  }
  
  # export 2 types of centroids and within-cluster and total sum of squares and the number of points in each cluster
  return(list(feature_centroid = feature_centroid,
              curve_centroid = curve_centroid,
              size = size))
}


f_rbf_bp <- function(train_input = fd.input, train_output = ref_amp, 
                     cluster_source = 'feature space', centroid_method ='mean', 
                     ncluster = 10, seed = 1,  step = 10,
                     alpha = 0.0001,gamma_min = 0, gamma_max = 1,
                     cluster_step = 10,
                     nstart = 10,
                     kmeans_algorithm ='Hartigan-Wong')
{
  
  # calculate hidden neuron centroid
  centroid <- centroid(input = train_input, output = train_output, 
                       cluster_source = cluster_source, 
                       centroid_method =centroid_method, 
                       ncluster = ncluster, seed = seed,
                       cluster_step = cluster_step,
                       nstart = nstart,
                       kmeans_algorithm = kmeans_algorithm)
  
  feature.centroid <- centroid$feature_centroid # feature centroid
  
  curve.centroid <- centroid$curve_centroid # curve centroid
  
  size <- centroid$size # the number of points in each cluster
  
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
  
  for(iter in 1:step)
  {
    for(i in 1:nrow(train_input))
    {
      a[,i] <- apply(feature.centroid,2,
                     function(x) act_func(object = train_input[i,],
                                          target = x))
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
    }
    # compute gradient for feature.centroid
    for(j in 1:ncluster)
    {
      grad_mu <- matrix(0, nrow = ncol(train_input), ncol = ncluster)
      for(i in 1:nrow(grad_a))
      {
        grad_mu[,j] <- grad_mu[,j] + grad_a[j,i] * (train_input[i,]-feature.centroid[,j])
      }
    }
    gamma <- gamma - alpha * rowSums(grad_g)
    feature.centroid <- feature.centroid - alpha * grad_mu
    curve.centroid <- curve.centroid - alpha * Reduce('+',grad_curve)
  }
  
  # ========================== Final model ==================================== #
  for(i in 1:nrow(train_input))
  {
    a[,i] <- apply(feature.centroid,2,
                   function(x) act_func(object = train_input[i,],
                                        target = x))
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
  return(list(output_dimension = ncol(train_output),
              weight = rbf.w,
              gamma = gamma,
              feature_centroid = feature.centroid,
              curve_centroid = curve.centroid))
}

# ========================== Prediction part ================================ #
# prediction one a single input observation
rbf_predict <- function(input_sample,output_dimension,ncluster,rbf.w,
                        gamma,feature.centroid,curve.centroid)
{
  # predicted basis

  pred.basis <- (matrix(1,nrow = output_dimension) %*% 
                 t(exp(matrix(gamma,ncol = 1) * 
                         matrix(apply(feature.centroid,2,
                                      function(x) act_func(object = input_sample,
                                                           target = x)),ncol = 1)))) * 
  curve.centroid
  
  # predicted output
  f.curve <- pred.basis %*% rbf.w
  return(list(radial_bases = pred.basis,curve = f.curve))
}