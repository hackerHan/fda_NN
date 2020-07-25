act_func <- function(object, target, sigma)
{
  object <- as.numeric(object) # standardize object data
  
  target <- as.numeric(target) # standardize target data
  
  euclidean_distance <- norm(object-target,'2') # 2-norm
  
  act <- exp((-1*euclidean_distance^2)/(2*sigma)) # Gaussian function
  
  return(act) # return activation function
}

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


centroid <- function(input = fd.input, output = ref_amp, cluster_source = 'feature space', 
                     centroid_method ='mean', ncluster = 10, seed = 1)
{
  # initialization
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
    input_cluster <- kmeans(input, centers = ncluster)
    
    # cluster index
    cluster_idx <- input_cluster$cluster
    
    # feature centroid
    feature_centroid <- t(input_cluster$centers) # each centroid is a column vector
    
    WCSS <- input_cluster$withinss # within-cluster sum of squares
    
    totss <- input_cluster$totss # total sum of squares
    
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
    output_cluster <- kmeans(output, centers = ncluster)
    
    # cluster index
    cluster_idx <- output_cluster$cluster
    
    # curve centroid
    curve_centroid <- t(output_cluster$centers) # each centroid is a column vector
    
    WCSS <- output_cluster$withinss # within-cluster sum of squares
    
    totss <- output_cluster$totss # total sum of squares
    
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
    input_cluster <- kmeans(input,centers = ncluster)
    # cluster index
    cluster_idx <- input_cluster$cluster
    
    WCSS <- input_cluster$withinss # within-cluster sum of squares
    
    totss <- input_cluster$totss # total sum of squares
    
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
    output_cluster <- kmeans(output,centers = ncluster)
    
    # cluster index
    cluster_idx <- output_cluster$cluster
    
    WCSS <- output_cluster$withinss # within-cluster sum of squares
    
    totss <- output_cluster$totss # total sum of squares
    
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
              WCSS = WCSS,
              totss = totss,
              size = size))
}



f_rbf <- function(train_input = fd.input, train_output = ref_amp, 
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
  
  
  return(list(weight = rbf.w, feature_centroid = feature.centroid,
              curve_centroid = curve.centroid,sigma_value = sigma.value))
}

# ============================ predict function ====================== #
f_rbf_predict <- function(input_sample,feature.centroid,
                          curve.centroid,sigma.value,rbf.w)
{
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
