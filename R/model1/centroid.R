# For both cluster and cluster mean, the observation will be column vectors;
# centroid of each cluster will be the average of input (features) and output(curve) respectively
# @ input represents input feature space
# @ output represents output curve space; Default output is 'reflection amplitude curve'
# @ cluster_source == 'feature space', then input feature will be used to generate clusters
# @ cluster_source == 'curve space', then output curve will be used to generate clusters
# @ ncluster represents number of clusters; i.e, number of hidden layer neurons; Default value is 10
# @ seed is used to set.seed(seed)
# @ centroid method is the way to get centroid; 
# @ If centroid_method = 'mean', obtain centroid using average value;
# @ If centroid_method = 'nearest neighbor', obtain centroid using observation which is closest to the cluster mean
# clustering method is kmeans
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
