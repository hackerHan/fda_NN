# Activation function
# @ object is the input object
# @ target is the object used to compute the activation function for all input objects
# @ sigma is the parameter to control the scale
act_func <- function(object, target, sigma)
{
  object <- as.numeric(object) # standardize object data
  
  target <- as.numeric(target) # standardize target data
  
  euclidean_distance <- norm(object-target,'2') # 2-norm
  
  act <- exp((-1*euclidean_distance^2)/(2*sigma)) # Gaussian function
  
  return(act) # return activation function
}