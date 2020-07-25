# @ test is the test set
# @ input is the training set input
# @ output is the training set output
# @ learning.rate is the rate of change on gradient descent; default value is 0.0001
# @ basis is the presumed basis
# @ step is the iteration times; Default value is 20

basis_MLP <- function(test,input,output,learning.rate=0.0001,basis,step = 20,seed  = 1)
{
  # input dimension
  m <- dim(input)[2]
  # number of training samples
  N <- dim(input)[1]
  # output dimension
  n <- dim(output)[2]
  # number of basis elements
  n_bas <- ncol(basis)
  # initialize weight matrix connecting input layer and first hidden layer
  set.seed(seed)
  W_1 <- matrix(rnorm(n_bas*m), nrow = n_bas, ncol = m)
  b_2 <- rep(list(matrix(0, nrow = n,ncol = n_bas)), 
             times = N)
  # iterate training
  for(iter in 1:step)
  {
    Z_2 <- W_1 %*% t(input) # n_bas \times N
    a_2 <- exp(Z_2) # n_bas \times N
    for(j in 1:N)
    {
      b_2[[j]] <- (matrix(1,nrow = n) %*% t(a_2[,j])) * basis
    }

    # calculate weight connecting hidden layer and output layer
    R <- 0
    r <- 0
    for(i in 1:length(b_2))
    {
      R <- t(b_2[[i]]) %*% b_2[[i]] + R
      r <- t(b_2[[i]]) %*% output[i,] +r
    }
    
    # model weight
    wt <- solve(R,tol = 1e-17) %*% r
    
    Loss <- mean(apply(matrix(c(1:nrow(input))),1,
                       function(x) 0.5*norm(output[x,]-b_2[[x]] %*% 
                                              wt,"2")^2))/n
    message(paste('step',iter,sep = ''),' ','loss',' ',Loss)
    
    # initialize gradient
    grad_b2 <- rep(list(matrix(0, nrow = n,ncol = n_bas)), 
                   times = N)
    grad_a2 <- matrix(0,nrow = n_bas, ncol = N)
    
    grad_phi <- rep(list(matrix(0,nrow = n, ncol = n_bas)),
                    times = N)
    grad_wt <- rep(list(matrix(0,nrow = n_bas, ncol = m)),
                   times = N)
    
    for(j in 1:N)
    {
      grad_b2[[j]] <- (b_2[[i]] %*% wt - output[j,]) %*% t(wt)/(N*n)
      grad_a2[,j] <- t(grad_b2[[j]] * basis) %*% matrix(1,nrow = n)
      grad_phi[[j]] <- grad_b2[[j]] * (matrix(1,nrow = n)%*%t(a_2[,j]))
      grad_wt[[j]] <- (grad_a2[,j] * a_2[,j]) %*% 
        matrix(input[j,],ncol = 4)
    }
    
    W_1 <- W_1 - learning.rate * Reduce('+',grad_wt)
    basis <- basis - learning.rate * Reduce('+',grad_phi)
  }
  
#================================ Final model ==================================#
  Z_2 <- W_1 %*% t(input) # n_bas \times N
  a_2 <- exp(Z_2) # n_bas \times N
  for(j in 1:N)
  {
    b_2[[j]] <- (matrix(1,nrow = n) %*% t(a_2[,j])) * basis
  }
  
  # calculate weight connecting hidden layer and output layer
  R <- 0
  r <- 0
  for(i in 1:length(b_2))
  {
    R <- t(b_2[[i]]) %*% b_2[[i]] + R
    r <- t(b_2[[i]]) %*% output[i,] +r
  }
  
  # model weight
  wt <- solve(R,tol = 1e-17) %*% r
  
  Loss <- mean(apply(matrix(c(1:N)),1,
                     function(x) 0.5*norm(output[x,]-b_2[[x]] %*% 
                                            wt,"2")^2))/n
  message(paste('final model',sep = ''),' ','loss',' ',Loss)
  
#============================== Model prediction ===============================#
  Ntest <- dim(test)[1] # number of test samples
  z_test <- W_1 %*% t(test)
  a_test <- exp(z_test)
  prediction <- apply(matrix(c(1:Ntest),ncol = Ntest),2,
                      function(x) ((matrix(1,nrow = n) %*%
                        t(a_test[,x])) * basis) %*% wt)
  
  # output function attributes
  return(prediction = prediction)
}


