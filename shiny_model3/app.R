#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(ggpubr)
library(ggplot2)
library(zoo)

source('functions.R')
fd.input <- readRDS('data/fd_input.rds')
ref_amp <- readRDS('data/ref_amp.rds')
trans_amp <- readRDS('data/trans_amp.rds')
fd.output <- readRDS('data/fd_output.rds')

# normalize fd.input
norm.input <- matrix(0,nrow = 500,ncol = 4)
for(i in 1:4)
{
  norm.input[,i] <- (fd.input[,i]-mean(fd.input[,i]))/sqrt(var(fd.input[,i]))
}


# each observation use row vector
dim(ref_amp) # 500*1001
dim(trans_amp) # 500*1001

# User interface ----
ui <- fluidPage(
  tags$style(type = 'text/css',".irs-grid-pol.small {height: 0px;}"),
  titlePanel(fluidRow(align = 'center', "Neural Network for Functional Response")),
  fluidRow(
    column(12, 
           wellPanel(fluidRow(
             column(3,br(),br(),
                    selectInput('algorithm','kmeans algorithm',
                                list("Hartigan-Wong", "Lloyd", "Forgy",
                                     "MacQueen"),
                                selected = 'Hartigan-Wong'),
                    sliderInput('iter','kmeans iteration step',
                                min = 10, max = 100,
                                step = 10,value = 10),
                    sliderInput('nstart','number of initial center',
                                min = 1, max = 100,
                                step = 1 ,value = 10))
             ,
             column(3,br(),br(),
                    selectInput('Output', 'response', 
                                list('reflection amplitude',
                                     'transmission amplitude'), 
                                selected = 'transmission amplitude'),
                    selectInput('source', 'cluster source', 
                                list('feature space','curve space'), 
                                selected = 'curve space'),
                    # need to switch cluster mean to mean
                    selectInput('method', 'centroid method', 
                                list('cluster mean','nearest neighbor'), 
                                selected = 'cluster mean'),
                    selectInput('order', 'observation id', 
                                lapply(1:500,function(x) list(x)[[1]]), selected  = 5)
             ),
             column(3,h4('Model tuning parameters',align = 'center'),
                    sliderInput('ncluster','number of clusters', 
                                 value = 10, min = 2, max = 20, step = 1),
                    sliderInput('seed', 'set.seed', 
                                 value = 1, min = 0, max = 100, step =1),
                    sliderInput('alpha','learning rate',
                                value = 0.05,min = 0.05,max = 0.15,step = 0.01),
                    numericInput('step','step',
                                value = 2,min = 0, max = 2500, step = 1)),
             column(3,h4("Prediction structure parameters",align = 'center'),
                    sliderInput('a', 'a', min = 0, max = 0.9, 
                                 step = 0.001, value = 0),
                    sliderInput('b','b', min = 0, max = 0.9, 
                                 step = 0.001, value = 0),
                    sliderInput('t', 'tBCB',  min = 0.1, max = 0.9, 
                                 step =0.001, value = 0.34),
                    sliderInput('r', 'rMDA', min = 0.61111, max = 0.77778, 
                                 step = 0.00001, value  = 0.77778)
             )
             )
           ))
    ),
  fluidRow(
    column(6,
           plotOutput('a1', height = "300px"),
           plotOutput('a2',height = '300px')
    ),
    column(6,
           plotOutput('basis',height = '300px'),
           plotOutput('b1', height = "300px")
    )
  ))


# Server logic ----
server <- function(input, output) {
  a.original.curve <- reactive({
    switch(input$Output,
           'reflection amplitude' = ref_amp, 
           'transmission amplitude' = trans_amp)[as.numeric(input$order),]
  })
  model_parameter <- reactive({
    f_rbf_bp(train_input = norm.input, 
             train_output = switch(input$Output,'reflection amplitude' = ref_amp, 
                                   'transmission amplitude' = trans_amp), 
             cluster_source = input$source, 
             centroid_method = switch(input$method,
                                      'cluster mean' = 'mean',
                                      'nearest neighbor' = 'nearest neighbor'), 
             ncluster = input$ncluster, seed = input$seed, 
             step = input$step,alpha = input$alpha,
             gamma_min = 0, 
             gamma_max = 2,
             cluster_step = input$iter,
             nstart = input$nstart,
             kmeans_algorithm = input$algorithm)
  })
  a.pred.curve <- reactive({
    rbf_predict(input_sample = norm.input[as.numeric(input$order),],
                output_dimension = model_parameter()$output_dimension,
                ncluster = input$ncluster,
                rbf.w = model_parameter()$weight,
                gamma = model_parameter()$gamma,
                feature.centroid = model_parameter()$feature_centroid,
                curve.centroid = model_parameter()$curve_centroid)
  })

  b.pred.curve <- reactive({
    rbf_predict(input_sample = c((input$a-mean(fd.input[,1]))/sqrt(var(fd.input[,1])),
                                 (input$b-mean(fd.input[,2]))/sqrt(var(fd.input[,2])),
                                 (input$t-mean(fd.input[,3]))/sqrt(var(fd.input[,3])),
                                 (input$r-mean(fd.input[,4]))/sqrt(var(fd.input[,4]))
                                 ),
                output_dimension = model_parameter()$output_dimension,
                ncluster = input$ncluster,
                rbf.w = model_parameter()$weight,
                gamma = model_parameter()$gamma,
                feature.centroid = model_parameter()$feature_centroid,
                curve.centroid = model_parameter()$curve_centroid)
  })
  output$`a1` <- renderPlot({
    ggplot(data = data.frame(freq = fd.output[,1], Abs = a.original.curve()))+
      geom_line(mapping = aes(x = freq, 
                              y = Abs)) +
      labs(x = "frequency",y = "Abs") +
      ggtitle(paste("Original Training Curve")) +
      theme(plot.title = element_text(hjust=0.5)) +
      ylim(0,1)
  })
  output$`a2` <- renderPlot({
    ggplot(data = data.frame(freq = fd.output[,1], Abs = a.pred.curve()$curve))+
      geom_line(mapping = aes(x = freq, 
                              y = Abs)) +
      labs(x = "frequency",y = "Abs") +
      ggtitle(paste("Fitted Curve")) +
      theme(plot.title = element_text(hjust=0.5)) +
      ylim(0,1)
  })
  output$basis <- renderPlot({
    matplot(fd.output[,1],b.pred.curve()$radial_bases,
            type = "l",ylab = 'amplitude',xlab = 'frequency',
            main = "Radial Bases for Prediction",ylim = c(0,1))
  })
  
  output$`b1` <- renderPlot({
    ggplot(data = data.frame(freq = fd.output[,1], Abs = b.pred.curve()$curve))+
      geom_line(mapping = aes(x = freq, 
                              y = Abs)) +
      labs(x = "frequency",y = "Abs") +
      ggtitle(paste("Predicted Curve")) +
      theme(plot.title = element_text(hjust=0.5)) +
      ylim(0,1)
  })
  
  
}

# Run app ----
shinyApp(ui, server)