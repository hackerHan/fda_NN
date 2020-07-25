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

source('functions.R')
fd.input <- readRDS('data/fd_input.rds')
ref_amp <- readRDS('data/ref_amp.rds')
trans_amp <- readRDS('data/trans_amp.rds')
fd.output <- readRDS('data/fd_output.rds')


# each observation use row vector
dim(ref_amp) # 500*1001
dim(trans_amp) # 500*1001

# User interface ----
ui <- fluidPage(
  
  titlePanel(fluidRow(align = 'center', "functional curve")),
  
  mainPanel(
            fluidRow(align = 'center',
                     column(6,withMathJax(code("if centroid method == mean"),
                                 "$$\\mu_{i} = \\frac{1}{N_{i}}\\sum_{X\\in C_i}X_i\\text{ and }
                                 g_{\\mu_{i}}=\\frac{1}{N_{i}}\\sum_{X\\in C_i}g_{X}$$")),
                     column(6,withMathJax(code('if centroid method = nearest neighbor'),
                                "$$\\mu_{i} = argmin_{X}||X_i-\\frac{1}{N_{i}}\\sum_{X\\in C_i}X_i||_2^2\\text{ and }
                                g_{\\mu_i}=\\text{ corresponding curve of }\\mu_i\\text{ if cluster source == feature space}$$"),
                                withMathJax("$$g_{\\mu_{i}}=\\frac{1}{N_{i}}\\sum_{X\\in C_i}g_{X_i}\\text{ and }
                                            \\mu_{i} = \\text{ corresponding feature of }g_{\\mu_i}\\text{ if cluster source == curve space}$$"))),
            fluidRow(align = 'center',column(6,withMathJax(code("if sigma option == within-cluster"),
                        "$$\\sigma_i^2 = \\frac{1}{N_i}\\sum_{X\\in C_i}||X-\\mu_i||_2^2$$")),
            column(6,withMathJax(code('if sigma option == total variance'),
                        "$$\\sigma^2 = \\frac{1}{N_{total}}\\sum_{i=1}^{N_{total}}||X_i-\\bar{X}||_2^2$$"))),
            width = 12),
  
  
  fluidRow(
    column(12, 
           wellPanel(fluidRow(
             column(4,
                    selectInput('Output', 'prediction', 
                                list('reflection amplitude','transmission amplitude'), 
                                selected = 'reflection amplitude'),
                    selectInput('source', 'cluster source', 
                                list('feature space','curve space'), 
                                selected = 'curve space'),
                    selectInput('method', 'centroid method', 
                                list('mean','nearest neighbor'), 
                                selected = 'nearest neighbor'),
                    selectInput('sigma', 'sigma option', 
                                list('within-cluster variance','total variance','user-defined variance'), 
                                selected = 'user-defined variance')
                    ),
             column(4,
                    numericInput('order', 'observation id', 
                                 value = 1,min = 1, max = 125, step = 1),
                    numericInput('ncluster','number of clusters', 
                                 value = 10, min = 2, max = NA, step = 1),
                    numericInput('seed', 'set.seed', 
                                 value = 1, min = 1, max = NA, step =1),
                    numericInput('sigma_value', 'sigma value', 
                                 value = 1, min = NA, max = NA)
             ),
             column(4,
                    numericInput('a', 'a', min = 0, max = 0.9, step = 0.01, value = 0.46),
                    numericInput('b','b', min = 0, max = 0.9, step = 0.01, value = 0.65),
                    numericInput('t', 'tBCB',  min = 0, max = 0.9, step =0.01, value = 0.9),
                    numericInput('r', 'rMDA', min = 0, max = 0.9, step = 0.01, value  = 0.6)
             )
           ))
           )
  
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
  )
  
  
  
)


# Server logic ----
server <- function(input, output) {
  a.original.curve <- reactive({
    switch(input$Output,
           'reflection amplitude' = ref_amp, 
           'transmission amplitude' = trans_amp)[input$order,]
  })
  model_parameter <- reactive({
    f_rbf(train_input = fd.input, 
          train_output = switch(input$Output,'reflection amplitude' = ref_amp, 
                                'transmission amplitude' = trans_amp), 
          cluster_source = input$source, centroid_method = input$method, 
          ncluster = input$ncluster, seed = input$seed, 
          sigma_option = switch(input$sigma,'within-cluster variance' = 'within-cluster',
                                'total variance' = 'totss',
                                'user-defined variance' = 'user-defined'), 
          sigma_value = input$sigma_value)
  })
  a.pred_curve <- reactive({
    f_rbf_predict(input_sample = fd.input[input$order,],
                  feature.centroid = model_parameter()$feature_centroid,
                  curve.centroid = model_parameter()$curve_centroid,
                  sigma.value = input$sigma_value,
                  rbf.w = model_parameter()$weight)[[1]]
  })
  b.pred_curve <- reactive({
    f_rbf_predict(input_sample = c(input$a,input$b,input$t,input$r),
                  feature.centroid = model_parameter()$feature_centroid,
                  curve.centroid = model_parameter()$curve_centroid,
                  sigma.value = input$sigma_value,
                  rbf.w = model_parameter()$weight)[[1]]
  })
  output$`a1` <- renderPlot({
    ggplot(data = data.frame(freq = fd.output[,1], Abs = a.original.curve()))+
      geom_line(mapping = aes(x = freq, 
                              y = Abs)) +
      labs(x = "frequency",y = "Abs") +
      ggtitle(paste("original",input$Output)) +
      theme(plot.title = element_text(hjust=0.5)) +
      ylim(0,1)
  })
  output$`a2` <- renderPlot({
    ggplot(data = data.frame(freq = fd.output[,1], Abs = a.pred_curve()))+
      geom_line(mapping = aes(x = freq, 
                              y = Abs)) +
      labs(x = "frequency",y = "Abs") +
      ggtitle(paste("predicted",input$Output)) +
      theme(plot.title = element_text(hjust=0.5)) +
      ylim(0,1)
  })
  output$basis <- renderPlot({
    matplot(1:dim(fd.output)[1], model_parameter()$curve_centroid,
            type = "l",main = "k-means radial basis",
            xlab = 'frequency',ylab = 'amplitude')
  })
  output$`b1` <- renderPlot({
    ggplot(data = data.frame(freq = fd.output[,1], Abs = b.pred_curve()))+
      geom_line(mapping = aes(x = freq, 
                              y = Abs)) +
      labs(x = "frequency",y = "Abs") +
      ggtitle(paste("predicted",input$Output)) +
      theme(plot.title = element_text(hjust=0.5)) +
      ylim(0,1)
  })
  
  
}

# Run app ----
shinyApp(ui, server)