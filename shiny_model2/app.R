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
library(splines2)
library(graphics)

source('functions.R')
fd.input <- readRDS('data/fd_input.rds')
ref_amp <- readRDS('data/ref_amp.rds')
trans_amp <- readRDS('data/trans_amp.rds')
fd.output <- readRDS('data/fd_output.rds')


#norm.input <- matrix(0,nrow = 500,ncol = 4)
#for(i in 1:4)
#{
#  norm.input[,i] <- (fd.input[,i]-mean(fd.input[,i]))/sqrt(var(fd.input[,i]))
#}


# each observation use row vector
dim(ref_amp) # 500*1001
dim(trans_amp) # 500*1001



# ============================= User interface ======================== #
ui <- fluidPage(
  
  titlePanel(fluidRow(align = 'center', "functional curve")),
  
  
  
  fluidRow(
    column(12, 
           wellPanel(fluidRow(
             column(4,
                    selectInput('Output', 'prediction', 
                                list('reflection amplitude',
                                     'transmission amplitude'), 
                                selected = 'reflection amplitude'),
                    numericInput('learning_rate','learning.rate',
                                 value = 0.0001),
                    numericInput('step','iteration step',
                                 value = 2)
             ),
             column(4,
                    sliderInput('a', 'a', 
                                 min = 0.1, max = 0.9, 
                                 step = 0.01, value = 0.46),
                    sliderInput('b','b', 
                                 min = 0.1, max = 0.9, 
                                 step = 0.01, value = 0.65),
                    sliderInput('t', 'tBCB',  
                                 min = 0.1, max = 0.9, 
                                 step =0.01, value = 0.9),
                    sliderInput('r', 'rMDA', 
                                 min = 0.61111, max = 0.77778, 
                                 step = 0.01, value  = 0.61111)
             ),
             column(4,
                    sliderInput('test','test id',
                                value = 10, min = 1, max = 125),
                    numericInput('degree','degree of the spline',
                                 value = 10),
                    checkboxInput("intercept","intercept",
                                  value = FALSE),
                    checkboxInput('transformation','basis transformation',
                                  value = TRUE))
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
           'transmission amplitude' = trans_amp)[input$test,]
  })
  # ============================= Generate B-spline basis =============== #
  basis <- reactive({
    knot_pair <- apply(switch(input$Output,
                              'reflection amplitude' = ref_amp, 
                              'transmission amplitude' = trans_amp),
                       1,function(x) c(which(x == min(x)),which(x == max(x))))
    knot_pair <- sort(unique(unlist(knot_pair)))
    freq <- seq.int(1, dim(fd.output)[1], 1)
    if(input$transformation == FALSE)
    {
      bSpline(freq, 
              # make sure all internal knots are placed inside of boundary knots
              knots = knot_pair[which(knot_pair<1001 & knot_pair>1)], 
              degree = input$degree, 
              intercept = input$intercept) 
    }
    else if(input$transformation == TRUE)
    {
      exp(bSpline(freq, 
                  # make sure all internal knots are placed inside of boundary knots
                  knots = knot_pair[which(knot_pair<1001 & knot_pair>1)], 
                  degree = input$degree, 
                  intercept = input$intercept))
    }
  })
  model_parameter <- reactive({
    basis_MLP(input = fd.input,
              output = switch(input$Output,
                              'reflection amplitude' = ref_amp, 
                              'transmission amplitude' = trans_amp),
              learning.rate= input$learning_rate,
              basis = basis(),step = input$step)
  })
  a.pred.curve <- reactive({
    bs_MLP_predict(test = matrix(fd.input[input$test,],nrow = 1),
                   n = model_parameter()$output_dimension,
                   W_1 = model_parameter()$W_1 , 
                   wt = model_parameter()$wt,
                   basis = model_parameter()$basis)
  })
  b.pred.curve <- reactive({
    bs_MLP_predict(test = matrix(c(input$a,input$b,input$t,input$r
                                    ),nrow = 1),
                   n = model_parameter()$output_dimension,
                   W_1 = model_parameter()$W_1,
                   wt = model_parameter()$wt,
                   basis = model_parameter()$basis)
  })
  output$`a1` <- renderPlot({
    ggplot(data = data.frame(freq = fd.output[,1], Abs = a.original.curve()))+
      geom_line(mapping = aes(x = freq, 
                              y = Abs)) +
      labs(x = "frequency",y = "Abs") +
      ggtitle(paste("original",input$Output,"for validation set")) +
      theme(plot.title = element_text(hjust=0.5)) +
      ylim(0,1)
    
    
    
  })
  output$basis <-renderPlot({
    matplot(1:dim(fd.output)[1],basis(),type = "l",
            xlab = "frequency",
            ylab = 'amplitude',
            main = paste('Initial Piecewise constant',input$degree,'degree','B-spline bases'))
  })
  
  output$`a2` <- renderPlot({
    ggplot(data = data.frame(freq = fd.output[,1], Abs = a.pred.curve()))+
      geom_line(mapping = aes(x = freq, 
                              y = Abs)) +
      labs(x = "frequency",y = "Abs") +
      ggtitle(paste("predicted",input$Output,"for validation set")) +
      theme(plot.title = element_text(hjust=0.5)) +
      ylim(0,1)
  })
  
  output$`b1` <- renderPlot({
    ggplot(data = data.frame(freq = fd.output[,1], Abs = b.pred.curve()))+
      geom_line(mapping = aes(x = freq, 
                              y = Abs)) +
      labs(x = "frequency",y = "Abs") +
      ggtitle(paste("predicted",input$Output,"for test set")) +
      theme(plot.title = element_text(hjust=0.5)) +
      ylim(0,1)
  })
  
  
}

# Run app ----
shinyApp(ui, server)