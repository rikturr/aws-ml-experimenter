library(shiny)
library(shinyjs)
library(aws.s3)
library(dplyr)
library(readr)
library(glue)
library(rpivotTable)
library(stringr)
library(purrr)
library(tidyr)
library(kableExtra)
library(ggplot2)
library(plotly)

passwords <- str_split(Sys.getenv('PASSWORDS'), ',')[[1]]
bucket <- Sys.getenv('S3_BUCKET')
experiment <- Sys.getenv('EXPERIMENT')

##########################################
##########################################
##### COMMENT BEFORE DEPLOY OR COMMIT!!!!
# dev <- T
##########################################
##########################################

results <- get_object(bucket = bucket, object = glue("experiments/cv_results_combined/{experiment}.csv")) %>% 
  read_csv %>% 
  mutate(classifier = case_when(startsWith(model, 'MultinomialNB') ~ 'MNB',
                                startsWith(model, 'CalibratedClassifierCV') ~ 'SVM',
                                startsWith(model, 'LogisticRegression') ~ 'LR',
                                startsWith(model, 'DecisionTreeClassifier') ~ 'DT',
                                startsWith(model, 'RandomForestClassifier') ~ 'RF',
                                startsWith(model, 'XGBClassifier') ~ 'XGB'),
         sampling_ratio = coalesce(as.character(sampling_ratio), '0'),
         reduce_pos_samples = coalesce(as.character(reduce_pos_samples), 'All'))

non_dim_cols <- c('config', 'dataset', 'fit_time', 'folds', 'instance_id',
                  'instance_type', 'model_uuid', 'reduce_pos_run', 'reduce_pos_runs', 'score_time',
                  'model')
results_cols <- colnames(results)
metrics_cols <- results_cols[startsWith(results_cols, 'train_') | startsWith(results_cols, 'test_')]
dimensions <- results_cols[!results_cols %in% non_dim_cols & !results_cols %in% metrics_cols]

mean_results <- function(df, gb, metric_name) {
  df %>% 
    group_by_at(vars(gb)) %>% 
    summarise_at(vars(metric_name), funs(metric = mean))
}

results_kable <- function(df, row_name, col_name, metric_name, digits = 2) {
  filtered <- mean_results(df, c(row_name, col_name), metric_name)
  clf_max <- df %>% 
    group_by_(row_name) %>% 
    summarise_at(vars(metric_name), funs(max_metric = max))
  abs_max <- filtered %>%
    ungroup() %>% 
    arrange(desc(metric)) %>%
    top_n(1, metric) %>%
    mutate(abs_max = T) %>%
    select_('abs_max', row_name, col_name)
  
  filtered %>% 
    inner_join(clf_max, by = row_name) %>%
    left_join(abs_max, by = c(row_name, col_name)) %>%
    rowwise() %>%
    mutate(metric = cell_spec(signif(metric, digits), 'html',
                        bold = ifelse(metric == max_metric, T, F),
                        color = ifelse(!is.na(abs_max), 'blue', 'black'))) %>%
    select(-max_metric, -abs_max) %>%
    spread_(col_name, 'metric')
}

results_plot <- function(results, facet, group, x, y, ncol = 2) {
  df <- mean_results(results, c(facet, group, x), y)
  p <- ggplot(df, aes_string(x = x, y = 'metric', color = group, group = group)) +
    geom_line() + geom_point() +
    facet_wrap(c(facet), ncol = ncol) +
    theme_light()
  p
}

ui <- fluidPage(
  shinyjs::useShinyjs(),
  navbarPage("AWS Experiments Dashboard",
             selected = "Plot",
             # tabPanel("Datasets",
             #          mainPanel(
             #          )
             # ),
             # 
             # tabPanel("Experiments",
             #          mainPanel(
             #            
             #          )
             # ),
             
             tabPanel('Plot',
                      sidebarLayout(
                        sidebarPanel(
                          actionButton("plot_refresh", "Refresh", icon = icon('refresh')),
                          selectInput('plot_facet',
                                      'Facet',
                                      choices = NULL),
                          selectInput('plot_group',
                                      'Group',
                                      choices = NULL),
                          selectInput('plot_x',
                                      'X',
                                      choices = NULL),
                          selectInput('plot_y',
                                      'Y',
                                      choices = NULL),
                          textInput('plot_height',
                                    'Height',
                                    value = '600px'
                                    ),
                          numericInput('plot_ncol',
                                       'Facet Columns',
                                       value = 2,
                                       min = 1),
                          actionButton("plot_reset", "Reset")
                        ),
                        mainPanel(
                          uiOutput('plot_ui')
                        )
                      )
             ),
             
             tabPanel("Results",
                      sidebarLayout(
                        sidebarPanel(
                          div(
                            id = "form",
                            actionButton("results_refresh", "Refresh", icon = icon('refresh')),
                            br(),
                            selectInput('results_table',
                                        'Tables',
                                        choices = NULL),
                            selectInput('results_row',
                                        'Rows',
                                        choices = NULL),
                            selectInput('results_col',
                                        'Columns',
                                        choices = NULL),
                            selectInput('results_metric',
                                        'Metric',
                                        choices = NULL),
                            numericInput('results_digits',
                                         'Significant Digits',
                                         value = 4,
                                         min = 1),
                            actionButton("results_reset", "Reset")
                          ),
                        width = 3),
                        mainPanel(
                          uiOutput('results_tables')
                        )
                      )
             ),
             tags$script(HTML("var header = $('.navbar > .container-fluid');
                       header.append('<div id=\"nav-pass\" class=\"form-group shiny-input-container\"><label for=\"passwd\">Enter password to view data:</label><input id=\"passwd\" type=\"password\" class=\"form-control\" value=\"\"/></div>');
                              console.log(header)"))
             
  )
)

server <- function(input, output, session) {
  observe({
    if (input$passwd %in% passwords | exists('dev')) {
      runjs('$(".tab-content").css("visibility","visible")')
    }  else {
      runjs('$(".tab-content").css("visibility","hidden")')
    }
  })
  
  ######## PLOT TAB ##############
  observe({
    updateSelectInput(session,
                      'plot_facet',
                      choices = dimensions,
                      selected = 'reduce_pos_samples')
  })
  
  observe({
    updateSelectInput(session,
                      'plot_group',
                      choices = dimensions,
                      selected = 'classifier')
  })
  
  observe({
    updateSelectInput(session,
                      'plot_x',
                      choices = dimensions,
                      selected = 'sampling_ratio')
  })
  
  observe({
    updateSelectInput(session,
                      'plot_y',
                      choices = metrics_cols,
                      selected = 'test_roc_auc')
  })
  
  plot_height <- reactive({
    req(input$plot_height)
    refresh <- input$plot_refresh
    input$plot_height
  })
  
  output$plot <- renderPlot({
    req(plot_height)
    results_plot(results, 
                facet = input$plot_facet,
                group = input$plot_group, 
                x = input$plot_x, 
                y = input$plot_y,
                ncol = input$plot_ncol)
    })
  
  output$plot_ui <- renderUI({
    plotOutput('plot', height = plot_height())
  })

  ###############################
  
  
  ######## RESULTS TAB ##############
  observe({
    updateSelectInput(session,
                      'results_table',
                      choices = dimensions,
                      selected = 'classifier')
  })
  
  observe({
    updateSelectInput(session,
                      'results_row',
                      choices = dimensions,
                      selected = 'sampling_ratio')
  })
  
  observe({
    updateSelectInput(session,
                      'results_col',
                      choices = dimensions,
                      selected = 'reduce_pos_samples')
  })
  
  observe({
    updateSelectInput(session,
                      'results_metric',
                      choices = metrics_cols,
                      selected = 'test_roc_auc')
  })
  
  results_tables <- reactive({
    req(input$results_table)
    refresh <- input$results_refresh
    results %>% select(input$results_table) %>% distinct() %>% pull(input$results_table)
  })

  output$results_tables <- renderUI({
           map(results_tables(),
                function(x) {
                  tableOutput(x)
                })
  })
  
  observe({
    map(results_tables(),
      function(x) {
        output[[x]] <- renderTable(results_kable(results %>% filter_(glue("{input$results_table}=='{x}'")), 
                                                 input$results_row, input$results_col, input$results_metric,
                                                 digits = input$results_digits), 
                                   caption = paste0(input$results_table, '=', x),
                                   sanitize.text.function = function(x) x,
                                   caption.placement = getOption("xtable.caption.placement", "top"))
      })
  })
  
  ###############################
  
  
}

shinyApp(ui = ui, server = server)
