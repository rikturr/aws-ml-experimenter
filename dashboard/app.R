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

passwords <- str_split(Sys.getenv('PASSWORDS'), ',')[[1]]
bucket <- Sys.getenv('S3_BUCKET')
experiment <- Sys.getenv('EXPERIMENT')

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

results_kable <- function(df, row_name, col_name, metric_name, digits = 2) {
  filtered <- df %>% 
    group_by_(row_name, col_name) %>% 
    summarise_at(vars(metric_name), funs(metric = mean))
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
    mutate(metric = cell_spec(signif(metric, digits),
                        bold = ifelse(metric == max_metric, T, F),
                        color = ifelse(!is.na(abs_max), 'blue', 'black'))) %>%
    select(-max_metric, -abs_max) %>%
    spread_(col_name, 'metric')
}

ui <- fluidPage(
  shinyjs::useShinyjs(),
  navbarPage("AWS Experiments Dashboard",
             selected = "Results",
             tabPanel("Datasets",
                      mainPanel(
                      )
             ),
             
             tabPanel("Experiments",
                      mainPanel(
                        
                      )
             ),
             
             tabPanel("Results",
                      sidebarLayout(
                        sidebarPanel(
                          div(
                            id = "form",
                            # actionButton("refresh", "Refresh"),
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
                                         min = 1)
                          )
                        ),
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
    if (input$passwd %in% passwords) {
      runjs('$(".tab-content").css("visibility","visible")')
    }  else {
      runjs('$(".tab-content").css("visibility","hidden")')
    }
  })

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
  
  
}

shinyApp(ui = ui, server = server)
