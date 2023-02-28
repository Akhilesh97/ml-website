library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
#install.packages("factoextra")
library(factoextra) # clustering visualization
library(dendextend) # for comparing two dendrograms

df <- read.csv("Nutrient_Vals/nut_val_df_.csv", sep = ",", check.names=FALSE)

macro_nutrients = c('Protein (g)',
                   'Carbohydrate (g)', 'Sugars, total\n(g)', 'Fiber, total dietary (g)',
                   'Total Fat (g)')

vitamins = c('Vitamin A, RAE (mcg_RAE)', 'Vitamin B-12 (mcg)',
            'Vitamin B-12, added\n(mcg)', 'Vitamin C (mg)',
            'Vitamin D (D2 + D3) (mcg)', 'Vitamin E (alpha-tocopherol) (mg)',
            'Vitamin E, added\n(mg)', 'Vitamin K (phylloquinone) (mcg)')

minerals = c('Calcium (mg)', 'Phosphorus (mg)', 'Magnesium (mg)', 'Iron\n(mg)',
            'Zinc\n(mg)', 'Copper (mg)', 'Selenium (mcg)', 'Potassium (mg)')


fatty_acids = c('Fatty acids, total saturated (g)',
               'Fatty acids, total monounsaturated (g)',
               'Fatty acids, total polyunsaturated (g)', 'Cholesterol (mg)',
               'Retinol (mcg)')

amino_acids = c('Carotene, alpha (mcg)',
               'Carotene, beta (mcg)', 'Cryptoxanthin, beta (mcg)', 'Lycopene (mcg)',
               'Lutein + zeaxanthin (mcg)', 'Thiamin (mg)', 'Riboflavin (mg)',
               'Niacin (mg)', 'Vitamin B-6 (mg)', 'Folic acid (mcg)',
               'Folate, food (mcg)', 'Folate, DFE (mcg_DFE)', 'Folate, total (mcg)')



num_occurance <- data.frame(table(df$`WWEIA Category description`))
df_macros_1 = df[df$`WWEIA Category description`=="Vegetable dishes", c('Main food description','Protein (g)',
                   'Carbohydrate (g)', 'Sugars, total\n(g)', 'Fiber, total dietary (g)',
                   'Total Fat (g)')]
df_macros_2 = df[df$`WWEIA Category description`=="Cheese", c('Main food description','Protein (g)',
                                                                        'Carbohydrate (g)', 'Sugars, total\n(g)', 'Fiber, total dietary (g)',
                                                                        'Total Fat (g)')]
df_macros_3 = df[df$`WWEIA Category description`=="Doughnuts, sweet rolls, pastries", c('Main food description','Protein (g)',
                                                                        'Carbohydrate (g)', 'Sugars, total\n(g)', 'Fiber, total dietary (g)',
                                                                        'Total Fat (g)')]
df_macros_4 = df[df$`WWEIA Category description`=="Fish", c('Main food description','Protein (g)',
                                                                        'Carbohydrate (g)', 'Sugars, total\n(g)', 'Fiber, total dietary (g)',
                                                                        'Total Fat (g)')]

df_macros_5 = df[df$`WWEIA Category description`=="Liquor and cocktails", c('Main food description','Protein (g)',
                                                            'Carbohydrate (g)', 'Sugars, total\n(g)', 'Fiber, total dietary (g)',
                                                            'Total Fat (g)')]

tab <- table(df$`WWEIA Category description`)
tab_sorted <- tab[order(tab, decreasing = TRUE)]

barplot(tab_sorted, main="Distribution of Food Categories",
        xlab="Number of Items under the category")


plot_dendro <- function(df_macros, col_name){
  
  row.names(df_macros) = df_macros$`Main food description`
  df_macros <- df_macros[,-1]  
  
  df_macros <- scale(df_macros)
  
  
  # Dissimilarity matrix
  d <- dist(df_macros, method = "euclidean")
  
  # Hierarchical clustering using Complete Linkage
  hc1 <- hclust(d, method = "complete" )
  
  # Plot the obtained dendrogram
  nodePar <- list(lab.cex = 0.6, pch = c(NA, 19), 
                  cex = 0.7, col = "blue")
  
  png(paste(col_name,"_dend",".png"),width=1600,height=800)
  plot(hc1, cex = 0.8, hang = -1)
  dev.off()
}

plot_dendro(df_macros_1, "veggies")
plot_dendro(df_macros_2, "cheese")
plot_dendro(df_macros_3, "dougnuts")
plot_dendro(df_macros_4, "fish")
plot_dendro(df_macros_5, "liqour")
