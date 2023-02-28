if(sessionInfo()['basePkgs']=="dplyr" | sessionInfo()['otherPkgs']=="dplyr"){
  detach(package:dplyr, unload=TRUE)
}

if(sessionInfo()['basePkgs']=="tm" | sessionInfo()['otherPkgs']=="tm"){
  detach(package:sentiment, unload=TRUE)
  detach(package:tm, unload=TRUE)
}
#install.packages("arulesViz")
library(plyr)
library(arules)
library(arulesViz)

groceries <- read.csv("Groceries_dataset.csv")
class(groceries)

str(groceries)

sorted <- groceries[order(groceries$Member_number),]

sorted$Member_number <- as.numeric(sorted$Member_number)
str(sorted)

itemList <- ddply(sorted, c("Member_number","Date"), function(df1)paste(df1$itemDescription,collapse = ","))

head(itemList,15)

itemList$Member_number <- NULL
itemList$Date <- NULL
colnames(itemList) <- c("itemList")

write.csv(itemList,"ItemList.csv", quote = FALSE, row.names = TRUE)
head(itemList)

txn = read.transactions(file="ItemList.csv", rm.duplicates= TRUE, format="basket",sep=",",cols=1);
print(txn)
txn@itemInfo$labels <- gsub("\"","",txn@itemInfo$labels)

basket_rules <- apriori(txn, parameter = list(minlen=3, sup = 0.001, conf = 0.1, target="rules"))
print(length(basket_rules))

inspect(basket_rules[1:15])
itemFrequencyPlot(txn, topN = 15)
plot(basket_rules, jitter = 0)
plot(basket_rules, method = "grouped", control = list(k = 5))
plot(basket_rules[1:15], method="graph")
plot(basket_rules[1:15], method="paracoord")

