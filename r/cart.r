data <- read.table("cereals.txt", header=T)

library(rpart)
sol.rpart <- rpart(rating ~ fiber + sugars, data=data)

plot(sol.rpart, uniform=T, compress=T, lty=3, branch=0.7)
text(sol.rpart, all=T, digits=7, use.n=T, cex=0.9, xpd=T)

data.pred <- predict(sol.rpart)
sol.error <- sqrt(mean((data.pred - data$rating) ^ 2))

leaf <- which(sol.rpart$frame$var == "<leaf>")
point.col <- vector(length=nrow(data))
for (i in 1:nrow(data)) {
  point.col[i] <- which(leaf == sol.rpart$where[i])
}
plot(data$fiber, data$sugars, pch=16, col=point.col, xlab="fiber", ylab="sugars")

plotcp(sol.rpart, minline=T, lty=3, col=1, upper=c("size", "splits", "none"))
