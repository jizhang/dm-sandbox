data <- read.csv("../classification/logistic_regression/ex2data1.txt",
                 header=F, col.names=c("x1", "x2", "y"))

plot(data$x1[which(data$y == 0)], data$x2[which(data$y == 0)], pch=4, xlab="x1", ylab="x2")
points(data$x1[which(data$y == 1)], data$x2[which(data$y == 1)], pch=19)

sol.glm <- glm(y ~ x1 + x2, data, family=binomial("logit"))
print(summary(sol.glm))

intercept <- coef(sol.glm)[1] / (-coef(sol.glm)[3])
slope <- coef(sol.glm)[2] / (-coef(sol.glm)[3])
abline(intercept, slope)

test.data <- data.frame(x1=c(45), x2=c(85))
npv <- predict(sol.glm, test.data)
npv <- 1 / (1 + exp(-npv))
npv <- as.factor(ifelse(npv >= 0.5, 1, 0))
test.data$y <- npv
print(test.data)
