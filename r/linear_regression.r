cost <- c(1.8, 1.2, 0.4, 0.5, 2.5, 2.5, 1.5, 1.2, 1.6, 1.0, 1.5, 0.7, 1.0, 0.8)
sales <- c(104, 68, 39, 43, 127, 134, 87, 77, 102, 65, 101, 46, 52, 33)
data <- data.frame(cost=cost, sales=sales)
plot(data, pch=16, xlab="cost促销让利费用（十万元）", ylab="sales促销销量（十万元）")
sol.lm <- lm(sales ~ cost, data)
abline(sol.lm, col="red")

sol.coef <- summary(sol.lm)$coefficients
df <- sol.lm$df.residual
alpha <- 0.05
left <- sol.coef[,1] - sol.coef[,2] * qt(1 - alpha / 2, df)
right <- sol.coef[,1] + sol.coef[,2] * qt(1 - alpha / 2, df)

new.data <- data.frame(cost=c(0.98, 0.88))
sol.pre <- predict(sol.lm, new.data, level=0.95, interval="prediction")

# multiple linear regression http://www.stat.yale.edu/Courses/1997-98/101/linmult.htm
data1 <- read.table("cereals.txt", header=T)
sol.lm1 <- lm(rating ~ fat + fiber + sugars, data1)
print(summary(sol.lm1))

# discrete variable
data2 <- data1
data2$shelf <- as.factor(data2$shelf)
sol.lm2 <- lm(rating ~ shelf, data2)
print(summary(sol.lm2))


# from lecture
library(UsingR)
y <- galton$child
x <- galton$parent
beta1 <- cor(y, x) * sd(y) / sd(x)
beta0 <- mean(y) - beta1 * mean(x)
rbind(c(beta0, beta1), coef(lm(y ~ x)))

# residual
x <- diamond$carat
y <- diamond$price
plot(x, y)
fit <- lm(y ~ x)
abline(fit)
plot(x, resid(fit))
abline(h=0)
