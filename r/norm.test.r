norm.test <- function(input.data, alpha=0.05, pic=T) {
    if (pic == TRUE) {
        dev.new()
        par(mfrow=c(2, 1))
        qqnorm(input.data, main="QQ图")
        qqline(input.data)
        hist(input.data, freq=F, main="直方图和密度估计曲线")
        lines(density(input.data), col="blue")
        x <- c(round(min(input.data)):round(max(input.data)))
        lines(x, dnorm(x, mean(input.data), sd(input.data)), col="red")
    }
    sol <- shapiro.test(input.data)
    if (sol$p.value > alpha) {
        print(paste("success: 服从正态分布, p.value = ", sol$p.value, " > ", alpha, sep=""))
    } else {
        print(paste("error: 不服从正态分布, p.value = ", sol$p.value, " <= ", alpha, sep=""))
    }
}
