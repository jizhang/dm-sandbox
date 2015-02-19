data <- data.frame(pre=c(113, 134, 123, 145, 137, 196, 187),
                   now=c(129, 122, 134, 149, 146, 215, 208))
ylim.max <- 550
col <- c("azure4", "brown4")
barplot(as.matrix(rbind(data$pre, data$now)),
        beside=T, ylim=c(0, ylim.max), col=col, axes=F)
axis(2)

title(main=list("本周PV趋势分析图", cex=1.5, col="red", font=3),
      sub=paste("范围：2013.4.22 - 2013.4.28", "\n", "网站板块：军事科技"),
      ylab="网站日页面浏览量PV")

text.legend <- c("上周PV", "本周PV", "PV同比增长", "PV环比增长")
col2 <- c("black", "blue")
legend("topleft", pch=c(15, 15, 16, 16), legend=text.legend, col=c(col, col2), bty="n", horiz=T)

text.x <- c("周一", "周二", "周三", "周四", "周五", "周六", "周日")
axis(1, c(2, 5, 8, 11, 14, 17, 20), labels=text.x, tick=F, cex.axis=0.75)

axis(4, at=seq(from=250, length.out=7, by=40), labels=c("-60%", "-40%", "-20%", "0", "20%", "40%", "60%"))

same.per.growth <- (data$now - data$pre) / data$pre
ring.growth <- c(NA, diff(data$now) / data$now[1:(length(data$now) - 1)])
a <- 200; b <- 370
lines(c(2, 5, 8, 11, 14, 17, 20), a * same.per.growth + b, type="b", lwd=2)
lines(c(2, 5, 8, 11, 14, 17, 20), a * ring.growth + b, type="b", lwd=2, col="blue")

for (i in 1:length(data[,1])) {
    text(3 * i - 1, a * same.per.growth[i] + b - 5, paste(round(same.per.growth[i] * 10000) / 100, "%", sep=""))
    text(3 * i - 1, a * ring.growth[i] + b + 5, paste(round(ring.growth[i] * 10000) / 100, "%", sep=""), col="blue")
}

for (i in 1:length(data[,1])) {
    text(3 * i - 1.5, data$pre[i] + 10, data$pre[i], col="deepskyblue4")
    text(3 * i - 0.5, data$now[i] + 10, data$now[i], col="deepskyblue4")
}
