tbl = read.table("/Users/danielesilvestro/Software/LiteRate/ADE_discrete/Weib.log",h=T)

library(scales)
transp = 0.2
par(mfrow=c(1,2))
colors = rep("gray",length(tbl$true_shape))
colors[tbl$true_shape<1] = "red"
colors[tbl$true_shape>1] = "blue"

# plot(tbl$true_shape, tbl$est_shape, pch=19,col = alpha(colors,transp),)
# abline(a=0,b=1,lty=2)
plot(log(tbl$true_shape), log(tbl$est_shape), pch=19,col = alpha(colors,transp), main= "log Shape parameter")
abline(a=0,b=1,lty=2)

plot(tbl$true_scale, tbl$est_scale, pch=19,col = alpha(colors,transp), main="Scale parameter")
abline(a=0,b=1,lty=2)

rel_err_shape =( tbl$true_shape - tbl$est_shape)/tbl$true_shape
print(mean(rel_err_shape))

rel_err_scale =abs( tbl$true_scale - tbl$est_scale)/tbl$true_scale
print(mean(rel_err_scale))




plot(log(tbl$true_longevity), log(tbl$este_longevity), pch=19,col = alpha(colors,transp))
abline(a=0,b=1,lty=2)


plot(log(rel_err_scale) ~ (tbl$true_shape))
plot(log(rel_err_scale) ~ (tbl$true_scale))


plot((rel_err_shape) ~ log(tbl$true_shape))