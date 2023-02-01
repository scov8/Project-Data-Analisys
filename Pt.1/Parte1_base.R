######################## Plot ########################


###### LI LASCIAMO???????????????????????!!!!
dev.new()
# pdf(file="backward_cp_v20.pdf", width=20, height=10)
plot(bwd.regfit, scale = "Cp")
title("Backward selection with Cp")
# dev.off()

dev.new()
# pdf(file="forward_cp_v20.pdf", width=20, height=10)
plot(fwd.regfit, scale = "Cp")
title("Forward selection with Cp")
# dev.off()

dev.new()
# pdf(file="hybrid_cp_v20.pdf", width=20, height=10)
plot(hyb.regfit, scale = "Cp")
title("Hybrid selection with Cp")
# dev.off()

dev.new()
# pdf(file="backward_bic_v20.pdf", width=20, height=10)
plot(bwd.regfit, scale = "bic")
title("Backward selection with bic")
# dev.off()

dev.new()
# pdf(file="forward_bic_v20.pdf", width=20, height=10)
plot(fwd.regfit, scale = "bic")
title("Forward selection with bic")
# dev.off()

dev.new()
# pdf(file="hybrid_bic_v20.pdf", width=20, height=10)
plot(hyb.regfit, scale = "bic")
title("Hybrid selection with bic")
# dev.off()

dev.new()
# pdf(file="backward_adjr_v20.pdf", width=20, height=10)
plot(bwd.regfit, scale = "adjr2")
title("Backward selection with adjr2")
# dev.off()

dev.new()
# pdf(file="forward_adjr_v20.pdf", width=20, height=10)
plot(fwd.regfit, scale = "adjr2")
title("Forward selection with adjr2")
# dev.off()

dev.new()
# pdf(file="hybrid_adjr_v20.pdf", width=20, height=10)
plot(hyb.regfit, scale = "adjr2")
title("Hybrid selection with adjr2")
# dev.off()
