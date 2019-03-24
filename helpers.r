scatter <- function(data,X,Y,class){
  ggplot(data, aes(x=X, y=Y, color=class))+ 
  geom_point()+
  labs(title="Scatter plot")
}

train_valid_div <- function(data, pct, index){
    train <- data %>% sample_frac(pct)
    valid <- anti_join(data, train, by=index)
    sets  <- list("train"=train,"valid"=valid)
    return(sets)
}

cutoff_exam <- function(model, true_class, start, stop){
    t_pred <- as.data.frame(model$votes)
    colnames(t_pred) <- c("votes_0","votes_1")
    t_pred$pred_Class <- ifelse(t_pred$votes_1 > 0.5,1,0)
    
    tresholds <- seq(start,stop,by=0.05)

    results <- NULL
    for (thr in tresholds){
      t_pred$pred_Class <- ifelse(t_pred$votes_1 > thr,1,0)
      tab <- table(true_class,t_pred$pred_Class)
      rec <- tab[2,2] / (tab[2,2] + tab[2,1])
      prc <- tab[2,2] / (tab[2,2] + tab[1,2])
      f1  <- 2*(prc*rec) / (prc+rec)
      res <- c(thr, rec, prc,f1)
      results <- rbind(results,res)
    }
    
    results <- as.data.frame(results,rownames=FALSE)
    colnames(results) <- c("threshold","recall","precision","f1")

    opt_th <- results$threshold[results$f1==max(results$f1)]

    vis <- ggplot(results, aes(threshold))+
              geom_line(aes(y = recall, colour='recall'))+
              geom_line(aes(y = precision, colour='precision'))+
              geom_line(aes(y = f1, colour = 'f1'))+
              geom_vline(xintercept = opt_th, linetype="dotted")+
              labs(colour="metric", y = 'value', title = "Goodness of fit metrics for Random Forest on Train")+
              annotate(geom="text",x=opt_th-0.05, y=0.6,label=paste0("Max F-1: ",round(max(results$f1),2)," for: ",round(opt_th,2)),angle=90)
    
    f_results <- list("results"=results,"opt_th"=opt_th,"chart"=vis)
    return(f_results)
}

pred_only <- function(model,data,th){
    v_pred <- predict(model, data, type="vote")
    v_pred <- as.data.frame(v_pred)
    colnames(v_pred) <- c("votes_0","votes_1")
    v_pred$pred_Class <- ifelse(v_pred$votes_1 > th,1,0)
    g <- table(valid$Class,v_pred$pred_Class)
    return(g)
}

cv_knn_train <- function(fold_list, df, k){
    
    g <- detectCores() - 1
    registerDoParallel(g)

    no_sets = c(1:length(fold_list))
    res_round <- NULL

    r <- foreach(i=1:length(fold_list)) %dopar% {

        valid_idx <- unlist(fold_list[i])
        valid_data <- df[valid_idx,]

        train_idx <- unlist(fold_list[no_sets[no_sets!=i]])
        train_data <- df[train_idx,]

        knn <- foreach(m=1:k) %dopar% {

            #tic()
            mod <- DMwR::kNN(Class~.,train_data,valid_data,k=m)
            tab <- table(valid_data[,"Class"],mod)
            acc <- (tab[1,1]+tab[2,2])/sum(tab)
            rec <- tab[2,2] / (tab[2,1] + tab[2,2])
            prc <- tab[2,2] / (tab[1,2] + tab[2,2])
            f1  <- 2 * (prc*rec) / (prc+rec)

            res <- c("round"=i,"k"=m,"acc"=acc,"rec"=rec,"prc"=prc,"f1"=f1)
            #msg <- (paste0("Cross round: ",i," n:",m," time:","\n"))
            #sink("log.txt", append=TRUE)
            #cat(msg)
            res_round <- c(res_round,res)
        }
    }
return(r)
}

cv_xgb <- function(eta_lvs, dep_lvs, data, n, round){
    result <- NULL
    for(e in eta_levels) {
            for(d in dep_levels) {
        s <- xgb.cv(params = list("objective"="reg:linear","eta"=e,"max_depth"=d,nthread=7),
                    data = data,
                    nround=round,
                    metrics = list("rmse"),
                    nfold = n)
        res <- list("depth"=d, "eta"=e, "evals"=s$evaluation_log)
        result <- c(result, res)
      }}
    return(result)
}