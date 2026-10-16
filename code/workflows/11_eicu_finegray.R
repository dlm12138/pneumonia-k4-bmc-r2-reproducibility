#!/usr/bin/env Rscript
# R独立复算常规回归、逐人事件，并实际重拟合修订样本的Fine–Gray。
suppressPackageStartupMessages({library(arrow); library(survival); library(cmprsk)})
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2)
out <- args[1]; project <- args[2]
tables <- file.path(out,'tables'); private <- file.path(out,'private')
write_tsv <- function(x,name) write.table(x,file.path(tables,name),sep='\t',quote=FALSE,row.names=FALSE,na='NA',fileEncoding='UTF-8')
as_rows <- function(name) {
  p <- file.path(tables,name)
  if (!file.exists(p)) return(list())
  d <- read.delim(p,check.names=FALSE)
  lapply(seq_len(nrow(d)),function(i) d[i,,drop=FALSE])
}
checks <- as_rows('13_R独立数值与事件核查.tsv')
push <- function(name,error,tolerance=0) {
  checks[[length(checks)+1]] <<- data.frame(check=name,error=error,tolerance=tolerance,
      passed=is.finite(error)&&error<=tolerance)
  write_tsv(do.call(rbind,checks),'13_R独立数值与事件核查.tsv')
  if (!is.finite(error)||error>tolerance) stop(paste(name,error))
}

designs <- as.data.frame(read_parquet(file.path(private,'修订回归独立核查矩阵.parquet')))
expected <- read.delim(file.path(tables,'02_回归数值与稀疏性诊断.tsv'),check.names=FALSE)
reg_rows <- list()
for (key in unique(designs$model_key)) {
  d <- designs[designs$model_key==key,]
  terms <- strsplit(d$term_names[1],';',fixed=TRUE)[[1]]
  x <- as.matrix(d[,paste0('x',seq_along(terms)-1),drop=FALSE]); y <- d$y
  fit <- glm.fit(x,y,family=binomial(),control=glm.control(epsilon=1e-14,maxit=200))
  fit <- glm.fit(x,y,family=binomial(),start=fit$coefficients,control=glm.control(epsilon=1e-14,maxit=200))
  stopifnot(fit$converged)
  mu <- plogis(as.vector(x %*% fit$coefficients))
  bread <- solve(crossprod(x,x * (mu*(1-mu))))
  scores <- x * (y-mu)
  if (startsWith(key,'eICU_')) {
    grouped <- rowsum(scores,d$cluster,reorder=FALSE)
    g <- nrow(grouped); n <- nrow(x); k <- ncol(x)
    vc <- bread %*% crossprod(grouped) %*% bread * g/(g-1)*(n-1)/(n-k)
  } else vc <- bread %*% crossprod(scores) %*% bread
  index <- match(if ('fixed_HRRR_rule' %in% terms) 'fixed_HRRR_rule' else 'P3_vs_P2',terms)
  b <- fit$coefficients[index]; se <- sqrt(vc[index,index])
  lookup <- paste(expected$database,expected$model,sep='_')==key
  e <- expected[lookup,]
  stopifnot(nrow(e)==1)
  push(paste0(key,'_R人数与事件'),max(abs(c(nrow(d)-e$N,sum(y)-e$events))))
  push(paste0(key,'_R与Python_OR'),abs(exp(b)-e$OR),1e-7)
  push(paste0(key,'_R与Python_SE'),abs(se-e$SE),1e-7)
  reg_rows[[length(reg_rows)+1]] <- data.frame(model_key=key,N=nrow(d),events=sum(y),OR=exp(b),SE=se,
      CI_low=exp(b-1.96*se),CI_high=exp(b+1.96*se))
}
write_tsv(do.call(rbind,reg_rows),'13_R独立常规模型.tsv')
cat('修订常规模型R交叉核查完成；准完全分离模型不作常规推断\n');flush.console()

# 仅读取并执行已审定独立事件重建函数，不运行历史R脚本的拟合或写文件代码。
historical_script <- file.path(project,'code/core/eicu_event_reconstruction.R')
expressions <- parse(file=historical_script)
selected <- Filter(function(e) is.call(e) && identical(e[[1]],as.name('<-')) &&
                  is.symbol(e[[2]]) && identical(as.character(e[[2]]),'reconstruct'),as.list(expressions))
stopifnot(length(selected)==1); eval(selected[[1]])
evidence <- as.data.frame(read_parquet(file.path(private,'修订eICU逐人原始结局证据.parquet')))
events <- as.data.frame(read_parquet(file.path(private,'修订eICU竞争风险逐人事件.parquet')))
fg_rows <- as_rows('13_修订FineGray实际重拟合.tsv')
cif_rows <- as_rows('13_修订各表型累积事件发生.tsv')
cross_rows <- as_rows('13_修订FineGray独立实现核查.tsv')
attempts <- as_rows('13_独立crr迭代诊断.tsv')
drop_key <- function(rows,scenario,horizon) Filter(function(r) !(r$scenario==scenario&&r$horizon_hours==horizon),rows)
for (scenario in unique(events$scenario)) for (horizon in c(24,48)) {
  done <- if (length(cross_rows)) do.call(rbind,cross_rows) else data.frame()
  if (nrow(done)&&any(done$scenario==scenario&done$horizon_hours==horizon)) {
    cat('续用已完成且独立核查通过的Fine–Gray：',scenario,horizon,'\n');flush.console();next
  }
  actual <- reconstruct(evidence,scenario,horizon)
  d <- events[events$scenario==scenario&events$horizon_hours==horizon,]
  ix <- match(d$stay_id,actual$stay_id)
  push(paste(scenario,horizon,'逐人状态',sep='_'),sum(d$status!=actual$status[ix]))
  push(paste(scenario,horizon,'逐人时间缺失',sep='_'),sum(is.na(d$time_from_landmark)!=is.na(actual$time[ix])))
  finite <- is.finite(d$time_from_landmark)&is.finite(actual$time[ix])
  push(paste(scenario,horizon,'逐人时间',sep='_'),max(abs(d$time_from_landmark[finite]-actual$time[ix][finite])),1e-10)
  d <- d[d$status %in% c(0,1,2),]
  d <- d[complete.cases(d[,c('time_from_landmark','phenotype','age','male','observed_dimensions_48d','hospital_id')]),]
  d$p1 <- as.integer(d$phenotype=='P1');d$p3 <- as.integer(d$phenotype=='P3');d$p4 <- as.integer(d$phenotype=='P4')
  d$age10 <- d$age/10;d$coverage10 <- d$observed_dimensions_48d/10;d$row_id <- seq_len(nrow(d))
  d$status_factor <- factor(d$status,levels=c(0,1,2),labels=c('censor','target','discharge'))
  fg <- finegray(Surv(time_from_landmark,status_factor)~p1+p3+p4+age10+male+coverage10+hospital_id+stay_id,
                 data=d,etype='target',id=row_id)
  formula <- Surv(fgstart,fgstop,fgstatus)~p1+p3+p4+age10+male+coverage10+cluster(hospital_id)
  fit <- coxph(formula,data=fg,weights=fgwt,robust=TRUE,ties='efron')
  b <- coef(fit)['p3'];se <- sqrt(vcov(fit)['p3','p3'])
  result <- data.frame(scenario=scenario,horizon_hours=horizon,N=nrow(d),events=sum(d$status==1),
       competing=sum(d$status==2),window_end=sum(d$status==0),SHR=exp(b),CI_low=exp(b-1.96*se),CI_high=exp(b+1.96*se),
       hospitals=length(unique(d$hospital_id)),variance='医院聚类稳健',ties='Efron')
  push(paste(scenario,horizon,'总数对账',sep='_'),abs(result$events+result$competing+result$window_end-result$N))
  fg_rows <- drop_key(fg_rows,scenario,horizon)
  fg_rows[[length(fg_rows)+1]] <- result
  write_tsv(do.call(rbind,fg_rows),'13_修订FineGray实际重拟合.tsv')
  alternative <- coxph(formula,data=fg,weights=fgwt,robust=TRUE,ties='breslow')
  cr <- crr(d$time_from_landmark,d$status,cov1=as.matrix(d[,c('p1','p3','p4','age10','male','coverage10')]),
            failcode=1,cencode=0,gtol=1e-10,maxiter=100,variance=FALSE)
  attempts[[length(attempts)+1]] <- data.frame(scenario=scenario,horizon_hours=horizon,
       attempt='零初值严格容差',gtol=1e-10,converged=cr$converged,max_score=max(abs(cr$score)))
  if (!cr$converged) {
    cr <- crr(d$time_from_landmark,d$status,cov1=as.matrix(d[,c('p1','p3','p4','age10','male','coverage10')]),
              failcode=1,cencode=0,init=as.numeric(coef(alternative)),gtol=1e-8,maxiter=500,variance=FALSE)
    attempts[[length(attempts)+1]] <- data.frame(scenario=scenario,horizon_hours=horizon,
         attempt='同一Breslow模型初值增加迭代',gtol=1e-8,converged=cr$converged,max_score=max(abs(cr$score)))
  }
  write_tsv(do.call(rbind,attempts),'13_独立crr迭代诊断.tsv')
  stopifnot(cr$converged)
  difference <- max(abs(coef(alternative)-cr$coef))
  push(paste(scenario,horizon,'独立FineGray系数',sep='_'),difference,2e-5)
  cross_rows[[length(cross_rows)+1]] <- data.frame(scenario=scenario,horizon_hours=horizon,
       max_Breslow_coefficient_difference=difference,SHR_Efron=exp(b),
       SHR_Breslow=exp(coef(alternative)['p3']),SHR_crr=exp(cr$coef['p3']))
  write_tsv(do.call(rbind,cross_rows),'13_修订FineGray独立实现核查.tsv')
  cif_rows <- drop_key(cif_rows,scenario,horizon)
  for (p in c('P1','P2','P3','P4')) {
    z <- d[d$phenotype==p,]
    ci <- cuminc(z$time_from_landmark,z$status,cencode=0)
    name <- names(ci)[grepl(' 1$',names(ci))][1]
    estimate <- variance <- NA_real_
    if (!is.na(name)) {j <- max(which(ci[[name]]$time<=horizon));estimate <- ci[[name]]$est[j];variance <- ci[[name]]$var[j]}
    cif_rows[[length(cif_rows)+1]] <- data.frame(scenario=scenario,horizon_hours=horizon,phenotype=p,
        N=nrow(z),events=sum(z$status==1),CIF=estimate,CIF_variance=variance)
  }
  write_tsv(do.call(rbind,cif_rows),'13_修订各表型累积事件发生.tsv')
  cat('修订Fine–Gray完成：',scenario,horizon,'小时，N=',nrow(d),'，目标事件=',sum(d$status==1),'\n');flush.console()
}
cat('修订R常规模型、事件分类和竞争风险重拟合完成\n')
