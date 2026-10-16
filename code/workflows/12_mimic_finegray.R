#!/usr/bin/env Rscript
# 独立复原事件、HC0协方差，并重拟合两版MIMIC的四个竞争风险模型。
suppressPackageStartupMessages({library(arrow);library(survival);library(cmprsk)})
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1)
out <- args[1]; tables <- file.path(out,'tables'); private <- file.path(out,'private')
write_tsv <- function(d,name) write.table(d,file.path(tables,name),sep='\t',quote=FALSE,row.names=FALSE,na='NA',fileEncoding='UTF-8')
checks <- list()
push <- function(name,error,tolerance=0) {
  passed <- is.finite(error)&&error<=tolerance
  checks[[length(checks)+1]] <<- data.frame(check=name,error=error,tolerance=tolerance,passed=passed)
  write_tsv(do.call(rbind,checks),'08_R独立数值与事件核查.tsv')
  if (!passed) stop(paste('聚合核查失败',name,error))
}
read_rows <- function(name) {
  path <- file.path(tables,name)
  if (!file.exists(path)) return(list())
  d <- read.delim(path,check.names=FALSE)
  lapply(seq_len(nrow(d)),function(i)d[i,,drop=FALSE])
}

design <- as.data.frame(read_parquet(file.path(private,'MIMIC普通回归独立设计矩阵.parquet')))
expected <- read.delim(file.path(tables,'07_数值与分离诊断.tsv'),check.names=FALSE)
ordinary <- list()
for (key in unique(design$model_key)) {
  d <- design[design$model_key==key,]
  terms <- strsplit(d$term_names[1],';',fixed=TRUE)[[1]]
  x <- as.matrix(d[,paste0('x',seq_along(terms)-1),drop=FALSE]);y<-d$y
  fit <- glm.fit(x,y,family=binomial(),control=glm.control(epsilon=1e-14,maxit=200))
  fit <- glm.fit(x,y,family=binomial(),start=fit$coefficients,control=glm.control(epsilon=1e-14,maxit=200))
  push(paste0(key,'_R独立收敛'),as.integer(!fit$converged))
  mu <- plogis(as.vector(x%*%fit$coefficients))
  bread <- solve(crossprod(x,x*(mu*(1-mu))))
  scores <- x*(y-mu)
  vc <- bread%*%crossprod(scores)%*%bread
  j <- match('P3_vs_P2',terms)
  b <- fit$coefficients[j];se<-sqrt(vc[j,j])
  e <- expected[expected$model_key==key,]
  stopifnot(nrow(e)==1)
  push(paste0(key,'_R人数及事件'),max(abs(c(nrow(d)-e$N,sum(y)-e$events))))
  push(paste0(key,'_R与Python比值比'),abs(exp(b)-e$OR),1e-7)
  push(paste0(key,'_R与Python稳健标准误'),abs(se-e$SE),1e-7)
  ordinary[[length(ordinary)+1]] <- data.frame(model_key=key,N=nrow(d),events=sum(y),OR=exp(b),SE=se,
     CI_low=exp(b-1.96*se),CI_high=exp(b+1.96*se),variance='HC0三明治稳健')
}
write_tsv(do.call(rbind,ordinary),'09_R独立普通回归.tsv')
cat('四个普通回归R交叉核验完成\n');flush.console()

events <- as.data.frame(read_parquet(file.path(private,'MIMIC修订逐人复合与竞争事件.parquet')))
counts <- read.delim(file.path(tables,'03_新旧样本及结局定义差异.tsv'),check.names=FALSE)
fg_rows <- read_rows('10_修订MIMIC_FineGray.tsv')
cross_rows <- read_rows('11_FineGray独立实现核验.tsv')
cif_rows <- read_rows('12_各表型竞争风险累积发生率.tsv')
attempt_rows <- read_rows('13_crr独立迭代诊断.tsv')

manual_cif <- function(time,status,horizon) {
  surv <- 1; cif <- 0
  for (t in sort(unique(time[status!=0 & time<=horizon]))) {
    risk <- sum(time>=t)
    d1 <- sum(time==t & status==1); d2<-sum(time==t & status==2)
    cif <- cif+surv*d1/risk
    surv <- surv*(1-(d1+d2)/risk)
  }
  cif
}

for (db in c('MIMIC-IV','MIMIC-III')) for (horizon in c(24,48)) {
  complete <- if (length(cross_rows))do.call(rbind,cross_rows) else data.frame()
  if (nrow(complete)&&any(complete$database==db&complete$horizon_hours==horizon)) {
    cat('续用已完成且独立核验通过的竞争风险：',db,horizon,'\n');flush.console();next
  }
  d <- events[events$database==db&events$horizon_hours==horizon,]
  start <- if(db=='MIMIC-IV')d$intime else d$INTIME
  end <- if(db=='MIMIC-IV')d$outtime else d$OUTTIME
  death <- if(db=='MIMIC-IV')d$deathtime else d$DEATHTIME
  # 原数据库的这些时间戳均为整秒，已由Python逐条确认无亚秒值。
  # Arrow转POSIXct在去标识化远期年份可能出现约2微秒表示误差。
  # 在原始整秒分辨率恢复秒数，避免把表示误差变成同刻事件顺序变化。
  exit_seconds <- as.numeric(difftime(end,start,units='secs'))
  death_seconds <- as.numeric(difftime(death,start,units='secs'))
  push(paste(db,horizon,'出科浮点误差小于十微秒',sep='_'),max(abs(exit_seconds-round(exit_seconds))),1e-5)
  finite_death_seconds <- death_seconds[is.finite(death_seconds)]
  push(paste(db,horizon,'死亡浮点误差小于十微秒',sep='_'),max(abs(finite_death_seconds-round(finite_death_seconds))),1e-5)
  exit <- round(exit_seconds)/3600
  death_t <- round(death_seconds)/3600
  push(paste(db,horizon,'R原始时间戳出科',sep='_'),max(abs(exit-d$exit_hours)),1e-10)
  finite <- is.finite(death_t)&is.finite(d$death_hours)
  push(paste(db,horizon,'R原始时间戳死亡',sep='_'),max(abs(death_t[finite]-d$death_hours[finite])),1e-10)
  s <- as.matrix(d[,c('first_post12_vasopressor_time','first_post12_invasive_ventilation_time','first_post12_RRT_time')])
  s[s<=12] <- NA_real_
  support <- apply(s,1,function(z)if(all(is.na(z)))Inf else min(z,na.rm=TRUE))
  death_t[is.na(death_t)|death_t<=12] <- Inf
  target <- pmin(support,death_t)
  target_first <- target<=12+horizon & target<=exit
  discharge_first <- !target_first & exit<=12+horizon
  status <- ifelse(target_first,1L,ifelse(discharge_first,2L,0L))
  time <- ifelse(target_first,target,ifelse(discharge_first,exit,12+horizon))-12
  push(paste(db,horizon,'R独立逐人事件',sep='_'),sum(status!=d$status))
  push(paste(db,horizon,'R独立逐人时间',sep='_'),max(abs(time-d$time_from_landmark)),1e-10)
  ordinary <- as.integer(target<=12+horizon)
  push(paste(db,horizon,'R普通复合原始时间复原',sep='_'),sum(ordinary!=d$ordinary_composite))
  missing_model_rows <- sum(!complete.cases(d[,c('time_from_landmark','status','phenotype','age','male','observed_dimensions_48d')]))
  push(paste(db,horizon,'完整病例无额外损失',sep='_'),missing_model_rows)
  d$p1<-as.integer(d$phenotype=='P1');d$p3<-as.integer(d$phenotype=='P3');d$p4<-as.integer(d$phenotype=='P4')
  d$age10<-d$age/10;d$coverage10<-d$observed_dimensions_48d/10;d$row_id<-seq_len(nrow(d))
  d$status_factor <- factor(d$status,levels=c(0,1,2),labels=c('censor','target','discharge'))
  fg <- finegray(Surv(time_from_landmark,status_factor)~p1+p3+p4+age10+male+coverage10+stay_id,
       data=d,etype='target',id=row_id)
  formula <- Surv(fgstart,fgstop,fgstatus)~p1+p3+p4+age10+male+coverage10+cluster(stay_id)
  fit <- coxph(formula,data=fg,weights=fgwt,robust=TRUE,ties='efron')
  b<-coef(fit)['p3'];se<-sqrt(vcov(fit)['p3','p3'])
  alternative <- coxph(formula,data=fg,weights=fgwt,robust=TRUE,ties='breslow')
  covars <- as.matrix(d[,c('p1','p3','p4','age10','male','coverage10')])
  cr <- crr(d$time_from_landmark,d$status,cov1=covars,failcode=1,cencode=0,gtol=1e-10,maxiter=100,variance=FALSE)
  attempt_rows[[length(attempt_rows)+1]]<-data.frame(database=db,horizon_hours=horizon,attempt='零初值严格容差',
         gtol=1e-10,converged=cr$converged,max_score=max(abs(cr$score)))
  if(!cr$converged) {
    cr <- crr(d$time_from_landmark,d$status,cov1=covars,failcode=1,cencode=0,
       init=as.numeric(coef(alternative)),gtol=1e-8,maxiter=500,variance=FALSE)
    attempt_rows[[length(attempt_rows)+1]]<-data.frame(database=db,horizon_hours=horizon,attempt='同一Breslow系数初值增加迭代',
         gtol=1e-8,converged=cr$converged,max_score=max(abs(cr$score)))
  }
  write_tsv(do.call(rbind,attempt_rows),'13_crr独立迭代诊断.tsv')
  push(paste(db,horizon,'独立竞争风险收敛',sep='_'),as.integer(!cr$converged))
  difference <- max(abs(coef(alternative)-cr$coef))
  push(paste(db,horizon,'独立竞争风险全系数',sep='_'),difference,2e-5)
  result <- data.frame(database=db,horizon_hours=horizon,N=nrow(d),events=sum(d$status==1),
         competing=sum(d$status==2),window_end=sum(d$status==0),SHR=exp(b),CI_low=exp(b-1.96*se),CI_high=exp(b+1.96*se),
         variance='患者记录聚类稳健',ties='Efron')
  e <- counts[counts$database==db&counts$horizon_hours==horizon,]
  push(paste(db,horizon,'预拟合人数事件对账',sep='_'),max(abs(c(result$N-e$revised_N,result$events-e$revised_FG_events,result$competing-e$revised_competing))))
  for (p in c('P1','P2','P3','P4')) {
    z <- d[d$phenotype==p,]
    ci <- cuminc(z$time_from_landmark,z$status,cencode=0)
    name <- names(ci)[grepl(' 1$',names(ci))][1]
    j <- max(which(ci[[name]]$time<=horizon));estimate<-ci[[name]]$est[j];variance<-ci[[name]]$var[j]
    independent <- manual_cif(z$time_from_landmark,z$status,horizon)
    push(paste(db,horizon,p,'独立累积发生递推',sep='_'),abs(estimate-independent),1e-10)
    cif_rows[[length(cif_rows)+1]]<-data.frame(database=db,horizon_hours=horizon,phenotype=p,N=nrow(z),
        events=sum(z$status==1),CIF_fraction=estimate,CIF_percent=100*estimate,CIF_variance=variance,
        CI_low_fraction=max(0,estimate-1.96*sqrt(variance)),CI_high_fraction=min(1,estimate+1.96*sqrt(variance)),
        interval='原历史方法：方差正态近似，界限截在0至1；非因果')
  }
  fg_rows[[length(fg_rows)+1]]<-result
  cross_rows[[length(cross_rows)+1]]<-data.frame(database=db,horizon_hours=horizon,
      max_Breslow_coefficient_difference=difference,SHR_Efron=exp(b),SHR_Breslow=exp(coef(alternative)['p3']),SHR_crr=exp(cr$coef['p3']))
  write_tsv(do.call(rbind,fg_rows),'10_修订MIMIC_FineGray.tsv')
  write_tsv(do.call(rbind,cross_rows),'11_FineGray独立实现核验.tsv')
  write_tsv(do.call(rbind,cif_rows),'12_各表型竞争风险累积发生率.tsv')
  cat('竞争风险完成且独立核验通过：',db,horizon,'小时；N=',nrow(d),'，事件=',result$events,'\n');flush.console()
}
cat('全部两版MIMIC定向模型及独立事件核查完成\n')
