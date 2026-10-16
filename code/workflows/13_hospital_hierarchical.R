#!/usr/bin/env Rscript
# 保留原bobyqa公式；额外用独立优化器核查，不按显著性选择随机结构。
suppressPackageStartupMessages({library(arrow);library(lme4);library(sandwich)})
args<-commandArgs(trailingOnly=TRUE);stopifnot(length(args)==1);out<-args[1]
tables<-file.path(out,'tables');private<-file.path(out,'private')
write_tsv<-function(d,name)write.table(d,file.path(tables,name),sep='\t',row.names=FALSE,quote=FALSE,na='NA',fileEncoding='UTF-8')
d<-as.data.frame(read_parquet(file.path(private,'修订医院P2P3输入.parquet')))
required<-c('death_24h','p3','age10','male','coverage10','hospital_id')
d<-d[complete.cases(d[,required]),required];d$hospital_id<-factor(d$hospital_id)
oracle<-read.delim(file.path(tables,'19_医院固定效应R核对输入.tsv'))
fixed<-glm(death_24h~p3+age10+male+coverage10,data=d,family=binomial(),control=glm.control(epsilon=1e-12,maxit=200))
vc<-vcovCL(fixed,cluster=d$hospital_id,type='HC1',cadjust=TRUE)
checks<-list()
push<-function(name,error,tolerance){
 checks[[length(checks)+1]]<<-data.frame(check=name,error=error,tolerance=tolerance,passed=is.finite(error)&&error<=tolerance)
 write_tsv(do.call(rbind,checks),'19_医院层R独立核查.tsv')
 if(!is.finite(error)||error>tolerance)stop(name)
}
push('R固定回归人数',abs(nrow(d)-oracle$N),0)
push('R固定回归事件',abs(sum(d$death_24h)-oracle$events),0)
push('R固定回归系数',abs(coef(fixed)['p3']-oracle$beta),1e-7)
push('R固定回归聚类SE',abs(sqrt(vc['p3','p3'])-oracle$SE),1e-7)
rows<-list();diagnostics<-list()
for(name in c('random_intercept','random_intercept_and_p3_slope')){
 formula<-if(name=='random_intercept')death_24h~p3+age10+male+coverage10+(1|hospital_id) else death_24h~p3+age10+male+coverage10+(1+p3|hospital_id)
 controls<-glmerControl(optimizer='bobyqa',optCtrl=list(maxfun=200000),check.conv.singular='ignore')
 fit<-glmer(formula,data=d,family=binomial(),control=controls)
 cf<-summary(fit)$coefficients;b<-cf['p3','Estimate'];se<-cf['p3','Std. Error'];variance<-as.data.frame(VarCorr(fit))
 alternate<-glmer(formula,data=d,family=binomial(),start=list(theta=getME(fit,'theta'),fixef=fixef(fit)),
       control=glmerControl(optimizer='Nelder_Mead',optCtrl=list(maxfun=200000),check.conv.singular='ignore'))
 difference<-max(abs(fixef(fit)-fixef(alternate)));ll_difference<-abs(as.numeric(logLik(fit)-logLik(alternate)))
 diagnostics[[length(diagnostics)+1]]<-data.frame(model=name,optimizer='bobyqa',converged=is.null(fit@optinfo$conv$lme4$messages),
    singular=isSingular(fit,tol=1e-4),max_abs_gradient=max(abs(fit@optinfo$derivs$gradient)),
    alternate_converged=is.null(alternate@optinfo$conv$lme4$messages),alternate_singular=isSingular(alternate,tol=1e-4),
    alternate_max_abs_gradient=max(abs(alternate@optinfo$derivs$gradient)),fixed_coefficient_max_difference=difference,loglik_difference=ll_difference)
 write_tsv(do.call(rbind,diagnostics),'19_随机效应优化器独立诊断.tsv')
 push(paste(name,'独立优化器固定系数'),difference,1e-3)
 push(paste(name,'独立优化器似然'),ll_difference,1e-5)
 rows[[length(rows)+1]]<-data.frame(model=name,N=nrow(d),events=sum(d$death_24h),hospitals=nlevels(d$hospital_id),
   OR=exp(b),CI_low=exp(b-1.96*se),CI_high=exp(b+1.96*se),
   random_intercept_SD=variance$sdcor[variance$var1=='(Intercept)'&is.na(variance$var2)][1],
   random_slope_SD=if(name=='random_intercept')NA_real_ else variance$sdcor[variance$var1=='p3'&is.na(variance$var2)][1],
   converged=is.null(fit@optinfo$conv$lme4$messages),singular=isSingular(fit,tol=1e-4),
   fixed_formula='death_24h ~ p3 + age10 + male + coverage10',nAGQ=1,optimizer='bobyqa',
   inference='条件关联OR；Wald区间；不等同因果效应')
 write_tsv(do.call(rbind,rows),'19_修订医院随机效应模型.tsv')
 cat(name,'完成\n');flush.console()
}
cat('医院随机效应与独立优化器核查完成\n')
