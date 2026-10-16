reconstruct <- function(f,scenario,horizon) {
  end <- 720+60*horizon; n <- nrow(f)
  composite <- scenario %in% c('FG0','FG_CI','FG_CH')
  hospital <- scenario %in% c('FG_H','FG_HE','FG_CH')
  status <- rep(0L,n); time <- rep(horizon,n)
  available <- rep(TRUE,n)
  if (composite) available <- f$support_free_at_12h
  status[!available] <- 6L; time[!available] <- NA_real_
  boundary <- !is.finite(f$unitdischargeoffset)|f$unitdischargeoffset<=720
  status[available&boundary] <- 7L; time[available&boundary] <- NA_real_
  available <- available&!boundary
  if (scenario!='FG0') {
    unknown <- !(f$unit_status %in% c('ALIVE','EXPIRED'))|!(f$hospital_status %in% c('ALIVE','EXPIRED'))
    status[available&unknown] <- 8L; time[available&unknown] <- NA_real_
    available <- available&!unknown
    status[available&f$core_conflict] <- 9L; time[available&f$core_conflict] <- NA_real_
    available <- available&!f$core_conflict
  }
  if (scenario=='FG0') {
    death <- ifelse(f$hospital_status=='EXPIRED',f$hospitaldischargeoffset,Inf)
    exit <- f$unitdischargeoffset; live <- rep(TRUE,n)
  } else if (hospital) {
    death <- ifelse(f$hospital_status=='EXPIRED',f$hospitaldischargeoffset,Inf)
    if (scenario=='FG_HE') death[f$unit_status=='EXPIRED'] <- f$unitdischargeoffset[f$unit_status=='EXPIRED']
    exit <- f$hospitaldischargeoffset; live <- f$hospital_status=='ALIVE'
  } else {
    death <- ifelse(f$unit_status=='EXPIRED',f$unitdischargeoffset,Inf)
    exit <- f$unitdischargeoffset; live <- f$unit_status=='ALIVE'
  }
  bad <- !is.finite(exit)|exit<=720|death<=720
  status[available&bad] <- 9L; time[available&bad] <- NA_real_
  available <- available&!bad
  first <- death
  if (composite) {
    support <- as.matrix(f[,c('first_post12_vasopressor_time','first_post12_invasive_ventilation_time','first_post12_RRT_time')])
    support[!is.finite(support)|support<=720] <- Inf
    if (scenario!='FG0') support[support>f$unitdischargeoffset] <- Inf
    first <- pmin(first,apply(support,1,min))
  }
  event <- available&first<=end&first<=exit
  competing <- available&!event&live&exit<=end
  uncertain <- available&!event&!competing&exit<=end
  status[event] <- 1L; time[event] <- (first[event]-720)/60
  status[competing] <- 2L; time[competing] <- (exit[competing]-720)/60
  status[uncertain] <- 9L; time[uncertain] <- NA_real_
  data.frame(stay_id=f$stay_id,status=status,time=time)
}
