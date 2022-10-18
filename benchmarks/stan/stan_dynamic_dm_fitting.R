library(tidyverse)
library(rstan)
library(reticulate)
library(bayesplot)
pd <- import("pandas")
np <- import("numpy")

# set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# read simulation data
sim_data <- pd$read_pickle("../data/sim_data/static_dm_data_100.pkl")


# initial value function
init = function(chains=4, rt) {
  L = list()
  for (c in 1:chains) {
    L[[c]]=list()
    
    L[[c]]$v     = runif(1, 0.5, 4.0)
    L[[c]]$a     = runif(1, 1.0, 1.5)
    L[[c]]$ndt   = runif(1, 0.01, 0.01)
    L[[c]]$v_s   = runif(1, 0.001, 0.01)
    L[[c]]$a_s   = runif(1, 0.001, 0.01)
    L[[c]]$ndt_s = runif(1, 0.001, 0.001)
    
    L[[c]]$v_t   = runif(length(rt), 0.5, 4.0)
    L[[c]]$a_t   = runif(length(rt), 1.0, 1.5)
    L[[c]]$ndt_t = runif(length(rt), 0.01, min(rt)*0.5)
    
  }
  return (L)
}

#-------------------------------------------------------------------
#-------------------------------------------------------------------
# iterate over simulations
model_first <- rstan::stan_model("dynamic_dm_first_trial.stan")
model_other <- rstan::stan_model("dynamic_dm_newForm_nC.stan")
for (sim in 1:1){
  start_time <- Sys.time()
  # subset data
  rt <- abs(sim_data$rt[sim, , 1])
  correct <- ifelse(sim_data$rt[sim, , 1] >= 0, 1, 0)
  
  post_samples <- data.frame()
  
  
  # iterate over time
  for (t in 1:100){
    # subset data
    rt_tmp <- rt[1:t]
    correct_tmp <- correct[1:t]
    
    # create stan data list
    stan_data = list(
      N       = length(rt_tmp),
      correct = correct_tmp,
      rt      = rt_tmp
    )
    
    
    if (t == 1){
      # fit model
      fit <- rstan::sampling(model_first,
                         init=init(4, rt_tmp),
                         data=stan_data,
                         chains=4,
                         iter=2000,
                         cores=parallel::detectCores(),
                         control=list(adapt_delta=0.99,
                                      max_treedepth=15))
      
      # extract and store first posterior dists
      samples <- as.data.frame(rstan::extract(fit)) %>%
        select(v, a, ndt)
      
      tmp_df <- data.frame(sim=rep(sim, 4000), time=rep(t, 4000),
                           v=samples$v, a=samples$a, ndt=samples$ndt)
      
      post_samples <- rbind(post_samples, tmp_df)
      
      
    }else{
      # fit model
      fit <- rstan::sampling(model_other,
                      init=init(4, rt_tmp),
                      data=stan_data,
                      chains=4,
                      iter=2000,
                      cores=parallel::detectCores(),
                      control=list(adapt_delta=0.99,
                                   max_treedepth=15))
      
      # select posterior dist from last time step only
      samples = as.data.frame(rstan::extract(fit)) %>%
        select(paste("v_t.", t, sep = ""),
               paste("a_t.", t, sep = ""),
               paste("ndt_t.", t, sep = ""))
      
      # store posteriors
      tmp_df <- data.frame(sim=rep(sim, 4000), time=rep(t, 4000),
                           v=samples[ , 1], a=samples[ , 2], ndt=samples[ , 3])
      
      post_samples <- rbind(post_samples, tmp_df)
      
    }
    
    print(paste("Estimation for time", t, "is finished :)"))
    
  }
  write.csv(post_samples, paste("dynamic_stan_post_samples_", sim, ".csv" ,sep=""))
  rm(post_samples)
  gc()
  end_time <- Sys.time()
}

duration <- end_time - start_time