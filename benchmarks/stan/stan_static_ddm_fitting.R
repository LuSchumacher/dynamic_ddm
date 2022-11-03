library(tidyverse)
library(rstan)
library(reticulate)
library(bayesplot)
pd <- import("pandas")

# set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# read sim data
sim_data <- pd$read_pickle("../data/sim_data/static_dm_data_100.pkl")


ablation_fitting <- function(data) {
  # initialize data structure
  post_samples <- data.frame()
  for (sim in 1:100) {
    # subset data
    rt <- abs(sim_data$rt[sim, , 1])
    correct <- ifelse(sim_data$rt[sim, , 1] >= 0, 1, 0)
    rt_min <- min(rt)
    
    # create stan data list
    stan_data = list(
      N       = length(rt),
      correct = correct,
      rt      = rt
    )
    
    # set initial values
    init = function(chains=4) {
      L = list()
      for (c in 1:chains) {
        L[[c]]=list()
        L[[c]]$v     = rgamma(1, 2.5, 2.0)
        L[[c]]$a     = rgamma(1, 4.0, 3.0)
        L[[c]]$ndt   = rt_min * 0.9
      }
      return (L)
    }
    
    # fit model
    fit <- rstan::stan("static_ddm.stan",
                       init=init(4),
                       data=stan_data,
                       chains=4,
                       iter=2000,
                       cores=parallel::detectCores(),
                       control=list(adapt_delta=0.99,
                                    max_treedepth=15))
    
    # store data
    tmp <- as.data.frame(rstan::extract(fit))
    tmp$sim <- sim
    post_samples <- rbind(post_samples, tmp)
    write.csv(post_samples, "static_stan_posteriors_100.csv", row.names = F)
    print(paste("Simulation:", sim, "is finished..."))
    gc()
  }
  
  return(post_samples)
}

post_samples <- ablation_fitting(sim_data)

