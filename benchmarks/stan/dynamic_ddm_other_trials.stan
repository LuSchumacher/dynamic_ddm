data {
  int<lower=1>          N;            // number of trials
  int<lower=0, upper=1> correct[N];   // correctness of response
  real<lower=0>         rt[N];        // response time
}

parameters {
  real<lower=0> v;         // drift rate
  real<lower=0> a;         // threshold
  real<lower=0> ndt;       // non-decision time                        
  
  real<lower=0> v_s;       // variation in drift rate
  real<lower=0> a_s;       // variation in threshold
  real<lower=0> ndt_s;     // variation in non-decision time 

  real          v_z[N-1];    // trial-by-trial drift rates
  real          a_z[N-1];    // trial-by-trial thresholds
  real          ndt_z[N-1];  // trial-by-trial non-decision times
}

transformed parameters {
  real<lower=0> v_t[N];    // trial-by-trial drift rates
  real<lower=0> a_t[N];    // trial-by-trial thresholds
  real<lower=0> ndt_t[N];  // trial-by-trial non-decision times
  
  v_t[1] = v;
  a_t[1] = a;
  ndt_t[1] = ndt;
  
  // implies: v_t ~ normal(v_t-1, v_s)
  for (i in 2:N){
    v_t[i] = v_t[i-1] + v_s * v_z[i-1];
    a_t[i] = a_t[i-1] + a_s * a_z[i-1];
    ndt_t[i] = ndt_t[i-1] + ndt_s * ndt_z[i-1];
  }

}


model {
  // priors
  v        ~ gamma(2.5, 2.0);
  a        ~ gamma(4.0, 3.0);
  ndt      ~ gamma(1.5, 5.0);
  
  v_s      ~ beta(1.0, 25.0);
  a_s      ~ beta(1.0, 25.0);
  ndt_s    ~ beta(1.0, 25.0);
  
  v_z      ~ std_normal();
  a_z      ~ std_normal();
  ndt_z    ~ std_normal();
  
  
  for (t in 1:N) {
    if (correct[t] == 1) {
      rt[t] ~ wiener(a_t[t], ndt_t[t], 0.5, v_t[t]);
    } else {
      rt[t] ~ wiener(a_t[t], ndt_t[t], 0.5, -v_t[t]);
    }
  }
}
