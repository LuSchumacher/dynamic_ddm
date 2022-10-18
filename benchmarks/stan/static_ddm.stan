data {
  int<lower=0> N;                 
  real<lower=0> rt[N];    
  int<lower=0, upper=1> correct[N];
}

parameters {
  real<lower=0> v;
  real<lower=0> a; 
  real<lower=0> ndt;
}

model {
  // Priors
  v ~ gamma(2.5, 2.0);
  a ~ gamma(4.0, 3.0);
  ndt ~ gamma(1.5, 5.0);
  
  for (n in 1:N) {
     if (correct[n] == 1) {
        rt[n] ~ wiener(a, ndt, 0.5, v);
     } 
     else {
        rt[n] ~ wiener(a, ndt, 0.5, -v);
     }
  }
}

