data {
  int<lower=1>          N;         // number of trials
  int<lower=0, upper=1> correct;   // correctness of response
  real<lower=0>         rt;        // response time
}

parameters {
  real<lower=0> v;         // drift rate
  real<lower=0> a;         // threshold
  real<lower=0> ndt;       // non-decision time                        
}


model {
  // priors
  v        ~ gamma(2.5, 2.0);
  a        ~ gamma(4.0, 3.0);
  ndt      ~ gamma(1.5, 5.0);

  if (correct == 1) {
    rt ~ wiener(a, ndt, 0.5, v);
  } else {
    rt ~ wiener(a, ndt, 0.5, -v);
  }
}
