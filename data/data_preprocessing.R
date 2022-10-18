library(tidyverse)
library(magrittr)

df <- read_csv("/users/lukas/desktop/dynamic_model_project/data/raw_data_lexical_decision.csv")

df %<>% 
  filter(vld == 1) %>% 
  rename(session = mzp,
         block = blk,
         trial = trl,
         stim_type = typ,
         stim = rto,
         resp = rsp,
         rt = RT,
         id = code) %>% 
  select(id, sex, age, session, block, trial,
         stim_type, stim, resp, acc, rt) %>% 
  mutate(rt = rt/1000,
         resp = ifelse(resp == 2, 0, 1),
         stim_type = stim_type + 1)

# difficult --> 2, 3
# word --> 1, 2
df$difficult <- NA
df$word <- NA
for (i in 1:nrow(df)){
  # difficult
  if (df$stim_type[i] == 2 | df$stim_type[i] == 3){
    df$difficult[i] <- 1
  } else{
    df$difficult[i] <- 2
  }
  
  # word
  if (df$stim_type[i] == 1 | df$stim_type[i] == 2){
    df$word[i] <- 1
  } else{
    df$word[i] <- 2
  }
}


# reassign subject id
nr <- 0
df$id[1] <- 1
for (i in 2:length(df$id)){
  if (df$id[i] != df$id[i - 1]){
    nr <- nr + 1
    df$id[df$id == df$id[i]] <- nr
  }
}

# order dataframe
df <- df %>%
  mutate(id=as.numeric(factor(df$id))) %>% 
  arrange(id, session, block, trial)

write.csv(df, "/users/lukas/documents/github/dynamic_models/data/data_lexical_decision.csv", row.names = F)

# stim_type
#----------------------------------------------------------------------------#
df <- read.csv("/users/lukas/documents/github/dynamic_models/data/data_lexical_decision.csv")

stim_type_id_1 <- df %>%
  filter(id == 1) %>% 
  select(stim_type)

stim_type_id_2 <- df %>%
  filter(id == 2) %>% 
  select(stim_type)


sumsum <- df %>% 
  group_by(id,
           stim_type) %>% 
  summarise(acc = mean(acc),
            rt_median = median(rt))

# rt's
sumsum %>% 
  ggplot(aes(x = as.factor(stim_type),
             y = rt_median,
             group = as.factor(stim_type))) +
  geom_point(alpha = 0.8) +
  geom_line(aes(group = 1),
            linetype = "dashed") +
  facet_grid(~id) +
  scale_y_continuous(breaks = c(seq(0.4, 0.8, 0.1)),
                     limits = c(0.4, 0.8)) +
  theme_light()

# accuracy
sumsum %>% 
  ggplot(aes(x = as.factor(stim_type),
             y = acc,
             group = as.factor(stim_type))) +
  geom_point(alpha = 0.8) +
  geom_line(aes(group = 1),
            linetype = "dashed") +
  facet_grid(~id) +
  scale_y_continuous(breaks = c(seq(0.7, 1.0, 0.1)),
                     limits = c(0.7, 1.0)) +
  theme_light()

# post error
#----------------------------------------------------------------------------#
df$post_error <- NA
error_trial_index <- which(df$acc == 0)
for (i in error_trial_index){
  if (df$acc[i-1] == 1 && df$acc[i-2] == 1 &&
      df$acc[i+1] == 1 && df$acc[i+2] == 1){
    df$post_error[i] <- "error"
    df$post_error[i+1] <- "post_error"
    df$post_error[i+2] <- "post_error"
    # df$post_error[i+3] <- "post_error"
    # df$post_error[i+4] <- "post_error"
    df$post_error[i-1] <- "pre_error"
    df$post_error[i-2] <- "pre_error"
    # df$post_error[i-3] <- "pre_error"
    # df$post_error[i-4] <- "pre_error"

  }
}

df %>% 
  group_by(post_error) %>% 
  summarise(rt_mean = mean(rt))


df %>%
  group_by(acc) %>% 
  summarise(rt_mean = mean(rt))

# response repetition
#----------------------------------------------------------------------------#

















