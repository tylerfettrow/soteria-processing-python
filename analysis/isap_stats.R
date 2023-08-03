library(lme4)
library(emmeans)
library(lmerTest)
library(plyr)
library(dplyr)
library(ggplot2)
library(huxtable)
library(tibble)
library(tidytable)
library(reshape2)
library(dplyr)
library(huxtable)

setwd("C:/Users/tfettrow/Documents/GitHub/soteria_code_appdat/soteria_code/analysis")
dat <- read.csv(file = 'Analysis_total_eventSmarteye_metric_dataframe.csv',fileEncoding="UTF-8-BOM")

lm_model <- lmer(gaze_variance ~ factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova_lm_model_gaze_variance_event1_delta <- anova(lm_model)
anova_lm_model_gaze_variance_event1_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance <- emmeans(rg_lm_model, "crew")
confint_lm_model_gaze_variance
ci <- hux(confint_lm_model_gaze_variance)
crew = t(t(ci$crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$asymp.LCL))
upper = t(t(ci$asymp.UCL))
confint_df = data.frame(crew, emmean, lower, upper)
figure_name = paste0("gaze_variance",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew, levels = crew), y = emmean)) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,.5)) +
  theme_minimal() +
  ggtitle("Gaze Variance") +
  labs(x="", y="Gaze Variance") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)

lm_model <- lmer(gaze_vel_avg ~ factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_avg <- anova(lm_model)
anova_lm_model_gaze_vel_avg
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance <- emmeans(rg_lm_model, "crew")
confint_lm_model_gaze_variance
ci <- hux(confint_lm_model_gaze_variance)
crew = t(t(ci$crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$asymp.LCL))
upper = t(t(ci$asymp.UCL))
confint_df = data.frame(crew, emmean, lower, upper)
figure_name = paste0("gaze_vel_avg",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew, levels = crew), y = emmean)) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,500)) +
  theme_minimal() +
  ggtitle("Gaze Vel") +
  labs(x="", y="Gaze Velocity (deg/sec)") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)


lm_model <- lmer(gaze_vel_std ~ factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std <- anova(lm_model)
anova_lm_model_gaze_vel_std
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance <- emmeans(rg_lm_model, "crew")
confint_lm_model_gaze_variance
ci <- hux(confint_lm_model_gaze_variance)
crew = t(t(ci$crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$asymp.LCL))
upper = t(t(ci$asymp.UCL))
confint_df = data.frame(crew, emmean, lower, upper)
figure_name = paste0("gaze_vel_std",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew, levels = crew), y = emmean)) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,300)) +
  theme_minimal() +
  ggtitle("Gaze Vel Std") +
  labs(x="", y="gaze_vel_std") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)



lm_model <- lmer(headHeading_avg ~ factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std <- anova(lm_model)
anova_lm_model_gaze_vel_std
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance <- emmeans(rg_lm_model, "crew")
confint_lm_model_gaze_variance
ci <- hux(confint_lm_model_gaze_variance)
crew = t(t(ci$crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$asymp.LCL))
upper = t(t(ci$asymp.UCL))
confint_df = data.frame(crew, emmean, lower, upper)
figure_name = paste0("headHeading_avg",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew, levels = crew), y = emmean)) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,40)) +
  theme_minimal() +
  ggtitle("Head Heading Velocity") +
  labs(x="", y="Head Heading Velocity") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)


lm_model <- lmer(headHeading_std ~ factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std <- anova(lm_model)
anova_lm_model_gaze_vel_std
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance <- emmeans(rg_lm_model, "crew")
confint_lm_model_gaze_variance
ci <- hux(confint_lm_model_gaze_variance)
crew = t(t(ci$crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$asymp.LCL))
upper = t(t(ci$asymp.UCL))
confint_df = data.frame(crew, emmean, lower, upper)
figure_name = paste0("headHeading_std",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew, levels = crew), y = emmean)) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,120)) +
  theme_minimal() +
  ggtitle("Head Heading rate std") +
  labs(x="", y="headHeading_std") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)


lm_model <- lmer(pupilD_avg ~ factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std <- anova(lm_model)
anova_lm_model_gaze_vel_std
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance <- emmeans(rg_lm_model, "crew")
confint_lm_model_gaze_variance
ci <- hux(confint_lm_model_gaze_variance)
crew = t(t(ci$crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$asymp.LCL))
upper = t(t(ci$asymp.UCL))
confint_df = data.frame(crew, emmean, lower, upper)
figure_name = paste0("pupilD_avg",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew, levels = crew), y = emmean)) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,.005)) +
  theme_minimal() +
  ggtitle("Pupil Diameter") +
  labs(x="", y="Pupil Diameter (m)") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)

lm_model <- lmer(pupilD_std ~ factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std <- anova(lm_model)
anova_lm_model_gaze_vel_std
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance <- emmeans(rg_lm_model, "crew")
confint_lm_model_gaze_variance
ci <- hux(confint_lm_model_gaze_variance)
crew = t(t(ci$crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$asymp.LCL))
upper = t(t(ci$asymp.UCL))
confint_df = data.frame(crew, emmean, lower, upper)
figure_name = paste0("pupilD_std",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew, levels = crew), y = emmean)) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,.0005)) +
  theme_minimal() +
  ggtitle("Pupil Diameter Std") +
  labs(x="", y="pupilD_std") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)









dat <- read.csv(file = 'total_eventSmarteye_metric_dataframe.csv',fileEncoding="UTF-8-BOM")


rpsa_factor_levels <- c("high", "moderate", "low")

dat$gaze_variance_event1_delta <- dat$gaze_variance_event1 - dat$gaze_variance_control
dat$gaze_variance_event2_delta <- dat$gaze_variance_event2 - dat$gaze_variance_control
dat$gaze_vel_avg_event1_delta <- dat$gaze_vel_avg_event1 - dat$gaze_vel_avg_control
dat$gaze_vel_avg_event2_delta <- dat$gaze_vel_avg_event2 - dat$gaze_vel_avg_control
dat$gaze_vel_std_event1_delta <- dat$gaze_vel_std_event1 - dat$gaze_vel_std_control
dat$gaze_vel_std_event2_delta <- dat$gaze_vel_std_event2 - dat$gaze_vel_std_control
dat$headHeading_avg_event1_delta <- dat$headHeading_avg_event1 - dat$headHeading_avg_control
dat$headHeading_avg_event2_delta <- dat$headHeading_avg_event2 - dat$headHeading_avg_control
dat$headHeading_std_event1_delta <- dat$headHeading_std_event1 - dat$headHeading_std_control
dat$headHeading_std_event2_delta <- dat$headHeading_std_event2 - dat$headHeading_std_control
dat$pupilD_avg_event1_delta <- dat$pupilD_avg_event1 - dat$pupilD_avg_control
dat$pupilD_avg_event2_delta <- dat$pupilD_avg_event2 - dat$pupilD_avg_control
dat$pupilD_std_event1_delta <- dat$pupilD_std_event1 - dat$pupilD_std_control
dat$pupilD_std_event2_delta <- dat$pupilD_std_event2 - dat$pupilD_std_control

indices_highResilient = which(dat$crew == 1 | dat$crew == 2 | dat$crew == 7 | dat$crew == 12)
indices_modResilient = which(dat$crew == 3 |dat$crew == 4 | dat$crew == 5 | dat$crew == 9 | dat$crew == 11  | dat$crew == 13)
indices_lowResilient = which(dat$crew == 6 |dat$crew == 8 | dat$crew == 10)
rpsa_label = data.frame(matrix(ncol = 1, nrow = length(dat$pupilD_std_control)), colnames("rpsa_label"))
names(rpsa_label)[1] =   c("rpsa_label")
rpsa_label[indices_highResilient,] <- 'high'
rpsa_label[indices_modResilient,] <- 'moderate'
rpsa_label[indices_lowResilient,] <- 'low'
dat['rpsa_label'] = rpsa_label


lm_model <- lmer(gaze_variance_control ~ factor(crew) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_variance_control <- anova(lm_model)
anova_lm_model_gaze_variance_control
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance_control <- emmeans(rg_lm_model, "crew")
confint_lm_model_gaze_variance_control

lm_model <- lmer(gaze_variance_event1 ~ factor(crew) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_variance_event1 <- anova(lm_model)
anova_lm_model_gaze_variance_event1
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance_event1 <- emmeans(rg_lm_model, "crew")
confint_lm_model_gaze_variance_event1

lm_model <- lmer(gaze_variance_event2 ~ factor(crew) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_variance_event2 <- anova(lm_model)
anova_lm_model_gaze_variance_event2
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance_event2 <- emmeans(rg_lm_model, "crew")
confint_lm_model_gaze_variance_event2

lm_model <- lmer(gaze_variance_event1_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_variance_event1_delta <- anova(lm_model)
anova_lm_model_gaze_variance_event1_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance_event1_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_gaze_variance_event1_delta
ci <- hux(confint_lm_model_gaze_variance_event1_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("gaze_variance_event1",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(-1,1)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta gaze_variance_event1") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)


lm_model <- lmer(gaze_variance_event2_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_variance_event2_delta <- anova(lm_model)
anova_lm_model_gaze_variance_event2_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance_event2_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_gaze_variance_event2_delta
ci <- hux(confint_lm_model_gaze_variance_event2_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("gaze_variance_event2",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(-1,1)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta gaze_variance_event2") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9)) 
ggsave(file)




lm_model <- lmer(gaze_vel_avg_control ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_avg_control <- anova(lm_model)
anova_lm_model_gaze_vel_avg_control
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_vel_avg_control <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_gaze_vel_avg_control

lm_model <- lmer(gaze_vel_avg_event1 ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_avg_event1 <- anova(lm_model)
anova_lm_model_gaze_vel_avg_event1
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_vel_avg_event1 <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_gaze_vel_avg_event1

lm_model <- lmer(gaze_vel_avg_event2 ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_avg_event2 <- anova(lm_model)
anova_lm_model_gaze_vel_avg_event2
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_vel_avg_event2 <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_gaze_vel_avg_event2

lm_model <- lmer(gaze_vel_avg_event1_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_avg_event1_delta <- anova(lm_model)
anova_lm_model_gaze_vel_avg_event1_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_vel_avg_event1_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_gaze_vel_avg_event1_delta
ci <- hux(confint_lm_model_gaze_vel_avg_event1_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("gaze_vel_avg_event1",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,100)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta gaze_vel_avg_event1") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)

lm_model <- lmer(gaze_vel_avg_event2_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_avg_event2_delta <- anova(lm_model)
anova_lm_model_gaze_vel_avg_event2_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_vel_avg_event2_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_gaze_vel_avg_event2_delta
ci <- hux(confint_lm_model_gaze_vel_avg_event2_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("gaze_vel_avg_event2",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,100)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta gaze_vel_avg_event2") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)




lm_model <- lmer(gaze_vel_std_control ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std_control <- anova(lm_model)
anova_lm_model_gaze_vel_std_control
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_vel_std_control <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_gaze_vel_std_control

lm_model <- lmer(gaze_vel_std_event1 ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std_event1 <- anova(lm_model)
anova_lm_model_gaze_vel_std_event1
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_vel_std_event1 <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_gaze_vel_std_event1

lm_model <- lmer(gaze_vel_std_event2 ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std_event2 <- anova(lm_model)
anova_lm_model_gaze_vel_std_event2
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_vel_std_event2 <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_gaze_vel_std_event2

lm_model <- lmer(gaze_vel_std_event1_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std_event1_delta <- anova(lm_model)
anova_lm_model_gaze_vel_std_event1_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_vel_std_event1_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_gaze_vel_std_event1_delta
ci <- hux(confint_lm_model_gaze_vel_std_event1_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("gaze_vel_std_event1",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,100)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta gaze_vel_std_event1") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)


lm_model <- lmer(gaze_vel_std_event2_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std_event2_delta <- anova(lm_model)
anova_lm_model_gaze_vel_std_event2_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_vel_std_event2_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_gaze_vel_std_event2_delta
ci <- hux(confint_lm_model_gaze_vel_std_event2_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("gaze_vel_std_event2",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,100)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta gaze_vel_std_event2") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)




lm_model <- lmer(headHeading_avg_control ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_headHeading_avg_control <- anova(lm_model)
anova_lm_model_headHeading_avg_control
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_headHeading_avg_control <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_headHeading_avg_control

lm_model <- lmer(headHeading_avg_event1 ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_headHeading_avg_event1 <- anova(lm_model)
anova_lm_model_headHeading_avg_event1
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_headHeading_avg_event1 <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_headHeading_avg_event1

lm_model <- lmer(headHeading_avg_event2 ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_headHeading_avg_event2 <- anova(lm_model)
anova_lm_model_headHeading_avg_event2
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_headHeading_avg_event2 <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_headHeading_avg_event2

lm_model <- lmer(headHeading_avg_event1_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_headHeading_avg_event1_delta <- anova(lm_model)
anova_lm_model_headHeading_avg_event1_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_headHeading_avg_event1_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_headHeading_avg_event1_delta
ci <- hux(confint_lm_model_headHeading_avg_event1_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("headHeading_avg_event1",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,1)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta headHeading_avg_event1") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)


lm_model <- lmer(headHeading_avg_event2_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_headHeading_avg_event2_delta <- anova(lm_model)
anova_lm_model_headHeading_avg_event2_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_headHeading_avg_event2_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_headHeading_avg_event2_delta
ci <- hux(confint_lm_model_headHeading_avg_event2_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("headHeading_avg_event2",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,1)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta headHeading_avg_event2") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)




lm_model <- lmer(headHeading_std_control ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_headHeading_std_control <- anova(lm_model)
anova_lm_model_headHeading_std_control
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_headHeading_std_control <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_headHeading_std_control


lm_model <- lmer(headHeading_std_event1 ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_headHeading_std_event1 <- anova(lm_model)
anova_lm_model_headHeading_std_event1
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_headHeading_std_event1 <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_headHeading_std_event1

lm_model <- lmer(headHeading_std_event2 ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_headHeading_std_event2 <- anova(lm_model)
anova_lm_model_headHeading_std_event2
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_headHeading_std_event2 <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_headHeading_std_event2

lm_model <- lmer(headHeading_std_event1_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_headHeading_std_event1_delta <- anova(lm_model)
anova_lm_model_headHeading_std_event1_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_headHeading_std_event1_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_headHeading_std_event1_delta
ci <- hux(confint_lm_model_headHeading_std_event1_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("headHeading_std_event1",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(-100,0)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta headHeading_std_event1") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)

lm_model <- lmer(headHeading_std_event2_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_headHeading_std_event2_delta <- anova(lm_model)
anova_lm_model_headHeading_std_event2_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_headHeading_std_event2_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_headHeading_std_event2_delta
ci <- hux(confint_lm_model_headHeading_std_event2_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("headHeading_std_event2",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(-100,0)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta headHeading_std_event2") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)






lm_model <- lmer(pupilD_avg_control ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_pupilD_avg_control <- anova(lm_model)
anova_lm_model_pupilD_avg_control
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_pupilD_avg_control <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_pupilD_avg_control

lm_model <- lmer(pupilD_avg_event1 ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_pupilD_avg_event1 <- anova(lm_model)
anova_lm_model_pupilD_std_event1
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_pupilD_std_event1 <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_pupilD_std_event1

lm_model <- lmer(pupilD_avg_event2 ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_pupilD_std_event2 <- anova(lm_model)
anova_lm_model_pupilD_std_event2
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_pupilD_std_event2 <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_pupilD_std_event2

lm_model <- lmer(pupilD_avg_event1_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_pupilD_avg_event1_delta <- anova(lm_model)
anova_lm_model_pupilD_avg_event1_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_pupilD_avg_event1_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_pupilD_avg_event1_delta
ci <- hux(confint_lm_model_pupilD_avg_event1_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("pupilD_avg_event1",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(-.0005,.0005)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta pupilD_avg_event1") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)

lm_model <- lmer(pupilD_avg_event2_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_pupilD_avg_event2_delta <- anova(lm_model)
anova_lm_model_pupilD_avg_event2_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_pupilD_avg_event2_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_pupilD_avg_event2_delta
ci <- hux(confint_lm_model_pupilD_avg_event2_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("pupilD_avg_event2",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(-.0005,.0005)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta pupilD_avg_event2") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)







lm_model <- lmer(pupilD_std_control ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_pupilD_std_control <- anova(lm_model)
anova_lm_model_pupilD_std_control
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_pupilD_std_control <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_pupilD_std_control

lm_model <- lmer(pupilD_std_event1 ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_pupilD_std_event1 <- anova(lm_model)
anova_lm_model_pupilD_std_event1
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_pupilD_std_event1 <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_pupilD_std_event1

lm_model <- lmer(pupilD_std_event2 ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_pupilD_std_event2 <- anova(lm_model)
anova_lm_model_pupilD_std_event2
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_pupilD_std_event2 <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_pupilD_std_event2

lm_model <- lmer(pupilD_std_event1_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_pupilD_std_event1_delta <- anova(lm_model)
anova_lm_model_pupilD_std_event1_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_pupilD_std_event1_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_pupilD_std_event1_delta
ci <- hux(confint_lm_model_pupilD_std_event1_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("pupilD_std_event1",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(-.005,.005)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta pupilD_std_event1") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)

lm_model <- lmer(pupilD_std_event2_delta ~ factor(rpsa_label) + (1|subject), data=dat, REML = FALSE)
anova_lm_model_pupilD_std_event2_delta <- anova(lm_model)
anova_lm_model_pupilD_std_event2_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_pupilD_std_event2_delta <- emmeans(rg_lm_model, "rpsa_label")
confint_lm_model_pupilD_std_event2_delta
ci <- hux(confint_lm_model_pupilD_std_event2_delta)
rpsa_label = t(t(ci$rpsa_label))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))
confint_df = data.frame(rpsa_label, emmean, lower, upper)
figure_name = paste0("pupilD_std_event2",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(rpsa_label, levels = rpsa_factor_levels), y = emmean, fill = rpsa_label )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(-.005,.005)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="delta pupilD_std_event2") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)







rpsa = read.csv("rpsa.csv", na.strings = "N/A",fileEncoding="UTF-8-BOM")
names(rpsa)[1:5] =   c("SubjectID","Crew", "Seat","RunNumber","ScenarioID")

rpsa$ScenarioID = factor(rpsa$ScenarioID)

rpsa_long = reshape2::melt(rpsa, measure.vars = names(rpsa)[grepl("X", names(rpsa))], variable.name = "Item", value.name = "Rating")


itemNames = data.frame(Item = paste("X",1:16, sep=""), ItemDescription = c("AppliedPriorKnowledge", "VerballySharedKnowledge", 
                                                                           "HeightenedAwareness", "MonitoredAircraftStatus", "DiscussedExpectedActions", "DevelopedWhatIfs", "GatheredInformation", 
                                                                           "IdentifiedCountermeasures", "IntervenedUnwantedCondition", "ChangedAutomationOrSystem", "CrossCheckedOtherPilot", 
                                                                           "DecreaseOtherPilotWorkload", "AskedExternalSource", "AskedOtherPilot", "ManagedTimeEffectively", "DebriefedAfterProblem"))

rpsa_long = left_join(rpsa_long, itemNames)
rpsa_long$Item = factor(rpsa_long$Item, levels = paste("X",1:16, sep=""))


lm_model <- lmer(Rating ~ factor(Crew)*factor(Seat) + (1|SubjectID), data=rpsa_long, REML = TRUE)
anova_lm_model <- anova(lm_model)
rg_lm_model <- ref_grid(lm_model)
confint_lm_model <- emmeans(rg_lm_model, "Crew")
ci <- hux(confint_lm_model)
crew = t(t(ci$Crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))

confint_df = data.frame(crew, emmean, lower, upper)

indices_highResilient = which(confint_df$crew == 1 | confint_df$crew == 2 | confint_df$crew == 7 | confint_df$crew == 12)
indices_modResilient = which(confint_df$crew == 3 |confint_df$crew == 4 | confint_df$crew == 5 | confint_df$crew == 9 | confint_df$crew == 11  | confint_df$crew == 13)
indices_lowResilient = which(confint_df$crew == 6 |confint_df$crew == 8 | confint_df$crew == 10)
rpsa_label = data.frame(matrix(ncol = 1, nrow = length(confint_df$Crew)), colnames("rpsa_label"))
names(rpsa_label)[1] =   c("rpsa_label")
rpsa_label[indices_highResilient,] <- 'high'
rpsa_label[indices_modResilient,] <- 'moderate'
rpsa_label[indices_lowResilient,] <- 'low'
confint_df['rpsa_label'] = rpsa_label

figure_name = paste0("rpsa",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew), y = emmean )) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,5)) +
  theme_minimal() +
  ggtitle("Self-Reported Reslience") +
  labs(x="crew number", y="rpsa score") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9)) +
  scale_fill_brewer(palette="Paired")
ggsave(file)

figure_name = paste0("rpsa_labeled",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew), y = emmean , fill = rpsa_label)) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,5)) +
  theme_minimal() +
  ggtitle("RPSA") +
  labs(x="", y="rpsa") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9)) +
  scale_fill_brewer(palette="Paired")
ggsave(file)

