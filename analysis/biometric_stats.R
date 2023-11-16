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

system(paste0('gsutil -m cp "gs://soteria_study_data/Analysis/total_eventEEG_metric_dataframe.csv" .'))

setwd("C:/Users/tfettrow/Documents/GitHub/soteria_code_appdat/soteria_code/analysis")
eeg_dat <- read.csv(file = 'total_eventEEG_metric_dataframe.csv',fileEncoding="UTF-8-BOM")



lm_model <- lmer(engagement_index_spec ~ event_label + factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova <- anova(lm_model)
rg_lm_model <- ref_grid(lm_model)
confint <- emmeans(rg_lm_model, ~ "event_label + crew")
confint
ci <- hux(confint)
crew = t(t(ci$crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$asymp.LCL))
upper = t(t(ci$asymp.UCL))
confint_df = data.frame(crew, emmean, lower, upper)
figure_name = paste0("engagement_index_spec",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew, levels = crew), y = emmean)) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  theme_minimal() +
  ggtitle("Engagement Index") +
  labs(x="", y="Engagement Index") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)

lm_model <- lmer(taskLoad_index_spec ~ event_label + factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova <- anova(lm_model)
rg_lm_model <- ref_grid(lm_model)
confint <- emmeans(rg_lm_model, ~" event_label + crew")
confint
ci <- hux(confint)
crew = t(t(ci$crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$asymp.LCL))
upper = t(t(ci$asymp.UCL))
confint_df = data.frame(crew, emmean, lower, upper)
figure_name = paste0("taskLoad_index_spec",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew, levels = crew), y = emmean)) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  theme_minimal() +
  ggtitle("Task Load Index") +
  labs(x="", y="Task Load Index") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)


system(paste0('gsutil -m cp "gs://soteria_study_data/Analysis/total_eventEKG_metric_dataframe.csv" .'))

setwd("C:/Users/tfettrow/Documents/GitHub/soteria_code_appdat/soteria_code/analysis")
dat <- read.csv(file = 'total_eventEKG_metric_dataframe.csv',fileEncoding="UTF-8-BOM")

lm_model <- lmer(beats_per_min ~ event_label + factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova <- anova(lm_model)
rg_lm_model <- ref_grid(lm_model)
confint <- emmeans(rg_lm_model, ~ "event_label + crew")
confint
ci <- hux(confint)
crew = t(t(ci$crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$asymp.LCL))
upper = t(t(ci$asymp.UCL))
confint_df = data.frame(crew, emmean, lower, upper)
figure_name = paste0("beats_per_min",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew, levels = crew), y = emmean)) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  theme_minimal() +
  ggtitle("Beats Per Min") +
  labs(x="", y="Beats Per Min") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)

system(paste0('gsutil -m cp "gs://soteria_study_data/Analysis/total_eventEKG_metric_dataframe.csv" .'))

setwd("C:/Users/tfettrow/Documents/GitHub/soteria_code_appdat/soteria_code/analysis")
ekg_dat <- read.csv(file = 'total_eventEKG_metric_dataframe.csv',fileEncoding="UTF-8-BOM")

lm_model <- lmer(hr_var ~ event_label + factor(crew) * seat + (1|seat), data=dat, REML = FALSE)
anova <- anova(lm_model)
rg_lm_model <- ref_grid(lm_model)
confint <- emmeans(rg_lm_model, ~ "event_label + crew")
confint
ci <- hux(confint)
crew = t(t(ci$crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$asymp.LCL))
upper = t(t(ci$asymp.UCL))
confint_df = data.frame(crew, emmean, lower, upper)
figure_name = paste0("hr_var",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew, levels = crew), y = emmean)) +
  geom_bar(stat="identity", position= "dodge") + #nb you can just use 'dodge' in barplots
  scale_fill_brewer(palette="Paired") +
  theme_minimal() +
  ggtitle("Heart Rate Variability") +
  labs(x="", y="Heart Rate Variability") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)



system(paste0('gsutil -m cp "gs://soteria_study_data/Analysis/total_eventSmarteye_metric_dataframe.csv" .'))

setwd("C:/Users/tfettrow/Documents/GitHub/soteria_code_appdat/soteria_code/analysis")
smarteye_dat <- read.csv(file = 'total_eventSmarteye_metric_dataframe.csv',fileEncoding="UTF-8-BOM")

lm_model <- lmer(gaze_variance ~ event_label + factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova_lm_model_gaze_variance_event1_delta <- anova(lm_model)
anova_lm_model_gaze_variance_event1_delta
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance <- emmeans(rg_lm_model, ~ "event_label + crew")
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
  coord_cartesian(ylim = c(0,1.00)) +
  theme_minimal() +
  ggtitle("Gaze Variance") +
  labs(x="", y="Gaze Variance") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  geom_errorbar(aes(ymin=lower, ymax=upper), width=.2, position = position_dodge(width = 0.9))
ggsave(file)

lm_model <- lmer(gaze_vel_avg ~ event_label + factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_avg <- anova(lm_model)
anova_lm_model_gaze_vel_avg
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance <- emmeans(rg_lm_model, ~" event_label +crew")
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


lm_model <- lmer(headHeading_avg ~ event_label + factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std <- anova(lm_model)
anova_lm_model_gaze_vel_std
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance <- emmeans(rg_lm_model, ~"event_label +crew")
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


lm_model <- lmer(headHeading_std ~ event_label + factor(crew) *seat + (1|seat), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std <- anova(lm_model)
anova_lm_model_gaze_vel_std
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance <- emmeans(rg_lm_model, ~"event_label +crew")
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


lm_model <- lmer(pupilD_avg ~ event_label + factor(crew) * seat + (1|seat), data=dat, REML = FALSE)
anova_lm_model_gaze_vel_std <- anova(lm_model)
anova_lm_model_gaze_vel_std
rg_lm_model <- ref_grid(lm_model)
confint_lm_model_gaze_variance <- emmeans(rg_lm_model, ~"event_label +crew")
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











system(paste0('gsutil -m cp "gs://soteria_study_data/Analysis/rpsa.csv" .'))

setwd("C:/Users/tfettrow/Documents/GitHub/soteria_code_appdat/soteria_code/analysis")
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

lm_model <- lmer(Rating ~ factor(Crew)*factor(Seat) + (1|SubjectID), data=rpsa_long, REML = FALSE)
anova_lm_model <- anova(lm_model)
rg_lm_model <- ref_grid(lm_model)
confint_lm_model <- emmeans(rg_lm_model, "Crew:Seat")
ci <- hux(confint_lm_model)
crew = t(t(ci$Crew))
emmean = t(t(ci$emmean))
lower = t(t(ci$lower.CL))
upper = t(t(ci$upper.CL))

confint_df = data.frame(crew, emmean, lower, upper)

ggplot(rpsa_long, aes(x=factor(Crew), y = Rating, fill = factor(Seat) )) + 
  geom_bar() +
  scale_fill_brewer(palette="Paired") +
  
  stat_summary(fun=mean, geom="point", aes(group=Seat), position=position_dodge(.9), 
               color="black", size=4)

ggplot(rpsa_long, aes(x=factor(Crew), y = Rating, fill = factor(Seat) )) +
  geom_boxplot(outlier.size = 0) + #nb you can just use 'dodge' in barplots
  geom_point(pch = 21, position = position_jitterdodge())
  scale_fill_brewer(palette="Paired") +
  coord_cartesian(ylim = c(0,5)) +
  theme_minimal() +
  ggtitle("Self-Reported Reslience") +
  labs(x="crew number", y="rpsa score") + 
  theme(legend.position = "none")+
  theme(panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.grid=element_blank()) +
  scale_fill_brewer(palette="Paired")

figure_name = paste0("rpsa",".tiff")
file = file.path("Figures", figure_name)
ggplot(confint_df, aes(x=factor(crew), y = emmean, fill = factor(Seat) )) +
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
