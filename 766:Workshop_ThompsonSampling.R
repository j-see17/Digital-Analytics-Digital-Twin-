### --------------------------------------------------------------------------------------------------------
###
### Multi-armed bandit experiments using Thompson sampling 
###
### based on the article "Multi-armed bandit experiments in the online service economy," by Scott (2014)
###
### --------------------------------------------------------------------------------------------------------


# Clear memory

rm(list = ls(all.names = TRUE)) 
cat("\014")

# set stopping function (optional)
stop_quietly <- function() {
  opt <- options(show.error.messages = FALSE)
  on.exit(options(opt))
  stop()
}



### --------------------------------------------------------------------------------------------------------


# True ad performances
#
# In the real life we don't know these numbers, so instead the performance statistics is  
# collected by observing user response (click/no click) 
#
##
   ad_A_CTR = 0.04
   ad_B_CTR = 0.01
   ad_C_CTR = 0.01
   ad_D_CTR = 0.03

   q_actual_performance = c(ad_A_CTR, ad_B_CTR, ad_C_CTR, ad_D_CTR)
   
##   


# How many times we want to replicate the experiment
REPLICATIONS = 1

# Maximum duration of the experiment (e.g., days)
T = 1000

# How many impressions need to be allocated per period.
# At extreme the value could be 1, i.e. alpha and beta are updated 
# with every ad impression served
#
# With ImpressionsPerPeriod > 1, we serve multiple impressions first across the variants 
# and then update alpha and beta.
# For example, if we assume that the unit of time is 1 day, than ImpressionsPerPeriod 
# would represent how many people will see our ads per day

ImpressionsPerPeriod = 100

# Number of draws to be used  to approximate integration over the posterior  
# distribution of q

N_intergrate = 1000


# Minimum number of observations to collect before stopping based on regret
BURNIN_REGRET = 0

### --------------------------------------------------------------------------------------------------------


# Number of variants is inferred from the input vector <q_actual_performance>
NumOfVariants = length(q_actual_performance)

# Setting up the priors (at <1,1> we get flat uninformative prior )
# Note that alpha_prior and beta_prior are vectors with each element corresponding to one variant (ad)
alpha_prior  = rep(1, NumOfVariants)
beta_prior   = rep(1, NumOfVariants)


# Placeholder to store replication results (i.e. if we want to repeat our experiment multiple times to see consistency) 
# Values to store: run index, number of times intervals until convergence, regret,  W_at[1], Ad_1_is_winner
replication_runs = matrix(0,ncol = 5, nrow = 0)
colnames(replication_runs) = c('RunIndex','IntervalsToStop', 'RegretatStop', 'W_Ad_1', 'Ad_1_is_winner')

# Track actual impression allocation per variant
replication_runs_volumes = matrix(0,ncol = NumOfVariants, nrow = 0)


# show starting time
print(paste("Started at ", Sys.time()))


for (iRUN in 1:REPLICATIONS){

  
print(paste("Replication #", iRUN))
  
# Set run-level counters to 0 
num_failures  = rep(0,NumOfVariants) 
num_successes = rep(0,NumOfVariants)

# The following are space holders for run-level tracking
variant_q_hat = matrix(0,ncol = NumOfVariants, nrow = 0) # tracking q_hat across T

variants_vol= matrix(0,ncol = NumOfVariants, nrow = 0)
variants_successes= matrix(0,ncol = NumOfVariants, nrow = 0)
variants_w_at =  matrix(0,ncol = NumOfVariants, nrow = 0)
regret = matrix(0, ncol = 1, nrow = 0)

###---

for (i in 1:T) {
  
  
  # Beta distribution with parameters shape1 = alpha and shape2 = beta
  #
  # pdf: f(x) = x^(alpha-1) * (1-x)^(beta-1) / BETA(alpha,beta)
  #
  # E[X] = alpha / (alpha + beta)
  # Mode[X] = (alpha - 1/3) / (alpha + beta - 2/3)
  #
  
  
  #----------------------------------------------------
  # KEY STEP: Bayesian updating
  #----------------------------------------------------
  alpha = alpha_prior + num_successes
  beta  = beta_prior  + num_failures
  #----------------------------------------------------
  
  
  # Tracking empirical expected value of q
  variant_q_hat = rbind(variant_q_hat, t(alpha / (alpha + beta)) )
  
  # Tracking empirical median of q
  #variant_q_hat = rbind(variant_q_hat, t((alpha-1/3) / (alpha + beta - 2/3)) )
  
  
  # Draw from the posterior distributions (i.e., q <==> CTR) of each variant <N_integrate> times 
  # and calculate probabilities W_at which represent the share of "wins" of each variant
  #
  # OUTPUT: q_Draws - N_intergrate-by-NumOfVariants matrix with each column representing draws from
  # a corresponding beta distribution
  
  q_Draws = matrix(  rbeta(N_intergrate * NumOfVariants , alpha, beta),  N_intergrate, byrow = TRUE)
  
  # Identify winning variant for each draw (i.e. best ad based on the highest q for each row in the matrix)
  #
  # Note: factor() is used to have a placeholder for variants which get 0 posterior draws
  #
  
  WinningVariantsPerDraw = factor(max.col(q_Draws), levels=1:NumOfVariants)
  
  
  # Get total count and share of wins (W_at) for each variant using table() function
  TotalWinsbyVariant = table(WinningVariantsPerDraw)
  
  #----------------------------------------------------
  # KEY STEP: Calculating "share of wins" 
  # (Probability of ad a being the best) for each Ad 
  #----------------------------------------------------
  W_at = TotalWinsbyVariant / N_intergrate
  #----------------------------------------------------
  
  
  #----------------------------------------------------
  # KEY STEP: Impression allocation for next time period
  #
  # Allocate next draw(s) according to W_at
  # by drawing from the Multinomial Distribution 
  #----------------------------------------------------
  ImpressionsPerVariant  = rmultinom(1, ImpressionsPerPeriod, W_at) 
  #----------------------------------------------------


  
  #----------------------------------------------------
  # KEY STEP: Observe ad performance
  #
  # Simulating real world ad campaign run for one time period
  #
  # Draw outcomes using *ACTUAL* variant performance (hence using "q_actual_performance")
  # (in real life <draw_successes> would come from observing users' response to the ads)
  #
  # Note, the first parameter of rbinom() in a multivariate case should be a vector of *any* values but matching dimension
  #----------------------------------------------------
  draw_successes = rbinom( seq(1,NumOfVariants), size = ImpressionsPerVariant, prob = q_actual_performance)
  #----------------------------------------------------
  
  
  #----------------------------------------------------
  # KEY STEP: Preparation for updating for next iteration
  #
  # update successes and failures counters
  #----------------------------------------------------
  num_successes = num_successes + draw_successes
  num_failures  = num_failures  + (ImpressionsPerVariant - draw_successes)
  #----------------------------------------------------
  
  
  #----------------------------------------------------
  # KEY STEP: Calculate Regret (see discussion on pages 5-6 Scott (2014))
  #----------------------------------------------------
  
  # Identify running Winning variant (ad): the best performing ad so far
  WinningVariant = which.max(W_at)
  # What is q (i.e., CTR) of *our best performing ad* when simulating 
  # draws from its posterior distributions
  q_of_RunningWinner   = q_Draws[,WinningVariant]

  # What is the best q we observe across *ALL ads* when simulating 
  # draws from there corresponding posterior distributions
  Highest_q_PerDraw    = apply(q_Draws, 1, FUN=max)
  
  # Calculate regret as a deviation of the performance of the current winner 
  # from the highest performing q in each draw
  # (see page 6, equation (5) in Scott (2014))
  regret_per_draw  = (Highest_q_PerDraw - q_of_RunningWinner) / q_of_RunningWinner
  
  # getting upper quantile (95%) for regret 
  # (see page 6 Scott (2014))
  top_5_regret = quantile(regret_per_draw, c(.95) )
  
  #----------------------------------------------------
  
  
  # Tracking various performance metrics for plotting
  variants_vol        = rbind(variants_vol, t(matrix(ImpressionsPerVariant)) )
  variants_successes  = rbind(variants_successes, t(matrix(draw_successes)) )
  variants_w_at       = rbind(variants_w_at, t(matrix(W_at)) )
  regret              = rbind(regret, top_5_regret )
  
  #----------------------------------------------------
  # KEY STEP: Applying Stopping rule
  #----------------------------------------------------
  if (top_5_regret < 0.01 & sum(num_successes + num_failures) > BURNIN_REGRET) {
      #break
   }
  
}

replication_runs = rbind(replication_runs, c(iRUN, i, top_5_regret, W_at[1], W_at[1] > W_at[2]) )
replication_runs_volumes = rbind(replication_runs_volumes, colSums(variants_vol)  )

} # end of runs

print(paste("Finished at ", Sys.time()))



### ------------------------------------------------------------------------------------------------
#
#   --- VISUALIZATIONS --- (Select and run individual section)
#
### ------------------------------------------------------------------------------------------------


### ------------------------------------------------------------------------------------------------
#
#  VISUALIZATION #1: PLOTS FOR A SINGLE RUN 
#
### ------------------------------------------------------------------------------------------------

par(mfrow=c(2,2))

y = variant_q_hat

line_colors = c("red", "green", "blue", "black")
plot(y[,1],type="l",col=line_colors[1],lty=1,
     main = "Ad Performance",
     ylab="Success Probability",lwd=2,
     xlab="Interations",xaxt="n",yaxt="n", ylim=c(min(y),max(q_actual_performance)*1.4))

for (j in 2:NumOfVariants){
lines(y[,j],type="l",col=line_colors[j],lty=1,lwd=2)
}
grid()
legend("topright",legend=q_actual_performance,lty=c(1),
       col=line_colors[1:NumOfVariants],bg="white",lwd=2)
axis(1, at = seq(0, dim(y)[1], by = 100), las=2)
axis(2, at = seq(0, max(q_actual_performance)*1.4, by = .01), las=2)



y = variants_vol
line_colors = c("red", "green", "blue", "black")
plot(y[,1],type="l",col=line_colors[1],lty=1,
     main = "Ad Volume per Iteration",
     ylab="Impressions per Variant",lwd=2,
     xlab="Interations",xaxt="n",yaxt="n", ylim=c(min(y),max(y)*1.1))
for (j in 2:NumOfVariants){
  lines(y[,j],type="l",col=line_colors[j],lty=1,lwd=2)
}


grid()
legend("topright",legend=q_actual_performance,lty=c(1),
       col=line_colors[1:NumOfVariants],bg="white",lwd=2)
axis(1, at = seq(0, dim(y)[1], by = 200), las=2)
axis(2, at = seq(0, max(y)*1.1, by = 5), las=2)


barplot(colSums(replication_runs_volumes),
        main = paste("Impressions by Ad ( total", sprintf("%d", sum(replication_runs_volumes)), ")"),
        xlab = "Ads",
        ylab = "Impressions",
        names.arg = q_actual_performance,
        col=line_colors)
legend("topright",legend=q_actual_performance,lty=c(1,2),
       col=line_colors,bg="white",lwd=2)



TotalVolumesperVariant = colSums(replication_runs_volumes)
TotalConversionsExperiment   = sum(TotalVolumesperVariant * q_actual_performance)
TotalConversionsBestPossible = sum(replication_runs_volumes) * max(q_actual_performance)

barplot(c(TotalConversionsExperiment, TotalConversionsBestPossible),
        main = paste("Conversions lost", 
                     floor(TotalConversionsBestPossible - TotalConversionsExperiment ),
                     "out of", floor(TotalConversionsBestPossible)),
        ylab = "Conversions",
        names.arg = c('Experiment', 'Best Possible'),
        col=line_colors[1:2])



stop_quietly() # Prevents the subsequent code from running


### ------------------------------------------------------------------------------------------------
#
#  VISUALIZATION #2: CONVERSIONS LOST ACROSS REPLICATIONS
#
### ------------------------------------------------------------------------------------------------


par(mfrow=c(1,2))

line_colors = c("red", "green", "blue", "black")

TotalVolumesperVariant = colSums(replication_runs_volumes)
TotalConversionsExperiment   = sum(TotalVolumesperVariant * q_actual_performance)
TotalConversionsBestPossible = sum(replication_runs_volumes) * max(q_actual_performance)

Avg_TotalConversionsExperiment   = TotalConversionsExperiment / dim(replication_runs_volumes)[1]
Avg_TotalConversionsBestPossible = TotalConversionsBestPossible / dim(replication_runs_volumes)[1]


barplot( c(Avg_TotalConversionsExperiment, Avg_TotalConversionsBestPossible),
        main = paste("Average Conversions lost", 
                     floor(Avg_TotalConversionsBestPossible - Avg_TotalConversionsExperiment),
                     "out of", floor(Avg_TotalConversionsBestPossible)),
        ylab = "Conversions",
        names.arg = c('Experiment', 'Best Possible'),
        col=line_colors[1:2])


ConversionsPerVariantRepl = replication_runs_volumes * matrix( rep(q_actual_performance, 
                                dim(replication_runs_volumes)[1]), dim(replication_runs_volumes)[1], 
                                byrow = TRUE)
ConversionsPerRepl = rowSums(ConversionsPerVariantRepl)
ConversionsPerReplBestPossible = sum(replication_runs_volumes[1,]) * max(q_actual_performance)
ConversionsLostPerRepl = floor(ConversionsPerReplBestPossible - ConversionsPerRepl)


hist(ConversionsLostPerRepl, 
     main="Conversions Lost Per Replication", 
     xlab="Conversions Lost", 
     ylab="Frequency", 
     border="blue", 
     col="green",
     las=1, 
     breaks=10)



### ------------------------------------------------------------------------------------------------
#
#  VISUALIZATION #3:  STOPPING BASED ON REGRET
#
### ------------------------------------------------------------------------------------------------


par(mfrow=c(1,2))

line_colors = c("red", "green", "blue", "black")

hist(replication_runs[,'IntervalsToStop'], 
     main=paste("Regret-based Stopping (",mean(replication_runs[,'IntervalsToStop']),")"),
     xlab="Number of observations", 
     ylab="Frequency", 
     border="blue", 
     col="green",
     las=1, 
     breaks=10)


Ad_1_is_winner = sum(replication_runs[,'Ad_1_is_winner'])

barplot( c(Ad_1_is_winner, dim(replication_runs_volumes)[1] - Ad_1_is_winner),
         main = paste("Number of Wins per Ad"),
         ylab = "Wins",
         names.arg = c(paste('Ad 1 (',Ad_1_is_winner,')'), 
                       paste('Ad 2 (',dim(replication_runs_volumes)[1] - Ad_1_is_winner,')')),
         col=line_colors[1:2])


### ------------------------------------------------------------------------------------------------


