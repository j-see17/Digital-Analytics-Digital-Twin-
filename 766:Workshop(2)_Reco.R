# Clear memory
rm(list = ls(all.names = TRUE)) 
cat("\014")

### --------------------------------------------------------------------------------------------------------

###- {r}
is.installed <- function(mypkg) is.element(mypkg, installed.packages()[,1]) 


if (!is.installed('reshape2')) { install.packages("reshape2")}  
library(reshape2)


# Wrapper for LIBMF: A Matrix-factorization Library for Recommendation Systems

if (!is.installed('recosystem')) { install.packages("recosystem")}  
library(recosystem)

# set stopping function (optional)
stop_quietly <- function() {
  opt <- options(show.error.messages = FALSE)
  on.exit(options(opt))
  stop()
}

#----------------------------------------------------

get_os <- function(){
  sysinf <- Sys.info()
  if (!is.null(sysinf)){
    os <- sysinf['sysname']
    if (os == 'Darwin')
      os <- "osx"
  } else { ## mystery machine
    os <- .Platform$OS.type
    if (grepl("^darwin", R.version$os))
      os <- "osx"
    if (grepl("linux-gnu", R.version$os))
      os <- "linux"
  }
  tolower(os)
}


CopyToClipboard <- function(x) {
  
  if (get_os() == 'osx')
    write.table(x, file = pipe("pbcopy"), sep="\t", na = "", row.names=FALSE)
  else  
    write.table(x, "clipboard", sep="\t", na = "", row.names=FALSE)
  
}




# function sets working directory to the file location
setwd_thisdir <- function () {
  this.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
  setwd(this.dir)
} 


### --------------------------------------------------------------------------------------------------------

CalculateSimilarityMatrix <- function(Ratings, min_sim = 0, min_overlap = 0) {
  

# Ratings: Users x  Movies array
NumOfUsers = nrow(Ratings)
NumOfItems = ncol(Ratings)

UserNames = rownames(Ratings)

RatingsBinary = Ratings
RatingsBinary[ is.na(Ratings) ] = 0
RatingsBinary[ Ratings > 0 ] = 1
#RatingsBinary   = as.matrix(RatingsBinary)

Sparseness  = sum(RatingsBinary) /( ncol(RatingsBinary) * nrow(RatingsBinary) )
print(paste("Rating matrix Sparseness ",(1-Sparseness)*100, "%"))

RatingsOverlap  = RatingsBinary %*% t(RatingsBinary)
RatingsbyUser   = rowSums(RatingsBinary)

opt <- options(show.error.messages = FALSE)
distance_matrix = suppressWarnings(cor(t(Ratings), method = "pearson", use = "pairwise.complete.obs"))
options(opt)

# do not include negative matches
distance_matrix[distance_matrix < min_sim] = NA

# do not use statistics based on small samples
distance_matrix[RatingsOverlap < min_overlap] = NA


# do not include self-correlation
diag(distance_matrix) = NA


# set all NA to zeros
distance_matrix[is.na(distance_matrix)] = 0


colnames(distance_matrix) = UserNames
rownames(distance_matrix) = UserNames

return(distance_matrix)

}


### --------------------------------------------------------------------------------------------------------


LoadMovieDataSmall <- function(){
  
  
  FOLDER_PATH = ''
  data_file   = 'Movie-data-small.txt'
  
  rating_table = read.table(paste(FOLDER_PATH,data_file, sep = ''), sep="\t", header=F)
  colnames( rating_table ) <- c('user', 'item', 'rating')
  
  
  # validation if everything worked correctly
  
  # add numeric ids for users and movies
  rating_table_w_ids = transform(rating_table, 
                                 user_id = as.numeric(factor(rating_table$user)),
                                 item_id = as.numeric(factor(rating_table$item)))
  
  # sort table by user ids
  rating_table_w_ids = rating_table_w_ids[order(rating_table_w_ids$user_id), ] 
  
  # create lookup tables for users and movie titles
  user_ids = unique( rating_table_w_ids[, c('user_id', 'user')])
  item_ids = unique( rating_table_w_ids[, c('item_id', 'item')])
  
  return(rating_table_w_ids)
  
}



### --------------------------------------------------------------------------------------------------------


# return top-k similar users

Get_Top_K_Neighors <- function(User, distance_matrix, Neighorhood_size) {
  
  
  NumOfUsers = nrow(distance_matrix)
  similarity = distance_matrix[User,]
  
  # get indeces of the closest neighbors
  idx <- order(similarity, decreasing = T)[1:Neighorhood_size]
  
  return(data.frame(idx = idx, similarity = similarity[idx]))
  
}

### --------------------------------------------------------------------------------------------------------

Predict_with_MF <- function(train = train_data, elements_to_predict, NumOfLatentFactors) {
  
  
  # Create a "recosystem" object
  
  rec = Reco()
  set.seed(123)
  
  # Prepare inputs  for "recosystem" object by mapping corresponding fields from the training dataset
  
  rec_train = data_memory(user_index = train$user_id, item_index = train$item_id, rating=train$rating, index1 = TRUE)

  # Train the model (i.e. do matrix  factorization )  
  
  res = rec$train(rec_train, opts = list(dim=NumOfLatentFactors, verbose=FALSE, nmf = TRUE))
  
  # Extract latent factors P(users) and Q(items)
  factors = res$output(out_P = out_memory(), out_Q = out_memory())
  
  print(paste("User factors (PxQ)", nrow(factors$P), "x", nrow(factors$Q)) )
  
  # Example of predicting for one user (16)/item (7) pair 
  Dimensions_User = factors$P
  Dimensions_Item = factors$Q
  
  Predict_User_16_Movie_7 = Dimensions_User[16,] %*% (Dimensions_Item[7,])
  
  # Or full  prediction matrix
  predicted_matrix = factors$P %*% t(factors$Q)
  
  # Using the trained model, predict ratings for the user/item pairs provided in <elements_to_predict>
  
  predictions_input = data_memory(user_index = elements_to_predict$user_id,  
                                  item_index = elements_to_predict$item_id,  index1 = TRUE)
  
  
  pred  = rec$predict(predictions_input, out_pred = out_memory())
  
  # link the predicted values with user and item indeces
  pred_matrix_long = cbind(elements_to_predict, rating = pred)  
  
  
  # convert to "wide" format 
  pred_matrix_wide       = dcast(pred_matrix_long, user_id ~ item_id,  value.var = 'rating')
  
  print( paste("Resulting prediction matrix dim", nrow(pred_matrix_wide),"x",ncol(pred_matrix_wide)) )
  
  
  # This is an alternative way to get full prediction matrix
  # pred_matrix_wide_factor_product = factors$P %*% t(factors$Q)
  
  
  # return predictions in "long" format
  return(pred_matrix_long)
  
}


### --------------------------------------------------------------------------------------------------------


main <- function() {


# Setting K (i.e. neighborhood size for k-NN predictions)
K = 2
  
# Load Rating data <User,User_id,Item,Item_id,Rating> format
data <-  LoadMovieDataSmall()

# View the data in "wide" format (visualization with user names and movie titles)
# Make a note of Focal users IDs
Rating_table_wide = dcast(data, user_id + user ~ item_id + item,  value.var = 'rating')

# Set Focal User IDs here
Focal_user1 = 19
Focal_user2 = 22

# Specify User IDs of your Focal Users
Focal_users = c(Focal_user1,Focal_user2)


# ---------------------------------------------------------------------------------------------------
# -- PART I - COLLABORATIVE FILTERING WITH k-NN 
# ---------------------------------------------------------------------------------------------------



# ----------------------------   OPTION 1 -----------------------------------------------------------
#  Transfer data to EXCEL to fill out the worksheets
# ---------------------------------------------------------------------------------------------------

# Copy to Clipboard all but Focal Users and paste the date in the corresponding area in EXCEL
CopyToClipboard(Rating_table_wide[-Focal_users,-1])


# Put a breakpoint between these copy commands (or comment one out) so you can transfer the clipboard to EXCEL
print(paste("All but Focal Users data copied to clipboard"))

# Copy to Clipboard Focal Users data only and paste the date in the corresponding area in EXCEL
CopyToClipboard(Rating_table_wide[Focal_users,-1])
print(paste("Focal Users data copied to clipboard"))


#return()

# ---------------------------------------------------------------------------------------------------
# The rest of your analysis for the k-NN based predictions under this option will be done in EXCEL
# ----------------------------   END OF OPTION 1 -----------------------------------------------------



# ----------------------------   OPTION 2 -----------------------------------------------------------
#  Implement k-NN based predictions in R 
# ---------------------------------------------------------------------------------------------------

# Here you would code the steps of your analysis following the assignment instruction.
#
# Note, that while some parts of the code are already provided to you, you would still need
# to do some coding yourself. Hence, OPTION 1 might be generally easier to implement if you feel 
# less comfortable with R coding


# Step 1: Calculate sample averages (for all but focal users)

Rating_table_matrix = as.matrix(Rating_table_wide[,-c(1,2)])
rownames(Rating_table_matrix) = rownames(Rating_table_wide)

# Calculate average ratings for each movie
data_means = colMeans( Rating_table_matrix[-which(Rating_table_wide$user_id %in% Focal_users),], na.rm = TRUE)

# Create average-based predictions for the focal users
data_means_predictions = cbind( user_id = Focal_users, matrix( rep(data_means,each=length(Focal_users)), nrow=length(Focal_users), 
                                                               dimnames = list(Focal_users,colnames(Rating_table_matrix)) ))

# Extract true (observed) ratings for the focal users
data_true_values = Rating_table_wide[which(Rating_table_wide$user_id %in% Focal_users),-2]


# Step 2: Calculate MAPE for sample averages-based predictions 
# HINT: 
#       (a) you can calculate MAPE only for the known ratings of your focal users 
#       (b) create a single MAPE metrics that combines prediction errors across all focal users
#       (c) INPUTS: data_means_predictions and data_true_values


# --- YOUR MAPE CALCULATIONs CODE GOES HERE ---


# Step 3: Calculate similarities 
#
#         HINT: while in this assignment you are asked to make predictions for the focal users only
#               you can take advantage of the R function cor() and calculate correlations across ALL users
#               in your data set. Note that if you have selected OPTION 1 and doing this workshop in EXCEL it 
#               might be easier for you to perform similarity analysis just for the focal users


SimilarityMatrix = CalculateSimilarityMatrix(Rating_table_matrix, min_sim = .0, min_overlap = 0)


# Step 4: Print Top K neighbors for the focal users
#
#     HINT: you may want to refer to this code if you are doing this assignment in EXCEL just to make sure you are getting 
#     the same results

  Neighorhood_size = K
  
  print( paste("Neighborhood of size", Neighorhood_size) )
  
  for(User in Focal_users)  # loop over the focal users
  {
    print( paste("Top neighbors of user (", Rating_table_wide[User,'user_id'], ") -", Rating_table_wide[User,'user']) )
    
    Top_K_Neighors = Get_Top_K_Neighors(User, SimilarityMatrix, Neighorhood_size)
    
    Top_K_Neighors_with_names = cbind( Rating_table_wide[Top_K_Neighors$idx,'user'], Top_K_Neighors$similarity )
    
    print(Top_K_Neighors_with_names)
     
  }


# Step 5: Calculate kNN-based predictions for the focal users
#   
#   Here you need to code the k-NN prediction formula provided in the assignment document (and lecture notes)
#   The two inputs to your calculations are 
#
#      (a) Results of the function Get_Top_K_Neighors()  
#      (b) Rating_table_wide[]
  

# --- YOUR KNN CALCULATIONs CODE GOES HERE ---  
  

# Step 6: Calculate MAPE for kNN-based predictions 
# HINT: 
#       (a) you can calculate MAPE only for the known ratings of your focal users 
#       (b) create a single MAPE metrics that combines prediction errors across all focal users  
    

# --- YOUR MAPE CALCULATIONs CODE GOES HERE ---  

    
# Step 7: Prediction of unobserved ratings for the focal users 
# 
#       From the outcome of Step 5 make a note of the predictions the algorithm made for the 
#       movies not rated by the focal users  

  
# Step 8: Repeat the above steps for the different neighborhood size

# Step 9: Select the best preforming model out of 3 (i.e. average, K = 3 and K = 5) and report 
#         your predictions for the movies not rated by the focal individuals
  

# ----------------------------   END OF OPTION 2 -----------------------------------------------------  

#return()
  

      
# ---------------------------------------------------------------------------------------------------
# -- PART II - COLLABORATIVE FILTERING WITH MATRIX FACTORIZATION
# ---------------------------------------------------------------------------------------------------

    
  NumOfLatentFactors = 3  
    

# Step 1: Prepare data to call MF function

  train_data = data[, c('user_id', 'item_id', 'rating')]


# Step 2: Create the list of user_id/item_id pairs for which we want to make predictions
    
  # Option (a) - predict for every user/movie pair in the database 
  elements_to_predict = merge( x = unique(data[ , 'user_id']), y = unique(data[ , 'item_id']), by=NULL)

  # Option (b) - predict ratings of all movie for the focal users only
  elements_to_predict = merge( x = Focal_users, y = unique(data[ , 'item_id']), by=NULL)
  
  
  colnames(elements_to_predict) = c('user_id', 'item_id')  


# Step 3: Call the main function to perform matrix factorization and predictions
    
  # The function returns predictions in "long" format  
  PredictionsMF_long = Predict_with_MF(train_data, elements_to_predict, NumOfLatentFactors)

  
  # View the data in "wide" format (visualization only with names and titles)

  UserLookupTable = unique(data[ , c('user', 'user_id')])
  ItemLookupTable = unique(data[ , c('item','item_id')])

  PredictionsMF_wide_with_names = merge(x = PredictionsMF_long, y = UserLookupTable, by = "user_id", all.x = TRUE)
  PredictionsMF_wide_with_names = merge(x = PredictionsMF_wide_with_names, y = ItemLookupTable, by = "item_id", all.x = TRUE)
  
  PredictionsMF_wide = dcast(PredictionsMF_wide_with_names, user_id + user ~ item_id + item,  value.var = 'rating')
  CopyToClipboard(PredictionsMF_wide)
# Step 4: Calculate MAPE for MF-based predictions
  
  # Extract known ratings of the focal users
  True_Values = data[which(data$user_id %in% Focal_users),c('user_id', 'item_id', 'rating')]

  # Calculate MAPE (i.e. comparing true values with MF-predicted values)
  #
  # HINT: 
  #       (a) you can calculate MAPE only for the known ratings of your focal users 
  #       (b) create a single MAPE metrics that combines prediction errors across all focal users  
  

  # --- YOUR MAPE CALCULATIONs CODE GOES HERE ---  
  

# Step 5: Change the number of latent factors to 5 (NumOfLatentFactors = 5) and repeat the steps above


# Step 6: Compare the performance of two MF-based models (i.e. 3 and 5 latent factors)  with the best model 
#         identified in Part I of the workshop using MAPE. If the MF-based model(s) outperforms the Part I models, 
#         report your updated predictions for the movies not rated by the focal individuals    

    
return()

}



#----------------------------------------------------
#
# Main Call
#
#----------------------------------------------------

setwd_thisdir()

print(paste("Started at ",Sys.time()))

main()

print(paste("Finished at ",Sys.time()))

#----------------------------------------------------

