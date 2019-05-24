dt$hole_correct     <-  ifelse(dt$event_type=='reward', dt$hole_no, 0)
dt$hole_incorrect   <-  ifelse(dt$event_type=='failure', dt$hole_no, 0)
dt$is_correct       <-  ifelse(dt$event_type=='reward', 1, 0)
dt$is_incorrect     <-  ifelse(dt$event_type=='failure', 1, 0)
dt$is_omission      <-  ifelse(dt$event_type=='time over', 1, 0)

dt$cumsum_correct   <-  cumsum(dt$is_correct)
dt$cumsum_incorrect <-  cumsum(dt$is_incorrect)
dt$cumsum_omission  <-  cumsum(dt$is_omission)

## cumsum

dt$cumsum_correct_taskreset <- dt$cumsum_correct
dt$cumsum_incorrect_taskreset <- dt$cumsum_incorrect
dt$cumsum_omission_taskreset <- dt$cumsum_omission

for (i in 1:length(task_start_idx)){
  idx_st <- task_start_idx[i]
  if (i==length(task_start_idx)){
    idx_end <- nrow(dt)
  }else{
    idx_end <- task_start_idx[i+1] - 1
  }
  dt$cumsum_correct_taskreset[idx_st:idx_end] <- dt$cumsum_correct_taskreset[idx_st:idx_end] - dt$cumsum_correct_taskreset[idx_st]
  dt$cumsum_incorrect_taskreset[idx_st:idx_end] <- dt$cumsum_incorrect_taskreset[idx_st:idx_end] - dt$cumsum_incorrect_taskreset[idx_st]
  dt$cumsum_omission_taskreset[idx_st:idx_end] <- dt$cumsum_omission_taskreset[idx_st:idx_end] - dt$cumsum_omission_taskreset[idx_st]
  
}

## is_holex
dt$is_hole1 <- ifelse(grepl("1",dt$hole_no), 1,NA)
dt$is_hole3 <- ifelse(grepl("3",dt$hole_no), 1,NA)
dt$is_hole5 <- ifelse(grepl("5",dt$hole_no), 1,NA)
dt$is_hole7 <- ifelse(grepl("7",dt$hole_no), 1,NA)
dt$is_hole9 <- ifelse(grepl("9",dt$hole_no), 1,NA)

# ## entropy counting
# ent <- function(x){
#   return(entropy(discretize(x, numBins=5, r=c(1,9))))
# }
# he <- rollapply(dt$hole_no,width=150,ent)
# he <- append(array(0,c(1,length(dt$hole_no)-length(he))), he)
# dt$hole_choice_entropy <- he
# dt <- transform(dt,index=as.numeric(rownames(dt)))

## burst
# dt$burst_group[1] <- 1
# for(i in 2:nrow(dt)){
#   if(as.numeric(dt$timestamps[i]-dt$timestamps[i-1],units="secs") <= 60){
#     dt$burst_group[i] <- dt$burst_group[i-1]
#     next
#   }
#   dt$burst_group[i] <- dt$burst_group[i-1] + 1
# }
