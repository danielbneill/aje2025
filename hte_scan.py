
import numpy as np
import pandas as pd
from sklearn import *
import scipy.stats as scs
import time
from sklearn import linear_model
logistic = linear_model.LogisticRegression(solver='liblinear',penalty='l1',C=1.0)

def compute_slopeterm_given_q(thesum,theprobs,q):
    return thesum-theprobs.apply(lambda x: q*x/(1-x+q*x)).sum()

def binary_search_on_slopeterm(thesum,theprobs):
    q_temp_min = 0.000001
    q_temp_max = 1000000.0
    while np.abs(q_temp_max-q_temp_min) > 0.000001:
        q_temp_mid = (q_temp_min+q_temp_max)/2
        if np.sign(compute_slopeterm_given_q(thesum,theprobs,q_temp_mid)) > 0:
            q_temp_min = q_temp_min+(q_temp_max-q_temp_min)/2
        else:
            q_temp_max = q_temp_max-(q_temp_max-q_temp_min)/2
    return (q_temp_min+q_temp_max)/2
    
def compute_score_given_q(thesum,theprobs,penalty,q):
    if (q <= 0):
        print("Warning: calling compute_score_given_q with thesum=",thesum,"theprobs of length",len(theprobs),"penalty=",penalty,"q=",q)
    return thesum*np.log(q)-np.log(1-theprobs+q*theprobs).sum() - penalty

def binary_search_on_score_for_q_min(thesum,theprobs,penalty,q_mle):
    q_temp_min = 0.000001
    q_temp_max = q_mle
    while np.abs(q_temp_max-q_temp_min) > 0.000001:
        q_temp_mid = (q_temp_min+q_temp_max)/2
        if np.sign(compute_score_given_q(thesum,theprobs,penalty,q_temp_mid)) > 0:
            q_temp_max = q_temp_max-(q_temp_max-q_temp_min)/2
        else:
            q_temp_min = q_temp_min+(q_temp_max-q_temp_min)/2
    return (q_temp_min+q_temp_max)/2

def binary_search_on_score_for_q_max(thesum,theprobs,penalty,q_mle):
    q_temp_min = q_mle
    q_temp_max = 1000000.0
    while np.abs(q_temp_max-q_temp_min) > 0.000001:
        q_temp_mid = (q_temp_min+q_temp_max)/2
        if np.sign(compute_score_given_q(thesum,theprobs,penalty,q_temp_mid)) > 0:
            q_temp_min = q_temp_min+(q_temp_max-q_temp_min)/2
        else:
             q_temp_max = q_temp_max-(q_temp_max-q_temp_min)/2
    return (q_temp_min+q_temp_max)/2


def compute_q(thesum,theprobs,penalty):
    q_mle = binary_search_on_slopeterm(thesum,theprobs)
    if compute_score_given_q(thesum,theprobs,penalty,q_mle) > 0:
        positive = 1
        q_min = binary_search_on_score_for_q_min(thesum,theprobs,penalty,q_mle)
        q_max = binary_search_on_score_for_q_max(thesum,theprobs,penalty,q_mle)
    else:
        positive = 0
        q_min = 0
        q_max = 0
    return positive, q_mle, q_min, q_max

def get_aggregates(coordinates,probs,outcomes,values_to_choose,column_name,penalty,direction='positive'):
    #print("Calling get_aggregates with column_name=",column_name)
    if values_to_choose:
        to_choose = coordinates[values_to_choose.keys()].isin(values_to_choose).all(axis=1)
        temp_df=pd.concat([coordinates.loc[to_choose], outcomes[to_choose],pd.Series(data=probs[to_choose],index=outcomes[to_choose].index,name='prob')],axis=1)
    else:
        temp_df= pd.concat([coordinates, outcomes, pd.Series(data=probs,index=outcomes.index,name='prob')],axis=1)
    aggregates = {}
    thresholds = set()
    for name, group in temp_df.groupby(column_name):
        thesum = group.iloc[:,-2].sum() 
        theprobs = group.iloc[:,-1]
        positive, q_mle, q_min, q_max = compute_q(thesum,theprobs,penalty)
        #print("name=",name,"q_mle=",q_mle,"q_min=",q_min,"q_max=",q_max)
        if positive:
            if direction == 'positive':
                if q_max < 1:
                    positive = 0
                elif q_min < 1:
                    q_min = 1
            else: 
                if q_min > 1:
                    positive = 0
                elif q_max > 1:
                    q_max = 1
            if positive:
                aggregates[name]={'positive':positive, 'q_mle':q_mle,'q_min':q_min,'q_max':q_max,'thesum':thesum,'theprobs':theprobs}
                thresholds.update([q_min,q_max])
    allsum = temp_df.iloc[:,-2].sum()
    allprobs = temp_df.iloc[:,-1]
    return [aggregates,sorted(thresholds),allsum,allprobs]

def choose_aggregates(aggregates,thresholds,penalty,allsum,allprobs,direction='positive'):
    best_score = 0
    best_q = 0
    best_names = []
    for i in range(len(thresholds)-1):
        thethreshold = (thresholds[i]+thresholds[i+1])/2
        names = []
        thesum = 0.0
        theprobs = []
        for key, value in aggregates.items():
            if (value['positive']) & (value['q_min'] < thethreshold) & (value['q_max'] > thethreshold):
                names.append(key)
                thesum += value['thesum']
                theprobs = theprobs + value['theprobs'].tolist()
        theprobs_series = pd.Series(theprobs, dtype='float64')
        current_q_mle = binary_search_on_slopeterm(thesum,theprobs_series)
        if ((direction == 'positive') & (current_q_mle < 1)) | ((direction != 'positive') & (current_q_mle > 1)):
            current_q_mle = 1
        current_score = compute_score_given_q(thesum,theprobs_series,penalty*len(names),current_q_mle)
        #print "In choose_aggregates, current_score = ",current_score+penalty*len(names),"-",penalty*len(names),"=",current_score
        if current_score > best_score:
            best_score = current_score
            best_q = current_q_mle
            best_names = names
        #print 'current',names,current_score,current_q_mle,'with penalty of',penalty*len(names)
    # also have to consider case of including all attributes values including those that never make positive contributions to the score
    allprobs_series = pd.Series(allprobs)
    current_q_mle = binary_search_on_slopeterm(allsum,allprobs_series)
    if ((direction == 'positive') & (current_q_mle < 1)) | ((direction != 'positive') & (current_q_mle > 1)):
        current_q_mle = 1
    current_score = compute_score_given_q(allsum,allprobs_series,0,current_q_mle)
    #print "In choose_aggregates, current_score = ",current_score,"-[no penalty]=",current_score
    if current_score > best_score:
        best_score = current_score
        best_q = current_q_mle
        best_names = []
    #print 'current',names,current_score,current_q_mle 
    #print 'best',best_names,best_score,best_q
    return [best_names,best_score]

def mk_subset_all_values(coordinates):
    subset_all_values = {}
    for theatt in coordinates:
        subset_all_values[theatt]=coordinates[theatt].unique().tolist()
    return subset_all_values

def mk_subset_random_values(coordinates,prob,minelements=0):
    subset_random_values = {}
    shuffled_column_names = np.random.permutation(coordinates.columns.values)
    for theatt in shuffled_column_names:
        temp = coordinates[theatt].unique()
        mask = np.random.rand(len(temp)) < prob
        if mask.sum() < len(temp):
            subset_random_values[theatt] = temp[mask].tolist()
            remaining_records = len(coordinates.loc[coordinates[subset_random_values.keys()].isin(subset_random_values).all(axis=1)])
            if remaining_records < minelements:
                del subset_random_values[theatt]
    return subset_random_values
      
def score_current_subset(coordinates,probs,outcomes,penalty,current_subset,direction='positive'):
    if current_subset:
        to_choose = coordinates[current_subset.keys()].isin(current_subset).all(axis=1)
        temp_df=pd.concat([coordinates.loc[to_choose], outcomes[to_choose], pd.Series(data=probs[to_choose],index=outcomes[to_choose].index,name='prob')],axis=1)
    else:
        temp_df= pd.concat([coordinates, outcomes, pd.Series(data=probs,index=outcomes.index,name='prob')],axis=1)
    thesum = temp_df.iloc[:,-2].sum()
    theprobs = temp_df.iloc[:,-1]
    current_q_mle = binary_search_on_slopeterm(thesum,theprobs)
    if ((direction == 'positive') & (current_q_mle < 1)) | ((direction != 'positive') & (current_q_mle > 1)):
        current_q_mle = 1
    # totalpenalty = penalty * sum of list lengths in current_subset
    totalpenalty = 0
    for i in current_subset.values():
        totalpenalty += len(i)
    totalpenalty *= penalty  
    penalized_score = compute_score_given_q(thesum,theprobs,totalpenalty,current_q_mle)
    #print "In score_current_subset, current_score = ",penalized_score+totalpenalty,"-",totalpenalty,"=",penalized_score
    return penalized_score

def md_scan(coordinates,probs,outcomes,penalty,num_iters,direction='positive',minelements=0, verbose = False):
    best_subset = {}
    best_score = -1E10
    all_scores = []
    all_subsets = []
    for i in range(num_iters):
        flags = np.empty(len(coordinates.columns))
        flags.fill(0)
        current_subset = mk_subset_random_values(coordinates,np.random.rand(),minelements)
        #current_subset = mk_subset_all_values(coordinates) if i == 0 else mk_subset_random_values(coordinates,np.random.rand(),minelements)
        current_score = score_current_subset(coordinates,probs,outcomes,penalty,current_subset,direction)
        #print("Starting subset with score of",current_score,":")
        #print(current_subset)
        while flags.sum() < len(coordinates.columns):
            attribute_number_to_scan = np.random.choice(len(coordinates.columns))
            while flags[attribute_number_to_scan]:
                attribute_number_to_scan = np.random.choice(len(coordinates.columns))
            attribute_to_scan = coordinates.columns.values[attribute_number_to_scan]
            #print 'SCANNING:',attribute_to_scan
            if attribute_to_scan in current_subset:
                del current_subset[attribute_to_scan]  # HERE!!! Now we must replace current_subset with temp_subset below
            aggregates,thresholds,allsum,allprobs = get_aggregates(coordinates,probs,outcomes,current_subset,attribute_to_scan,penalty,direction)
            temp_names,temp_score=choose_aggregates(aggregates,thresholds,penalty,allsum,allprobs,direction)
            temp_subset = current_subset.copy()
            if temp_names: # if temp_names is not empty (or null)
                temp_subset[attribute_to_scan]=temp_names
            temp_score =  score_current_subset(coordinates,probs,outcomes,penalty,temp_subset,direction)
            #print("Temp subset with score of",temp_score,":")
            #print(temp_subset)
            if temp_score > current_score+1E-6:
                flags.fill(0)
            elif temp_score < current_score-1E-6:
                print("WARNING SCORE HAS DECREASED from",current_score,"to",temp_score)
            flags[attribute_number_to_scan] = 1            
            current_subset = temp_subset
            current_score = temp_score
        if verbose:
            print("Subset found on iteration",i+1,"of",num_iters,"with score",current_score)
            print(current_subset)
        all_scores.append(current_score)
        all_subsets.append(current_subset)
        if (current_score > best_score):
            best_subset = current_subset.copy()
            best_score = current_score
            #print "Best score is now",best_score
        #else:
            #print "Current score of",current_score,"does not beat best score of",best_score
    return [best_subset,best_score,all_scores,all_subsets]

def compare_control_subset(treatment, 
                           treatment_outcomes,
                           controls, 
                           control_outcomes,
                           subset,
                           brief=False,
                           verbose = False):

    if (subset):
        treatment_to_choose = treatment[subset.keys()].isin(subset).all(axis=1)
        treatment_df = treatment[treatment_to_choose]
        treatment_outcomes_df = treatment_outcomes[treatment_to_choose]
        controls_to_choose = controls[subset.keys()].isin(subset).all(axis=1)
        control_df = controls[controls_to_choose]
        control_outcomes_df = control_outcomes[controls_to_choose]
    else:
        treatment_df = treatment
        treatment_outcomes_df = treatment_outcomes
        control_df = controls
        control_outcomes_df = control_outcomes
    if verbose:
        print(treatment_outcomes_df.count(),'treatment individuals with',100*treatment_outcomes_df.mean(),'% test positives')   
        print(control_outcomes_df.count(),'control individuals with',100*control_outcomes_df.mean(),'% test positives')
    for theatt in treatment:
        for thevalue in treatment[theatt].unique():
            count1a = treatment_df[treatment_df[theatt] == thevalue].iloc[:,0].count()
            count1b = treatment_df.iloc[:,0].count()-count1a
            count2a = control_df[control_df[theatt] == thevalue].iloc[:,0].count()
            count2b = control_df.iloc[:,0].count()-count2a
            odds, p = scs.fisher_exact([[count1a,count2a],[count1b,count2b]])
            if p < .05:
                if brief:
                    print(theatt,thevalue,"<" if (count1a*1.0/(count1a+count1b))<(count2a*1.0/(count2a+count2b)) else ">")
                else:
                    if verbose:
                      print("Attribute =",theatt,", Value =",thevalue,", p =",p,":",count1a,"of",count1a+count1b,"(",count1a*100.0/(count1a+count1b),"%)","vs.",count2a,"of",count2a+count2b,"(",count2a*100.0/(count2a+count2b),"%)")
            #else:
            #    print(f"Not significant p-value: {p}", "Attribute =",theatt,", Value =",thevalue)
    return

def evaluate_scan(treatments,
                  probs,
                  outcomes,
                  controls,
                  control_outcomes,
                  subset_scan_penalty=0.1,
                  subset_scan_num_iters=50,
                  direction='positive',
                  minelements=100,
                  verbose = False):
    start_time = time.time()
    current_subset,current_score,all_scores,all_subsets = md_scan(treatments,probs,outcomes,subset_scan_penalty,subset_scan_num_iters,direction,minelements, verbose = verbose)
    if verbose:
        print("Required time = ",time.time()-start_time,"seconds")
        print('Found subset with score of',current_score)
        print(current_subset)
    output_dataframe = pd.DataFrame({'subset' : all_subsets,
                                 'score': all_scores})
    output_dataframe['iteration'] = output_dataframe.index + 1
    if verbose:
        print("Summary statistics:")
    if current_subset:
        to_choose = treatments[current_subset.keys()].isin(current_subset).all(axis=1)
        temp_df = treatments[to_choose]
        temp_outcomes = outcomes[to_choose]
        temp_probs = probs[to_choose]
    else:
        temp_df = treatments
        temp_outcomes = outcomes
        temp_probs = probs
    if verbose:
        print('Number of people:',temp_outcomes.count(),'Proportion:',temp_outcomes.mean(),'Expected proportion:',temp_probs.mean())
    try:
        if verbose:      
            print("Comparing subset to control group...")
        compare_control_subset(treatments, 
                               outcomes,
                               controls,
                               control_outcomes,
                               current_subset,
                               verbose = verbose)
    except ValueError:
        if verbose:
            print("Error: No objects to concatenate") 
    return output_dataframe
  
  

def run_HTE(input_dat, index, n_iters, penalty, verbose = False):
    DataP = input_dat.copy()
    X_train = pd.get_dummies(DataP.drop([treatment_var,outcome_var],axis=1),drop_first=True)
    y_train = DataP.loc[:,treatment_var]
    logistic.fit(X_train, y_train)
    DataP[treatment_var+'_predicted_prob'] = logistic.predict_proba(X_train)[:,1]
    DataP_treatment = DataP[DataP[treatment_var]==1]
    DataP_control = DataP[DataP[treatment_var]==0]    
    del DataP_control[treatment_var]
    del DataP_treatment[treatment_var]
    # To obtain average treatment effect on the treated, weight control observations by Pr(treated)/1-Pr(treated).
    DataP_control_weights = (DataP_control.loc[:,treatment_var+'_predicted_prob'])/(1-DataP_control.loc[:,treatment_var+'_predicted_prob'])
    del DataP_control[treatment_var+'_predicted_prob']
    del DataP_treatment[treatment_var+'_predicted_prob']
    DataP_treatment_outcomes = DataP_treatment[outcome_var]
    DataP_control_outcomes = DataP_control[outcome_var]
    del DataP_control[outcome_var]
    del DataP_treatment[outcome_var]
    # Calculate expected outcomes (probability of test positive using the model learned from the control group) 
    # for each treatment observation.
    # Adapting code here to deal with situations where control or treatment data does not contain any levels of categorical variable due to permutations
    control_dummies = pd.get_dummies(DataP_control,drop_first=True)
    treatment_dummies = pd.get_dummies(DataP_treatment, drop_first = True)
    treatment_dummies_fixed, control_dummies_fixed = treatment_dummies.align(control_dummies, join = 'outer', axis = 1) 
    control_dummies_fixed = control_dummies_fixed.fillna(0)
    treatment_dummies_fixed = treatment_dummies_fixed.fillna(0)
    logistic.fit(control_dummies_fixed,DataP_control_outcomes,sample_weight=DataP_control_weights)
    DataP_treatment_probs = logistic.predict_proba(treatment_dummies_fixed)[:,1]
    # Converting DataP_treatment_probs to a Pandas series
    DataP_treatment_probs = pd.Series(DataP_treatment_probs)
    # adjust for constant multiplicative shift in odds from control to treatment (thus testing the hypothesis of homogenous effect as opposed to no effect)
    delta = binary_search_on_slopeterm(DataP_treatment_outcomes.sum(),DataP_treatment_probs)
    if verbose:
        print("Shifting odds by constant multiplicative factor",delta,"before running HTE-Scan")
    shift_odds = lambda x: delta*x/(delta*x+1-x)
    DataP_treatment_probs = shift_odds(DataP_treatment_probs)
    # reshaping DataP_treatment_probs
    DataP_treatment_probs = pd.DataFrame(DataP_treatment_probs)
    DataP_treatment_probs = DataP_treatment_probs.values
    DataP_treatment_probs = DataP_treatment_probs.ravel()
    hte_out = evaluate_scan(DataP_treatment, 
        DataP_treatment_probs, 
        DataP_treatment_outcomes,
        DataP_control, 
        DataP_control_outcomes,
        subset_scan_num_iters=n_iters,
        subset_scan_penalty=penalty,
        verbose = verbose)
    hte_out['run'] = index
    return hte_out



def permute_df(input_df, treatment_var):
    output_df = input_df.copy()
    output_df[treatment_var] = np.random.permutation(output_df[treatment_var])
    return output_df





ds = pd.read_csv("BHDDH_HTE.csv",dtype=str)

outcome_var = 'EVENT'

treatment_var = 'TREATED'

ds[outcome_var]=pd.to_numeric(ds[outcome_var])
ds[treatment_var]=pd.to_numeric(ds[treatment_var])

ds = ds.loc[:,['AGE_GROUP','SEX','RACE','EDUCATION','ARRESTED','PRIMARYDRUG',outcome_var,treatment_var]]

for c in ds.columns:
    ds[c] = ds[c].fillna(ds[c].mode()[0])
    
ds_pos = ds.copy()
ds_neg = ds.copy()

ds_neg[outcome_var] = 1-ds_neg[outcome_var]

# OBSERVED

obs_pos = run_HTE(ds_pos, "observed", 100, 0.5)

obs_neg = run_HTE(ds_neg, "observed", 100, 0.5)

obs_pos.to_csv("res_files/observed_results_pos2.csv")

obs_neg.to_csv("res_files/observed_results_neg2.csv")

### POSITIVE RUNS

pos_results_list = []

for i in range(1,1001):
    try:
      perm_data = permute_df(ds_pos, treatment_var)
      new_res = run_HTE(perm_data, i, n_iters = 100, penalty = 0.5, verbose = False)
      pos_results_list.append(new_res)
      print("XXXXXXXXXXXXXXXXXXXX")
      print("PERMUTATION",i,"COMPLETED")
      print("XXXXXXXXXXXXXXXXXXXX")
    except: 
      print("00000000000000000000")
      print("Error occurred in permutation",i)
      print("00000000000000000000")

final_list_pos = pd.concat(pos_results_list)

print("YYYYYYYYYYYYYYYYYYYY")
print("POSITIVE RESULTS COMPLETED!!")
print("YYYYYYYYYYYYYYYYYYYY")

final_list_pos.to_csv("res_files/perm1000_pos.csv")

### NEGATIVE RUNS

neg_results_list = []

for i in range(1,1001):
    try:
      perm_data = permute_df(ds_neg, treatment_var)
      new_res = run_HTE(perm_data, i, n_iters = 100, penalty = 0.5, verbose = False)
      neg_results_list.append(new_res)
      print("XXXXXXXXXXXXXXXXXXXX")
      print("PERMUTATION",i,"COMPLETED")
      print("XXXXXXXXXXXXXXXXXXXX")
    except: 
      print("00000000000000000000")
      print("Error occurred in permutation",i)
      print("00000000000000000000")

final_list_neg = pd.concat(neg_results_list)

print("YYYYYYYYYYYYYYYYYYYY")
print("NEGATIVE RESULTS COMPLETED!!")
print("YYYYYYYYYYYYYYYYYYYY")

final_list_neg.to_csv("res_files/perm1000_neg.csv")
