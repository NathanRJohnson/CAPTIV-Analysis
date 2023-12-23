import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy

def is_leaf(node_id, children_left, children_right):
  """
  @returns True if the node is a leaf, False otherwise
  node_id: id of current node
  children-left: list of nodes to the left of this node
  children-right: list of nodes to the right of this node
  """
  return children_left[node_id] == children_right[node_id]

def build_travel(node_id, stack, branches, children_left, children_right):
  """
  Uses pre-order travel to build a list of branches
  node_id: id of current node
  stack: inverse-stack of current branch up to node_id
  branches: list branches
  """
  # add current node to the inverse stack
  stack.append(node_id)

  # base case - leaf node
  # deep copy the current stack into branches, and
  # remove the leaf node off the current stack
  if is_leaf(node_id, children_left, children_right):
    branches.append(deepcopy(stack))
    stack.pop(-1)
    return

  # recursive case - split node
  # travel to the right, then the left. Once this node is no longer a part of
  # the current branch, pop it off the stack
  build_travel(children_left[node_id], stack, branches, children_left, children_right)
  build_travel(children_right[node_id], stack, branches, children_left, children_right)
  stack.pop(-1)

def tree_to_array(tree):
  """
  Uses pre-order travel to build a list of branches
  :param tree: A scikit-learn decision tree
  """
  branches = []
  stack = []
  children_left = tree.children_left
  children_right = tree.children_right

  build_travel(0, stack, branches, children_left, children_right)
  return branches

def predict_with_array(tree, branches, point, default_label):
  for branch in branches:
    # print("New branch")
    depth = len(branch)
    for i in range(depth):
      # reached our leaf node
      if i == depth - 1:
        # print(tree.value[branch[i]][0], np.argmax(tree.value[branch[i]][0]))
        return [np.argmax(tree.value[branch[i]][0])]
      
      split_feature = tree.feature[branch[i]]
      split_threshold = tree.threshold[branch[i]]
      # print("FEATURE:", split_feature)
      # print("SPLIT: ", point[split_feature], "<=", split_threshold)
      if (point[split_feature] <= split_threshold):
        # check left
        # print("LEFT  expected:", tree.children_left[branch[i]], " actual:", branch[i+1])
        
        if tree.children_left[branch[i]] != branch[i+1]:
          # our branch does not match our expected left move,
          # so it must be the wrong branch
          break
      else:
        # check right
        # print("RIGHT  expected:", tree.children_right[branch[i]], " actual:", branch[i+1])
        if tree.children_right[branch[i]] != branch[i+1]:
          # our branch does not match our expected right move,
          # so it must be the wrong branch
          break
  
  # TODO at this point if we run out of branches, we return the maximum likelihood estimate
  return [default_label]

def prune_after(node_id, tree):
  '''
  removes branch after i node
  '''
  # check left and right for node_id
  children_left = tree.children_left
  children_right = tree.children_right
  
  children_left[node_id] = -1
  children_right[node_id] = -1

def prune(tree):
  # step 1: split the tree into an array
  # don't know what the array is for yet, but I'm sure it will be handy 
  # if we need to look at node specific values
  branches = tree_to_array(tree)

  # step2: check each branch against the chop decider
  # mayyy have to check the stick set of this list
  # if it's chopping time, than it's chopping time
  removal_nodes = []
  for branch in branches:
    # stick_set = get_stick_set(branch)
    # for stick in stick_set:
    if is_bad_branch(branch): 
      removal_nodes.append(branch[1])

def boundary_travel(node_id, boundaries, tree, children_left, children_right):
  
  if is_leaf(node_id, children_left, children_right):
    return
  
  feature = tree.feature[node_id]
  if not feature in boundaries:
    boundaries[feature] = []
  if not tree.threshold[node_id] in boundaries[feature]:
    boundaries[feature].append(tree.threshold[node_id])
  boundary_travel(children_left[node_id], boundaries, tree, children_left, children_right)
  boundary_travel(children_right[node_id], boundaries, tree, children_left, children_right)

def get_boundaries(tree):
  boundaries = {}
  children_left = tree.children_left
  children_right = tree.children_right
  
  boundary_travel(0, boundaries, tree, children_left, children_right)
  for key in boundaries:
    boundaries[key].sort()
  return boundaries

def build_bucketed_data(training_data, boundaries):
  bucket_data = []
  sorted_keys = sorted(boundaries)
  # print(sorted_keys)
  for x_obs in range(len(training_data)):
    bucket_data.append([])
    
    # each index in the array is a point, each value is an array representing buckets
    # each index in the sub array is a feature of that point, each value is the corresponding bucket
    for feature in sorted_keys:
      # print(training_data[x_obs], boundaries.get(x_feature))
      bucket = 0
      was_set = False
      # print(feature, ":", boundaries.get(feature))
      for bound in boundaries.get(feature):
        if training_data[x_obs][feature] <= bound:
          # print(training_data[x_obs][feature],"is smaller than", bound)
          bucket_data[x_obs].append(bucket)
          was_set = True
          break
        else:
          bucket += 1
      if not was_set:
        bucket_data[x_obs].append(bucket)
      # I don't know why this is here
      # if we're not out of bounds on features (why would we be...)
      # if len(bucket_data[x_obs]) <= feature:
      #   bucket_data[x_obs].append(bucket)
  return bucket_data
#index is a branch
# key:feature, -1 is label
# value: list of buckets
def build_feature_bucket_branches(tree, branches, boundaries, branch_labels):
  feature_bucket_list = []
  left_children = tree.children_left
  right_children = tree.children_right
  ibranch = 0
  for branch in branches:
    branch_feat_bucket_list = {}
    branch_feat_bucket_list[-1] = [branch_labels[ibranch]]

    # skip leaf
    for x_node in range(len(branch) - 1):
      feat = tree.feature[branch[x_node]]
      thresh = tree.threshold[branch[x_node]]

      # Maybe if never seen this node, add all of the buckets??
      if not feat in branch_feat_bucket_list.keys():
        branch_feat_bucket_list[feat] = [i for i in range(len(boundaries[feat]) + 1)]
      # Then if we go left, remove all larger buckets
      # But if go right, remove this and all smaller buckets...
      # YUP
      
      split_boundary = boundaries.get(feat).index(thresh)
      index = branch_feat_bucket_list[feat].index(split_boundary)
      ## At this point, if we're continuing left we need to remove the buckets to the right
      if branch[x_node+1] == left_children[branch[x_node]]:
        branch_feat_bucket_list[feat] = branch_feat_bucket_list[feat][:index+1]
      ## if we're going right next, get rid of that bucket and everything before it
      elif branch[x_node+1] == right_children[branch[x_node]]:
        branch_feat_bucket_list[feat] = branch_feat_bucket_list[feat][index+1:]
  
    feature_bucket_list.append(branch_feat_bucket_list)
    ibranch = ibranch + 1

  return feature_bucket_list

# deprecated
# def count_branch_observations(feat_bucket_branches, bucketed_data):
  branch_obs_counts = [0] * len(feat_bucket_branches)
  valid = True
  # pick an observation
  for obs in bucketed_data:
    branch_index = 0
    # pick a branch
    for f_b_branch in feat_bucket_branches:
      valid = True
      # test to see if the buckets of the features in the branch match the buckets of the features in the observations
      for feat in f_b_branch.keys():
        # if there is a mismatch, we mark the branch as the wrong one, and move on to the next
        if not obs[feat] in f_b_branch[feat]:
          valid = False
          break
      # otherwise if every feature has been checked, we can add this observation to the branch, and move onto the next observation
      if valid == True:
        branch_obs_counts[branch_index] += 1
        break

      branch_index += 1
    
  return branch_obs_counts

'''
@retuns an array of lists of lists
The index of the array is the feature
The key of the outer list is label
The key of the inner list is the bucket
The value of the inner list is the number of observations falling into that bucket
'''
def sum_bucket_counts(bucketed_data, labels):
  # add a map for each feature and the label column, which will hold a bucket:count value 
  bucket_count_sums = [{} for x in range(len(bucketed_data[0]) + 1)]
  i = 0
  for obs in bucketed_data:
    feature = 0
    for bucket in obs:
      # add bucket:count value for the features
      if not bucket in bucket_count_sums[feature]:
        bucket_count_sums[feature][bucket] = 0
      bucket_count_sums[feature][bucket] += 1
      feature += 1
    
    # add label:count value for the labels
    # print("BCS", bucket_count_sums[feature])
    if not labels[i] in bucket_count_sums[feature]:
      bucket_count_sums[feature][labels[i]] = 0
    bucket_count_sums[feature][labels[i]] += 1 

    i += 1
  return bucket_count_sums

# TODO properly test this thing because it may be weird
def count_branch_primary_observations(feature_bucket_branch, bucket_count_sums):
  '''
  @returns a dictionary representing a branch, where the keys are the features and the
  values are the total number of observations in the buckets of that feature for this branch.
  '''
  primary_branch_obs = {}
  ifeature = 0
  for feature in feature_bucket_branch:
    # print("FEATURE", feature)
    buckets = feature_bucket_branch[feature]
    # print("BUCKETS", buckets)
    for bucket in buckets:
      # print("BUCCCKET", bucket)
      if not feature in primary_branch_obs:
        primary_branch_obs[feature] = 0
        # primary_branch_obs[feature] += bucket_count_sums[feature].get(bucket)
        try:
          if bucket in bucket_count_sums[feature]:
            primary_branch_obs[feature] += bucket_count_sums[feature].get(bucket)
        except:
          print("ERROR", feature_bucket_branch, feature, buckets, bucket)
          # print(bucket_count_sums[feature], bucket_count_sums[feature].get(bucket))
          exit()
      # so it seems that the feature_bucket_tree and the obs counts are not alighned

    # if not feature.keys() in primary_branch_obs:
    #   primary_branch_obs[feature] = 0
    # for bucket in feature:
    #   primary_branch_obs[feature] += 1
  return primary_branch_obs

def get_branch_labels(tree, branches):
  branch_labels = []
  for branch in branches:
    leaf = branch[-1]
    # TODO in the event of a tie, it will put the first greatest value...
    branch_labels.append(np.argmax(tree.value[leaf]))
  
  return branch_labels

def get_tree_primary_obs(feature_bucket_branches, bucket_count_sums, branch_labels):
  '''
  @returns a list of dictionaries, where each element of the list represents a branch, and
  where the keys of each dictionary the features, and the value are the number of observations attached to that feature
  '''
  primary_obs_by_branch = []
  for x_branch in range(len(feature_bucket_branches)):
    primary_obs_by_branch.append(count_branch_primary_observations(feature_bucket_branches[x_branch], bucket_count_sums))
  return primary_obs_by_branch

# TODO need to handle the case where it's a tie
# can possibly set it to something like -1 because if there is a tie this branch is gonna go
def get_branch_observations(tree, branches):
  branch_observations_counts = []
  for branch in branches:
    leaf = branch[-1]
    # TODO in the event of a tie, it just picks either one
    branch_observations_counts.append(np.max(tree.value[leaf]))
  return branch_observations_counts

def get_expected_branch_observations_product(primary_obs_count, N):
  product = 1
  for key in primary_obs_count:
    product *= primary_obs_count[key] / N
  return product

def get_expected_obs_counts(tree_primary_obs, N):
  expected_counts = []
  for primary_obs in tree_primary_obs:
    expected_counts.append(N * get_expected_branch_observations_product(primary_obs, N))
  return expected_counts

def calc_standardized_residuals(observed_counts, expected_counts):
  standardized_residuals = []
  for i in range(len(observed_counts)):
    if expected_counts[i] == 0:
      standardized_residuals.append(0)
    else:
      standardized_residuals.append((observed_counts[i] - expected_counts[i]) / (expected_counts[i] ** 0.5))
  return standardized_residuals

def calc_max_likelihood_variance(tree_primary_obs, N):
  max_likelihoods = []
  for primary_obs in tree_primary_obs:
    max_likelihoods.append(1 - get_expected_branch_observations_product(primary_obs, N))
  return max_likelihoods

def calc_adjusted_residuals(standardized_residuals, max_likelihoods_variance):
  adjusted_residuals = []
  for i in range(len(standardized_residuals)):
    adjusted_residuals.append(standardized_residuals[i] / math.sqrt(max_likelihoods_variance[i]))
  return adjusted_residuals

def prune_uninteresting_branches(branches, adjusted_residuals):
  start_length = len(branches)
  for i in range(1, start_length + 1):
    if adjusted_residuals[start_length-i] < 1.96:
      branches.pop(start_length - i)

def make_noise(shape, mu, sigma):
  # sigma is standard deviation, and mu is the variance
  noise = np.random.normal(mu, sigma, shape)
  # may have to transform the data to np array
  # or not if I do it before the flatten
  return noise

def score(tree, branches, data, labels):
  # get maximum likihood of the labels
  counts = {}
  for label in labels:
    if label not in counts:
      counts[label] = 0
    counts[label] += 1

  win_label = max(counts)
  num_correct = 0
  for i in range(len(data)):
    predicted_label = predict_with_array(tree, branches, data[i], win_label)
    if predicted_label == labels[i]:
      num_correct += 1
  
  return num_correct / len(data)

def prune_tree(tree, x_train, y_train):
  branches = tree_to_array(tree)
  observed_counts =  get_branch_observations(tree, branches)

  boundaries = get_boundaries(tree)
  # print("BOUNDARIES", boundaries)

  bucketed_data = build_bucketed_data(x_train, boundaries)
  # print("ACTUAL", sub_data)
  # print("BUCKET", bucketed_data)
  bucket_sums = sum_bucket_counts(bucketed_data, y_train)

  branch_labels = get_branch_labels(tree, branches)
  # FBB contains -1s
  fbb = build_feature_bucket_branches(tree, branches, boundaries, branch_labels)
  
  # print("FBB", fbb)
  # print("BUCKET_SUMS", bucket_sums)
  tpo = get_tree_primary_obs(fbb, bucket_sums, branch_labels)
  # print("TPO", tpo)
  expected_counts = get_expected_obs_counts(tpo, len(x_train))

  sr = calc_standardized_residuals(observed_counts, expected_counts)
  clv = calc_max_likelihood_variance(tpo, len(x_train))
  adjusted_residuals = calc_adjusted_residuals(sr, clv)
  # print(adjusted_residuals)
  # print(branches)
  prune_uninteresting_branches(branches, adjusted_residuals)
  # print(branches)
  return branches

# -----

def predict_points(tree, branches, data, labels):
  counts = {}
  for label in labels:
    if label not in counts:
      counts[label] = 0
    counts[label] += 1

  win_label = max(counts)
  predictions = []
  for point in data:
    predictions.append(predict_with_array(tree, branches, point, win_label))
  return predictions