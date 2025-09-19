import pandas as pd
import numpy as np

# extract the farm IDs, dates, and plants information from the dataframes
def extract_MetaData(lettuce_df):
    farm_ids = lettuce_df['FarmID'].unique()
    dates = lettuce_df['Date'].unique()
    return(farm_ids, dates)


# for a given farmID and a dataset masking data, return the appropriate train/valid/test masks 
def create_masks(masking_data, farm_id):

    filtered_lists = [[y for x, y in sublist if x == farm_id] for sublist in masking_data]

    total_len = sum(len(inner_list) for inner_list in filtered_lists)
    train_masks = [0] * total_len
    valid_masks = [0] * total_len
    test_masks = [0] * total_len

    train_masks = [1 if i + 1 in filtered_lists[0] else 0 for i in range(total_len)]
    valid_masks = [1 if i + 1 in filtered_lists[1] else 0 for i in range(total_len)]
    test_masks = [1 if i + 1 in filtered_lists[2] else 0 for i in range(total_len)]
    
    return train_masks, valid_masks, test_masks


# given a fieldID, a distance threshold and the distance between all pairs of plants in the given field, 
#create donor and receiver arrays. Also makes sure only plants observed on the day (valid ids) are returned to match the graph data.
def get_plant_distances(fieldID, threshold, distance_txt_file, valid_ids):
  array1 = []
  array2 = []
  for line in distance_txt_file:
      farmID, plant1, plant2, dist = line.split(',', 3)

      # conditions: find the right field,
      # plants are close enough together
      # avoids error if there was a partial observation that day (less nodes than the dist matrix has)
      # removes the last overflowing extra node if partial observation

      if (int(farmID) == fieldID and
      float(dist) <= threshold and
      all(int(x) in valid_ids for x in [plant1, plant2]) and
      ((max(int(plant1), int(plant2)) < max(valid_ids)))):
          array1.append(int(plant1))
          array2.append(int(plant2))

  return array1, array2


# given a graph and communities, return a dict of community_ids and subgraphs 
def split_graph_into_communities(graph, communities):
    community_subgraphs = {}
    
    for community_id, nodes in enumerate(communities):
        subgraph = graph.subgraph(nodes)
        community_subgraphs[community_id] = subgraph
    
    return community_subgraphs


def clean_ID_sparse(day_farm_data_subset, ID_column, donor, receiver):
    max_index = len(day_farm_data_subset[ID_column].tolist())
    node_id_list = day_farm_data_subset[ID_column].tolist()
    gaps = []
    for i in range(max_index):
        if i not in node_id_list:
            gaps.append(i)
    
    if len(gaps) > 0:
        id_to_replace = [i for i in node_id_list if i >= max_index]
        
        #gaps and id_to_replace will always be the same size logically
        for idx in range(len(gaps)):
            old_id = id_to_replace[idx]
            new_id = gaps[idx]
            #day_farm_data_subset['Plant_ID'] = day_farm_data_subset['Plant_ID'].replace(old_id, new_id)
            day_farm_data_subset.loc[day_farm_data_subset[ID_column] == old_id, ID_column] = new_id

            donor = [new_id if x == old_id else x for x in donor]
            receiver = [new_id if x == old_id else x for x in receiver]

    return day_farm_data_subset, donor, receiver