import random
import itertools
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

from aug_utils import *

def create_graph(donor, receiver, day_farm_data_subset, squamosa_rate_label, train_masks, valid_masks, test_masks):
    donor_np = np.array(donor)
    receiver_np = np.array(receiver)

    # we need edges going both ways 
    from_plant1 = torch.tensor(donor_np, dtype = int)
    to_plant2 = torch.tensor(receiver_np, dtype = int)

    plant_edges = torch.concat((from_plant1, to_plant2)).reshape(-1, len(from_plant1)).long()

    temp = day_farm_data_subset['Plant_ID'].to_list()
    #remove vars we dont want in the graph (the metadata)
    day_farm_data_clean = day_farm_data_subset.drop(['FarmID', 'Plant_ID', 'Date', 'cote_b_lactucae'], axis=1)

    #convert class values to tensor
    squamosa_rate_label = torch.tensor(squamosa_rate_label, dtype = int)

    # create plant node as tensor
    plants_tensor = torch.tensor(np.array(day_farm_data_clean), dtype = float)

    graph = Data(x=plants_tensor, edge_index=plant_edges, y=squamosa_rate_label)

    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
    graph = transform(graph)

    graph.train_mask = torch.Tensor(train_masks)
    graph.val_mask = torch.Tensor(valid_masks)
    graph.test_mask = torch.Tensor(test_masks)

    return graph, len(plant_edges[0])

def build_HGNN_aug_sliding_threshold(sub_df_norm, masking_data, distance_txt_file, minimum_distance, maximum_distance, step):

    farm_ids, dates = extract_MetaData(sub_df_norm)

    # list to store the graph for each farm and date
    graphs = []
    unique_farms_idx = {}
    aug_proof_concept = {}
    aug_proof_concept_iter = -1
    nbr_total_nodes = 0
    nbr_total_edges = 0


    # loop through each farm and date
    for (farm_id, date) in itertools.product(farm_ids, dates):

        train_masks, valid_masks, test_masks = create_masks(masking_data, farm_id)
        plant_ids = sub_df_norm[(sub_df_norm['FarmID'] == farm_id) & (sub_df_norm['Date'] == date)]['Plant_ID'].tolist()
        train_masks_subgraph = [train_masks[i - 1] for i in plant_ids]
        valid_masks_subgraph = [valid_masks[i - 1] for i in plant_ids]
        test_masks_subgraph = [test_masks[i - 1] for i in plant_ids]

        day_farm_data = sub_df_norm[(sub_df_norm['FarmID'] == farm_id) & (sub_df_norm['Date'] == date)]
        if day_farm_data.size > 0:
            aug_proof_concept_iter +=1 # only for visualisation purposes

            median_distance_threshold = int(round((minimum_distance+maximum_distance)/2))

            donor, receiver = get_plant_distances(farm_id, median_distance_threshold, distance_txt_file, day_farm_data['Plant_ID'].tolist())

            if len(plant_ids) != max(plant_ids):
                day_farm_data, donor, receiver = clean_ID_sparse(day_farm_data, "Plant_ID", donor, receiver)

            # first iteration is not an augmentation task, we preserve as much data as possible for training
            squamosa_rate_array = day_farm_data['cote_b_lactucae'].to_numpy()
            squamosa_rate_label = np.rint(squamosa_rate_array) # round up cote squamosa to nearest integer
            day_farm_data_subset = day_farm_data # for compatibility with the aug operations

            graph, nbr_edges = create_graph(donor, receiver, day_farm_data_subset, squamosa_rate_label, train_masks_subgraph, valid_masks_subgraph, test_masks_subgraph)
                
            median_graph_edges = nbr_edges
            graphs.append(graph)

            if farm_id not in unique_farms_idx:
                unique_farms_idx[farm_id] = graph
                
            if aug_proof_concept_iter == 0:
                aug_proof_concept[median_distance_threshold] = graph

            
            for dist_threshold in range(minimum_distance, median_distance_threshold, step):
                donor, receiver = get_plant_distances(farm_id, dist_threshold, distance_txt_file, day_farm_data['Plant_ID'].tolist())

                # first iteration is not an augmentation task, we preserve as much data as possible for training
                squamosa_rate_array = day_farm_data['cote_b_lactucae'].to_numpy()
                squamosa_rate_label = np.rint(squamosa_rate_array) # round up cote squamosa to nearest integer
                day_farm_data_subset = day_farm_data # for compatibility with the aug operations

                if len(donor) > 0:
                    graph, nbr_edges = create_graph(donor, receiver, day_farm_data_subset, squamosa_rate_label, train_masks_subgraph, valid_masks_subgraph, test_masks_subgraph)
                else:
                    nbr_edges = 0
                
                if nbr_edges >= 0.5 * median_graph_edges:
                    graphs.append(graph)

                # used to save a graph of each farm for visualisation purposes only
                if farm_id not in unique_farms_idx:
                    unique_farms_idx[farm_id] = graph
                
                if aug_proof_concept_iter == 0:
                    aug_proof_concept[dist_threshold] = graph
            
            for dist_threshold in range(median_distance_threshold + 1, maximum_distance+ 1, step):
                donor, receiver = get_plant_distances(farm_id, dist_threshold, distance_txt_file, day_farm_data['Plant_ID'].tolist())

                # first iteration is not an augmentation task, we preserve as much data as possible for training
                squamosa_rate_array = day_farm_data['cote_b_lactucae'].to_numpy()
                squamosa_rate_label = np.rint(squamosa_rate_array) # round up cote squamosa to nearest integer
                day_farm_data_subset = day_farm_data # for compatibility with the aug operations

                graph, nbr_edges = create_graph(donor, receiver, day_farm_data_subset, squamosa_rate_label, train_masks_subgraph, valid_masks_subgraph, test_masks_subgraph)
                
                if nbr_edges <= 1.5 * median_graph_edges:
                    graphs.append(graph)

                # used to save a graph of each farm for visualisation purposes only
                if farm_id not in unique_farms_idx:
                    unique_farms_idx[farm_id] = graph
                
                if aug_proof_concept_iter == 0:
                    aug_proof_concept[dist_threshold] = graph

    return (graphs, unique_farms_idx, aug_proof_concept)