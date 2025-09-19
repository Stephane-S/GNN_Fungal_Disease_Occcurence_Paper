import random
import itertools
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import degree

from aug_utils import *


def build_HGNN_aug(sub_df_norm, masking_data, distance_txt_file, dist_threshold_choice = [20, 25, 30, 35, 40], aug_multiplier = 3):

    farm_ids, dates = extract_MetaData(sub_df_norm)

    # list to store the graph for each farm and date
    graphs = []
    unique_farms_idx = {}
    aug_proof_concept = {}
    aug_proof_concept_iter = -1

    # loop through each farm and date
    for (farm_id, date) in itertools.product(farm_ids, dates):

        train_masks, valid_masks, test_masks = create_masks(masking_data, farm_id)
        day_farm_data = sub_df_norm[(sub_df_norm['FarmID'] == farm_id) & (sub_df_norm['Date'] == date)]
        if day_farm_data.size > 0:
            aug_proof_concept_iter +=1 # only for visualisation purposes

            for aug in range(aug_multiplier):

                removed_plant_ids = []

                # we select a distance threshold and extract the relevant plant combinations for the edges
                dist_threshold = random.choice(dist_threshold_choice)
                donor, receiver = get_plant_distances(farm_id, dist_threshold, distance_txt_file, day_farm_data['Plant_ID'].tolist())

                # first iteration is not an augmentation task, we preserve as much data as possible for further alterations
                if aug == 0:
                    node_drop_prob = 0
                    edge_drop_prob = 0
                    edge_add_nbr = 0

                    squamosa_rate_array = day_farm_data['cote_b_lactucae'].to_numpy()
                    squamosa_rate_label = np.rint(squamosa_rate_array) # round up cote squamosa to nearest integer
                    #print(f'nbr label:', len(squamosa_rate_label))
                    day_farm_data_subset = day_farm_data # for compatibility with the aug operations

                    plant_ids = sub_df_norm[(sub_df_norm['FarmID'] == farm_id) & (sub_df_norm['Date'] == date)]['Plant_ID'].tolist()
                    #print(f'nbr ids:', len(plant_ids))
                    train_masks_subgraph = [train_masks[i - 1] for i in plant_ids]
                    valid_masks_subgraph = [valid_masks[i - 1] for i in plant_ids]
                    test_masks_subgraph = [test_masks[i - 1] for i in plant_ids]

                    if len(plant_ids) != max(plant_ids):                      
                        day_farm_data_subset, donor, receiver = clean_ID_sparse(day_farm_data_subset, "Plant_ID", donor, receiver)
                
                # iteration > 1 are augmentation tasks, we apply a random assortment of operations
                else:
                    day_farm_data_subset = day_farm_data.copy(deep=True) # we want to keep the original values so each iteration is independant

                    node_drop_prob = random.choice([0, 0.1, 0.2])
                    edge_drop_prob = random.choice([0, 0.1, 0.2])
                    edge_add_nbr = random.choice([0, 1, 2, 3, 4, 5])

                    #node drop
                    #using the dataframe, we remove n rows based on the node drop probability
                    nbr_nodes_drop = int(node_drop_prob * len(day_farm_data_subset))
                    if nbr_nodes_drop > 0:
                        
                        # sample the df n times and remove the picked rows
                        day_farm_data_subset_filtered = day_farm_data_subset[~day_farm_data_subset['Plant_ID'].isin(conserved_node_ids)] # we keep 20% most connected nodes safe
                        removed_rows = day_farm_data_subset_filtered.sample(nbr_nodes_drop)
                        removed_plant_ids = (removed_rows['Plant_ID'].tolist())
                        day_farm_data_subset = day_farm_data_subset.drop(removed_rows.index)

                        # remove the donor-receiver pairs that include the removed nodes
                        plant_edges = list(zip(donor, receiver))
                        temp_edges = []
                        for don, rec in plant_edges:
                            if don not in removed_plant_ids and rec not in removed_plant_ids:
                                temp_edges.append((don, rec))
                        
                        donor, receiver = zip(*temp_edges)
                        
                        squamosa_rate_array = day_farm_data_subset['cote_b_lactucae'].to_numpy()
                        squamosa_rate_label = np.rint(squamosa_rate_array) # round up cote squamosa to nearest integer
                    
                    else:
                        day_farm_data_subset = day_farm_data
                        squamosa_rate_array = day_farm_data_subset['cote_b_lactucae'].to_numpy()
                        squamosa_rate_label = np.rint(squamosa_rate_array) # round up cote squamosa to nearest integer


                    #now we have to re_ID come of the plants so that there is no indexing errors down the line
                    day_farm_data_subset, donor, receiver = clean_ID_sparse(day_farm_data_subset, "Plant_ID", donor, receiver)

                    node_id_list = day_farm_data_subset["Plant_ID"].tolist()
                    
                    # extract appropriate masks
                    #####
                    train_masks_subgraph = [train_masks[i - 1] for i in node_id_list]
                    valid_masks_subgraph = [valid_masks[i - 1] for i in node_id_list]
                    test_masks_subgraph = [test_masks[i - 1] for i in node_id_list]

                    #edge drop
                    nbr_edges_drop = int(edge_drop_prob * len(donor))
                    plant_edges = list(zip(donor, receiver))

                    if nbr_edges_drop > 0:
                        random.shuffle(plant_edges)
                        plant_edges = plant_edges[:len(plant_edges)-nbr_edges_drop] # we remove n random edge pairs

                    # edge add
                    if edge_add_nbr > 0:
                        patience = 10
                        remaining_edges = edge_add_nbr
                        node_id_list = day_farm_data_subset["Plant_ID"].tolist()
                        while patience > 0 and remaining_edges > 0:
                            new_donor = random.choice(node_id_list)
                            new_receiver = random.choice(node_id_list)
                            temp_edge = (new_donor, new_receiver)
                            if temp_edge in plant_edges or new_donor == new_receiver:
                                patience -= 1
                            else:
                                plant_edges.append(temp_edge)
                                remaining_edges -= 1
                                patience = 10
                    
                    donor, receiver = zip(*plant_edges)

                # shared operations between augmented and non-augmented graphs

                donor_np = np.array(donor)
                receiver_np = np.array(receiver)

                # we need edges going both ways 
                from_plant1 = torch.tensor(donor_np, dtype = int)
                to_plant2 = torch.tensor(receiver_np, dtype = int)

                plant_edges = torch.concat((from_plant1, to_plant2)).reshape(-1, len(from_plant1)).long()

                #remove vars we dont want in the graph (the metadata)
                day_farm_data_clean = day_farm_data_subset.drop(['FarmID', 'Plant_ID', 'Date', 'cote_b_lactucae'], axis=1)

                #convert class values to tensor
                squamosa_rate_label = torch.tensor(squamosa_rate_label, dtype = int)

                # create plant node as tensor
                plants_tensor = torch.tensor(np.array(day_farm_data_clean), dtype = float)

                graph = Data(x=plants_tensor, edge_index=plant_edges, y=squamosa_rate_label)

                transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
                graph = transform(graph)

                graph.train_mask = torch.Tensor(train_masks_subgraph)
                graph.val_mask = torch.Tensor(valid_masks_subgraph)
                graph.test_mask = torch.Tensor(test_masks_subgraph)
                
                graphs.append(graph)


                ##################

                if aug == 0:
                    degrees = degree(graph.edge_index[0], num_nodes=graph.num_nodes, dtype=torch.float)
                    node_degrees = list(enumerate(degrees.tolist()))
                    # Sort nodes by degree in descending order
                    sorted_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)
                    # Extract node IDs from sorted list
                    sorted_node_ids = [node_id for node_id, _ in sorted_nodes]
                    conserved_node_ids = sorted_node_ids[:round(0.2*len(sorted_node_ids))]

                # used to save a graph of each farm for visualisation purposes only
                if farm_id not in unique_farms_idx:
                    unique_farms_idx[farm_id] = graph
                
                if aug_proof_concept_iter == 0:
                    aug_proof_concept[aug] = graph

    return (graphs, unique_farms_idx, aug_proof_concept)