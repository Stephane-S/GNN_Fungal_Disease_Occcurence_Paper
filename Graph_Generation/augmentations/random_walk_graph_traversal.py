import random
import itertools
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import networkx as nx

from aug_utils import *

def random_walk(graph, max_depth, samples):
    result = []
    unused_nodes = [n for n in graph.nodes()]
    for _ in range(samples):
        attempts = len(unused_nodes)
        while attempts > 0:
            start_node = random.choice(unused_nodes)
            unused_nodes.remove(start_node)
            walk = [start_node]
            visited = set()
            for depth in range(max_depth):
                if depth == 0:
                    neighbors = [n for n in graph.neighbors(start_node) if n not in visited]
                else:
                    neighbors = [n for node in walk for n in graph.neighbors(node) if n not in visited]
                if neighbors:
                    walk.extend(neighbors)
                    visited.update(neighbors)
                else:
                    break
            if len(walk) > 3:
                result.append(walk)
                break
            else:
                attempts -= 1
    return result


def build_HGNN_random_walk(sub_df_norm, masking_data, distance_txt_file, dist_threshold_choice = [20, 25, 30, 35, 40], max_depth = 2, samples = 5):

    farm_ids, dates = extract_MetaData(sub_df_norm)

    # list to store the graph for each farm and date
    graphs = []
    unique_farms_idx = {}
    aug_proof_concept = {}
    aug_proof_concept_iter = -1
    nbr_total_nodes = 0
    nbr_total_edges = 0
    size_communities = []
    nbr_subgraphs = 0


    # loop through each farm and date
    for (farm_id, date) in itertools.product(farm_ids, dates):

        train_masks, valid_masks, test_masks = create_masks(masking_data, farm_id)
        day_farm_data = sub_df_norm[(sub_df_norm['FarmID'] == farm_id) & (sub_df_norm['Date'] == date)]
        if day_farm_data.size > 0:
            day_farm_data_subset = day_farm_data.copy(deep=True)
            aug_proof_concept_iter +=1 # only for visualisation purposes

            #get edges
            dist_threshold = random.choice(dist_threshold_choice)
            donor, receiver = get_plant_distances(farm_id, dist_threshold, distance_txt_file, day_farm_data['Plant_ID'].tolist())
            plant_edges = list(zip(donor, receiver))

            # columns that should not be features in the graph
            exclude_columns = ['FarmID', 'Plant_ID', 'Date', 'cote_b_lactucae']
            node_id_column = day_farm_data_subset['Plant_ID']
            node_features = day_farm_data_subset.drop(exclude_columns, axis=1).to_dict('records')
            

            graph_nx = nx.Graph()
            for node_id, features in zip(node_id_column, node_features):
                graph_nx.add_node(node_id, **features)
            
            for edge in plant_edges:
                graph_nx.add_edge(*edge)
            

            walks = random_walk(graph_nx, max_depth, samples)

            if len(walks) == 0:
                #nx.draw(graph_nx,pos=nx.spring_layout(graph_nx), with_labels=True, node_color='lightgray', node_size=300)
                #plt.show()
                #print(f'walks: {walks}')
                continue

            walks_subgraphs = split_graph_into_communities(graph_nx, walks)

            #print(walks)
            #plot_initial_graph_with_partitions(graph_nx, walks)
            #plot_community_subgraphs(walks_subgraphs)
            
            for walk_id, subgraph in walks_subgraphs.items():
                node_id_list = subgraph.nodes
                edges_list = subgraph.edges

                subgraph_df = day_farm_data.loc[day_farm_data['Plant_ID'].isin(node_id_list)]

                # extract appropriate masks
                train_masks_subgraph = [train_masks[i - 1] for i in node_id_list]
                valid_masks_subgraph = [valid_masks[i - 1] for i in node_id_list]
                test_masks_subgraph = [test_masks[i - 1] for i in node_id_list]

                # resetting node IDS
                new_plant_id = list(range(0, len(node_id_list)))
                new_plant_id_tuples = list(zip(node_id_list, new_plant_id))

                for old_id, new_id in new_plant_id_tuples:
                    subgraph_df.loc[subgraph_df["Plant_ID"] == old_id, "Plant_ID"] = new_id

                id_dict = dict(new_plant_id_tuples)
                updated_edges_list = [(id_dict.get(old_id_1, old_id_1), id_dict.get(old_id_2, old_id_2)) for old_id_1, old_id_2 in edges_list]

                # shared operations between augmented and non-augmented graphs
                if len(updated_edges_list) == 0:
                    continue # we skip graphs with no edges at all

                donor, receiver = zip(*updated_edges_list)
                donor_np = np.array(donor)
                receiver_np = np.array(receiver)

                # we need edges going both ways 
                from_plant1 = torch.tensor(donor_np, dtype = int)
                to_plant2 = torch.tensor(receiver_np, dtype = int)

                plant_edges = torch.concat((from_plant1, to_plant2)).reshape(-1, len(from_plant1)).long()

                #convert class values to tensor
                squamosa_rate_array = subgraph_df['cote_b_lactucae'].to_numpy()
                squamosa_rate_label = np.rint(squamosa_rate_array) # round up cote squamosa to nearest integer
                squamosa_rate_label = torch.tensor(squamosa_rate_label, dtype = int)

                #remove vars we dont want in the graph (the metadata)
                day_farm_data_clean = subgraph_df.drop(['FarmID', 'Plant_ID', 'Date', 'cote_b_lactucae'], axis=1)

                # create plant node as tensor
                plants_tensor = torch.tensor(np.array(day_farm_data_clean), dtype = float)

                graph = Data(x=plants_tensor, edge_index=plant_edges, y=squamosa_rate_label)

                transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
                graph = transform(graph)

                graph.train_mask = torch.Tensor(train_masks_subgraph)
                graph.val_mask = torch.Tensor(valid_masks_subgraph)
                graph.test_mask = torch.Tensor(test_masks_subgraph)

                graphs.append(graph)

                ###############################

                if farm_id not in unique_farms_idx:
                    unique_farms_idx[farm_id] = graph
                
                if aug_proof_concept_iter == 0:
                    aug_proof_concept[walk_id] = graph
    
    return (graphs, unique_farms_idx, aug_proof_concept)


