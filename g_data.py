import os
import numpy as np
import torch
from pymatgen.core.periodic_table import Element
import glob
import pandas as pd
from scipy import interpolate


def search_energy_range(xyz_filelist, prop_raw_dir):
    # min_energy = float("-inf")
    # max_energy = float("inf")
    # for xyz_file in xyz_filelist:
    #     if xyz_file.endswith('.xyz'):
    #         filename = xyz_file.split("/")[-1].split(".")[0]
    #         material_id = str(int(filename.split("_")[-1]))
    #         # print(material_id)
    #         for i in range(20):
    #             for j in range(10):
    #                 prop_file = f"{prop_raw_dir}/{material_id}/{material_id}_{i}_{j}.csv"
    #                 if os.path.exists(prop_file):
    #                     df_ = pd.read_csv(prop_file)
    #                     energy = df_["energy"].to_numpy()
    #                     min_energy_temp = min(energy)
    #                     max_energy_temp = max(energy)
    #                     min_energy = max(min_energy, min_energy_temp)
    #                     max_energy = min(max_energy, max_energy_temp) # 最も小さいmax_energyを取りたい。右に0を外挿したくない
    # min_energy = 284.1
    # max_energy = 322.3
    min_energy = 288
    max_energy = 310
    # min_energy = round(min_energy, 1)
    # max_energy = round(max_energy, 1)
    print(min_energy, max_energy)
    return min_energy, max_energy

def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    edges = [rows, cols]
    return edges

def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 原子数を取得
    num_atoms = int(lines[0].strip())

    # コメント（分子名など）を取得
    # molecule_name = lines[1].strip()

    # 各原子の情報を取得
    atoms = []
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        atom = {
            'symbol': parts[0],
            'x': float(parts[1]),
            'y': float(parts[2]),
            'z': float(parts[3])
        }
        atoms.append(atom)

    return num_atoms, atoms


def one_hot_encode(unique_elements, input_array):
    one_hot_dict = {element: i for i, element in enumerate(unique_elements)}
    one_hot_encoded = np.zeros((input_array.size, unique_elements.size))

    for idx, element in enumerate(input_array):
        one_hot_encoded[idx, one_hot_dict[element]] = 1

    return one_hot_encoded

def process_xyz_file(xyz_path, prop_path, carbon_sites, id, min_energy, max_energy):
    # print("=====process_xyz_file=====")
    num_atoms, atoms = read_xyz(xyz_path)
    node_features = np.array([Element(atom["symbol"]).Z for atom in atoms])

    unique_elements = np.array([1., 6., 7., 8., 9.])
    one_hot_encoded = one_hot_encode(unique_elements, node_features)

    carbon_index = np.zeros(num_atoms)
    for c_site_idx in carbon_sites:
        carbon_index[c_site_idx] = 1

    node_coords = np.array([np.array([atom["x"], atom["y"], atom["z"]]) for atom in atoms])

    edge_index = get_edges(num_atoms)
    edge_index = [torch.tensor(edge_index[0], dtype=torch.long), torch.tensor(edge_index[1], dtype=torch.long)]
    edge_attr = np.linalg.norm(node_coords[edge_index[0]] - node_coords[edge_index[1]], axis=1, keepdims=True)

    node_features = torch.tensor(node_features, dtype=torch.float32).unsqueeze(1)
    one_hot_encoded = torch.tensor(one_hot_encoded, dtype=torch.float32).unsqueeze(1)
    node_coords = torch.tensor(node_coords, dtype=torch.float32)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    carbon_index = torch.tensor(carbon_index)

    df_ = pd.read_csv(prop_path)
    energy = df_["energy"].to_numpy()
    if min(energy) > min_energy:
        energy_to_add = np.arange(min_energy, min(energy), 0.2)
        intensity_to_add = np.zeros_like(energy_to_add)
        df_to_add = pd.DataFrame({"energy": energy_to_add, "x": intensity_to_add, "y": intensity_to_add, "z": intensity_to_add})
        df_ = pd.concat([df_to_add, df_])

    df_ = df_[df_["energy"] >= min_energy]
    df_ = df_[df_["energy"] <= max_energy]

    output_size = 256
    new_energy = np.linspace(min_energy, max_energy, output_size)
    interpolator_x = interpolate.interp1d(df_["energy"], df_["x"], fill_value="extrapolate")
    new_intensity_x = interpolator_x(new_energy)
    sum_intensity_x = np.sum(new_intensity_x)
    new_intensity_x = new_intensity_x / sum_intensity_x * output_size

    interpolator_y = interpolate.interp1d(df_["energy"], df_["y"], fill_value="extrapolate")
    new_intensity_y = interpolator_y(new_energy)
    sum_intensity_y = np.sum(new_intensity_y)
    new_intensity_y = new_intensity_y / sum_intensity_y * output_size

    interpolator_z = interpolate.interp1d(df_["energy"], df_["z"], fill_value="extrapolate")
    new_intensity_z = interpolator_z(new_energy)
    sum_intensity_z = np.sum(new_intensity_z)
    new_intensity_z = new_intensity_z / sum_intensity_z * output_size

    # df = pd.DataFrame({"energy": new_energy, "x": new_intensity_x, "y": new_intensity_y, "z": new_intensity_z})
    # df.to_csv(f"tfn_ck_data/prop/{material_id}_{i}_{j}.csv")

    # new_intensity_x = new_intensity_x.reshape(1, len(new_energy))
    # new_intensity_y = new_intensity_y.reshape(1, len(new_energy))
    # new_intensity_z = new_intensity_z.reshape(1, len(new_energy))
    # print(new_intensity_z.shape) # → (1, 256)

    # target = np.concatenate([new_intensity_x, new_intensity_y, new_intensity_z], 0)
    target = np.vstack([new_intensity_x, new_intensity_y, new_intensity_z])
    target = torch.tensor(target, dtype=torch.float32)
    # print(target.size()) # → (256*3, )

    return (node_features, one_hot_encoded, node_coords, edge_index, edge_attr, carbon_index), target, id


def main(data_size=100):
    # xyzファイルを処理して保存
    xyz_filelist = glob.glob("/Users/atsushitakigawa/Downloads/dsgdb9nsd.xyz/*.xyz")
    prop_raw_dir = "/Users/atsushitakigawa/Downloads/csv_spectra_0.5eV"
    prop_dir = "tfn_data/prop"
    processed_data = []

    min_energy, max_energy = search_energy_range(xyz_filelist, prop_raw_dir)

    for xyz_file in xyz_filelist:
        if xyz_file.endswith('.xyz'):
            filename = xyz_file.split("/")[-1].split(".")[0]
            material_id = int(filename.split("_")[-1])
            # print(filename)
            for i in range(20):
                for j in range(10):
                    prop_file = f"{prop_raw_dir}/{material_id}/{material_id}_{i}_{j}.csv"
                    structure_file = f"{prop_raw_dir}/{material_id}/{material_id}_structure.csv"
                    if os.path.exists(prop_file):
                        df_str = pd.read_csv(structure_file)
                        df_str = df_str[df_str['specie'] == "C"]
                        df_str = df_str[df_str['representative'] == i]
                        carbon_sites = df_str["site"].tolist()
                        id = f"{material_id}_{i}_{j}"

                        try:
                            data = process_xyz_file(xyz_file, prop_file, carbon_sites, id, min_energy, max_energy)
                            processed_data.append(data)
                        except:
                            print(f"skip {material_id}")
        if len(processed_data) > data_size:
            break

    # データを保存
    torch.save(processed_data, 'dataset.pt')
    print("データ数: ", len(processed_data))



if __name__ == '__main__':
    data_size = int(input())
    main(data_size)

