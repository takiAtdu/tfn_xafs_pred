import pandas as pd
import torch.optim as optim
from torch.nn import MSELoss
import torch
import numpy as np
import shutil
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from model import TFN  # assuming your TFN class is in tfn.py
from torchviz import make_dot
from torchsummary import summary

model_best = "model_best.pth.tar"
model_checkpoint = "checkpoint.pth.tar"

class TFNDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    batch_node_features, batch_one_hot_features, batch_node_coords, batch_edge_index, batch_edge_attr, batch_carbon_index = [], [], [], [], [], []
    batch_target, batch_ids = [], []
    batch_row, batch_col = [], []
    crystal_node_idx = []
    base_idx = 0
    for i, ((node_features, one_hot_encoded, node_coords, edge_index, edge_attr, carbon_index), target, id) in enumerate(batch):
        n_i = node_features.shape[0]
        batch_node_features.append(node_features)
        batch_one_hot_features.append(one_hot_encoded)
        batch_node_coords.append(node_coords)
        batch_row.append(edge_index[0]+base_idx)
        batch_col.append(edge_index[1]+base_idx)
        batch_edge_attr.append(edge_attr)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_node_idx.append(new_idx)
        batch_target.append(target)
        batch_ids.append(id)
        base_idx += n_i
        batch_carbon_index.append(carbon_index)

    batch_edge_index.append(torch.cat(batch_row, dim=0))
    batch_edge_index.append(torch.cat(batch_col, dim=0))
    return (torch.squeeze(torch.cat(batch_one_hot_features, dim=0), dim=1),
            torch.cat(batch_node_coords, dim=0),
            batch_edge_index,
            torch.cat(batch_edge_attr, dim=0),
            crystal_node_idx,
            batch_carbon_index), torch.stack(batch_target, dim=0), batch_ids

def get_train_val_test_loader(dataset, collate_fn=default_collate, batch_size=64,
                              train_ratio=None, val_ratio=0.1, test_ratio=0.1,
                              return_test=False, num_workers=1, pin_memory=False, **kwargs):
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
              f'test_ratio = {train_ratio} as training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1

    indices = list(range(total_size))
    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

def mae(pred, target):
    return torch.sum(torch.sum(torch.mean(torch.abs(target - pred), 2), 1))

def mse(pred, target):
    return torch.sum(torch.sum(torch.mean(torch.abs(target - pred)**2, 2), 1))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count / 3  # x, y, zがあるので3で割る

def save_checkpoint(state, is_best, filename=f'tfn/{model_checkpoint}'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'tfn/{model_best}')

def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    for index, (inputs, targets, ids) in enumerate(train_loader):
        inputs_one_hot, inputs_coords, _, _, crystal_atom_idx, carbon_idx = inputs
        # モデルのフォワードパス
        preds = model(inputs_coords, inputs_one_hot, crystal_atom_idx, carbon_idx)

        # 損失の計算
        # print(preds.size(), targets.size())
        # print(preds[0][0])
        loss = criterion(preds, targets)
        losses.update(loss.item())

        # バックプロパゲーションとパラメータ更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg

def validate(val_loader, model, is_test=False):
    model.eval()
    mae_errors = AverageMeter()
    mse_errors = AverageMeter()

    if is_test:
        test_preds_list = []
        test_targets_list = []
        test_ids_list = []

    for inputs, targets, ids in val_loader:
        inputs_one_hot, inputs_coords, _, _, crystal_atom_idx, carbon_idx = inputs
        with torch.no_grad():
            preds = model(inputs_coords, inputs_one_hot, crystal_atom_idx, carbon_idx)

            # mae_error = mae(preds, targets)
            # mae_errors.update(mae_error, targets.size(0))

            mse_error = mse(preds, targets)
            mse_errors.update(mse_error, targets.size(0))

            if is_test:
                for i in range(len(ids)):
                    test_preds_list.append(preds[i])
                    test_targets_list.append(targets[i])
                    test_ids_list.append(ids[i])

    if not is_test:
        star_label = '*'
    else:
        star_label = '**'
    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label, mae_errors=mae_errors))
    print(' {star} MSE {mse_errors.avg}'.format(star=star_label, mse_errors=mse_errors))

    if is_test:
        import csv
        with open("tfn/test_results.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "target_path", "pred_path"])
            for id, target, pred in zip(test_ids_list, test_targets_list, test_preds_list):
                for i in range(3):
                    if i == 0:
                        id_ = f"{id}_x"
                    elif i == 1:
                        id_ = f"{id}_y"
                    elif i == 2:
                        id_ = f"{id}_z"
                    target_ = np.array(target[i])
                    target_path = f"tfn/test_targets/{id_}.npy"
                    np.save(target_path, target_)
                    pred_ = np.array(pred[i])
                    pred_path = f"tfn/test_preds/{id_}.npy"
                    np.save(pred_path, pred_)

                    writer.writerow([id_, target_path, pred_path])

    return mse_errors.avg

def show_result(min_energy, max_energy, model):
    df = pd.read_csv("tfn/test_results.csv")
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    energy = np.linspace(min_energy, max_energy, 256)
    for i in range(10):
        _row = i // 10 * 2
        col = i % 10
        target_row = _row
        pred_row = 1 + _row

        df_ = df.iloc[i]
        id_ = df_["id"]
        target_path = df_["target_path"]
        pred_path = df_["pred_path"]
        x = energy
        target_y = np.load(target_path)
        pred_y = np.load(pred_path)

        ax_tar = axes[target_row, col]
        ax_tar.plot(x, target_y)
        ax_tar.set_title(f"{id_} target")

        ax_pre = axes[pred_row, col]
        ax_pre.plot(x, pred_y)
        ax_pre.set_title(f"{id_} pred")
    plt.tight_layout()
    plt.savefig("tfn/test_results.png")

    # make_dot(pred_y, params=dict(model.named_parameters()))

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path=f'tfn/{model_checkpoint}'):
        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            # self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            # self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する


def main():
    # データをロード
    print("Loading data...")
    batch_size = 32
    processed_data = torch.load('dataset.pt')
    dataset = TFNDataset(processed_data)
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        return_test=True
    )

    # TFNモデルの初期化
    num_atom_types = 5  # 適切な数に設定してください
    model = TFN(num_atom_types=num_atom_types, output_dim=256)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = MSELoss()
    earlystopping = EarlyStopping(patience=5, verbose=True)

    # 訓練ループ
    num_epochs = 3
    best_mse_error = 1e10
    for epoch in range(num_epochs):
        model.train()
        loss = train(train_loader, model, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}")

        mse_error = validate(val_loader, model)
        is_best = mse_error < best_mse_error
        best_mse_error = min(mse_error, best_mse_error)
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_mse_error": best_mse_error,
            "optimizer": optimizer.state_dict(),
        }, is_best)

        earlystopping(loss, model)
        if earlystopping.early_stop:
            print("Early stopping")
            break

    print("=====Evaluate Model on Test Set=====")
    best_checkpoint = torch.load(f"tfn/{model_best}")
    model.load_state_dict(best_checkpoint["state_dict"])
    _ = validate(test_loader, model, is_test=True)

    min_energy = 288
    max_energy = 310
    show_result(min_energy, max_energy, model)
    # torch.onnx.export(model, test_loader, "tfn/model.onnx", verbose=True)

    print("batch_size: ", batch_size)
    print("num_epochs: ", num_epochs)


if __name__ == '__main__':
    main()
