from data.dataset import DynamicPush
import collections
import yaml
import torch
from model.model import ForwardModel
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
import copy

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('-config_file', default='config/train.yaml', help='Config Path.')
args = parser.parse_args()

def parse_config(config):

    """
    Parse iGibson config file / object
    """
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    if isinstance(config, collectionsAbc.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        raise IOError(
            "config path {} does not exist. Please either pass in a dict or a string that represents the file path to the config yaml.".format(
                config
            )
        )
    with open(config, "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data

def main(args):
    config = parse_config(args.config_file)
    print(config["modality"])
    dataset = DynamicPush(config["train_path"], modality=config["modality"])
    testset = DynamicPush(config["test_path"], train= False, modality=config["modality"])
    trainLoader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=True, num_workers=12
        )
    testLoader = torch.utils.data.DataLoader(
        testset, batch_size=8, shuffle=True, num_workers=12
    )
    model = ForwardModel(config["in_ch"])
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    mse_loss = torch.nn.MSELoss()
    
    best_loss = 100  
    for e in tqdm(range(config["num_epoch"])):
        running_loss = 0.0
        model.train()
        
        for i, data in enumerate(trainLoader):
            for d in data:
                d.cuda()
            x_1, x_2, x_3, act_in, pose_4_init, pose_4_final = data
            optimizer.zero_grad()
            pred_final = model(x_1, x_2, x_3, pose_4_init, act_in)
            loss = mse_loss(pred_final, pose_4_final)
            loss.backward()
            optimizer.step()
            running_loss = (running_loss * i + loss.item()) / (i + 1)
            print("running loss", running_loss)
        model.eval()
        eval_loss = 0
        x_pred = []
        y_pred = []
        x_final = []
        y_final = []
        for i, data in enumerate(testLoader):
            for d in data:
                d.cuda()
            x_1, x_2, x_3, act_in, pose_4_init, pose_4_final = data
            with torch.no_grad():
                pred_final = model(x_1, x_2, x_3, pose_4_init, act_in)
                l = mse_loss(pred_final, pose_4_final)
                eval_loss = (eval_loss * i + l.item()) / (i + 1)
            print("Eval loss", eval_loss)
            x_pred.extend(pred_final[:, 0].cpu().numpy().tolist())
            y_pred.extend(pred_final[:, 1].cpu().numpy().tolist())
            x_final.extend(pose_4_final[:, 0].cpu().numpy().tolist())
            y_final.extend(pose_4_final[:, 1].cpu().numpy().tolist())
    
            for k in range(len(x_pred)):
                plt.plot((x_pred[k], x_final[k]), (y_pred[k], y_final[k]), 'ro-')
                plt.savefig(os.path.join(config["plot_path"], 'results_' + str(i) + '.png'))
            plt.close()
            x_pred = []
            y_pred = []
            x_final = []
            y_final = []
        if eval_loss <= best_loss:
            print("update best model at ", e, "th epoch")
            best_model = copy.deepcopy(model.state_dict())
            best_loss = eval_loss
            print("best loss is", best_loss)
        
        if e%2==0:
            modelDir = config["model_dir"]
            os.makedirs(modelDir, exist_ok=True)
            fnModel = os.path.join(modelDir, config["modality"] + str(e) + ".pth")
            torch.save(model.state_dict(), fnModel)
    
    modelDir = config["model_dir"]
    os.makedirs(modelDir, exist_ok=True)
    fnModel = os.path.join(modelDir, "final" + config["modality"] + ".pth")
    torch.save(model.state_dict(), fnModel)
    print("saving best model!")
    bestModel = os.path.join(modelDir, "best" + config["modality"] + ".pth")
    torch.save(best_model.state_dict(), bestModel)

def test(args):
    config = parse_config(args.config_file)
    print(config["modality"])
    config["test_path"] = "/viscam/projects/objectfolder_benchmark/Robot_related/Taxim_obj/dynamic_pushing/config/debug.txt"
    testset = DynamicPush(config["test_path"], train= False, modality=config["modality"])

    testLoader = torch.utils.data.DataLoader(
        testset, batch_size=8, shuffle=True, num_workers=12
    )
    model = ForwardModel(config["in_ch"])
    model.load_state_dict(torch.load("/viscam/projects/objectfolder_benchmark/Robot_related/Taxim_obj/dynamic_pushing/model/0130/vt/vt100.pth"))
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    mse_loss = torch.nn.MSELoss()        
    model.eval()
    eval_loss = 0
    x_pred = []
    y_pred = []
    x_final = []
    y_final = []
    for i, data in enumerate(testLoader):
        for d in data:
            d.cuda()
        x_1, x_2, x_3, act_in, pose_4_init, pose_4_final = data
        with torch.no_grad():
            pred_final = model(x_1, x_2, x_3, pose_4_init, act_in)
            print(pose_4_init)
            print(pose_4_final)
            print(pred_final)
            l = mse_loss(pred_final, pose_4_final)
            eval_loss = (eval_loss * i + l.item()) / (i + 1)
        print("Eval loss", eval_loss)
        x_pred.extend(pred_final[:, 0].cpu().numpy().tolist())
        y_pred.extend(pred_final[:, 1].cpu().numpy().tolist())
        x_final.extend(pose_4_final[:, 0].cpu().numpy().tolist())
        y_final.extend(pose_4_final[:, 1].cpu().numpy().tolist())
        x_pred = []
        y_pred = []
        x_final = []
        y_final = []

if __name__ == "__main__":
    main(args)