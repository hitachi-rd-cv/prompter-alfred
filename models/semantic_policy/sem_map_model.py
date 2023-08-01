import os
import torch
import einops
import random

import numpy as np
import pandas as pd
import torch.nn as nn
import plotly.express as px

from torchvision import transforms


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L10
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UNet(nn.Module):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1, num_sem_categories=16):  # input shape is (240, 240)

        super(UNet, self).__init__()

        #out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)
        out_size = int(15 * 15)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories+4, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.deconv_main = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=3),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(1, 1, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(1, 1, 3, stride=1, padding=2),
            nn.ReLU()
        )

        # outsize is 15^2 (7208 total)
        self.linear1 = nn.Linear(out_size * 32 + 256, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        #self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(73, 256)  # 73 object categories
        self.softmax = nn.Softmax(dim=1)
        self.flatten = Flatten()
        self.train()

    def forward(self, inputs, goal_cats):
        x = self.main(inputs)
        #print("x shape is ", x.shape)
        #orientation_emb = self.orientation_emb(extras[:,0])
        goal_emb = self.goal_emb(goal_cats).view(-1, 256)  # goal name
        #print("goal emb shape is ", goal_emb.shape)

        x = torch.cat((x, goal_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))

        x = x.view(-1, 1, 16, 16)
        x = self.deconv_main(x)  # WIll get Nx1x8x8
        x = self.flatten(x)
        #x = self.softmax(x)
        return x


class UNetMulti(nn.Module):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1, num_sem_categories=16):  # input shape is (240, 240)

        super(UNetMulti, self).__init__()

        #out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)
        out_size = int(15 * 15)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories+4, 32, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 73, 3, stride=1, padding=1),
            #nn.Conv2d(64, 32, 3, stride=1, padding=1),
            # nn.ReLU(),
            # Flatten()
        )

        self.softmax = nn.Softmax(dim=1)
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.train()

    def plotSample(self, pl, fname, plot_type=None, names=None, wrap_sz=None, img_sz=(1280, 720), zmax=1):
        dname = os.path.split(fname)[0]
        os.makedirs(dname, exist_ok=True)

        fig_recep = px.imshow(
            pl, facet_col=0, facet_col_wrap=wrap_sz, zmin=0, zmax=zmax)

        if os.path.splitext(fname)[-1] == ".html":
            config = dict({'scrollZoom': True})
            fig_recep.write_html(fname, config=config)
        else:
            fig_recep.write_image(fname, width=img_sz[0], height=img_sz[1])

    def forward(self, inputs, target_name, out_dname=None, steps_taken=None, temperature=1):
        x = self.main(inputs)
        #x = self.flatten(x)
        #x = self.softmax(x)

        return x


class UNetDot(nn.Module):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1, num_sem_categories=16):  # input shape is (240, 240)

        super(UNet, self).__init__()

        #out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)
        out_size = int(15 * 15)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories+4, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, stride=1, padding=1),
            nn.ReLU()
            #nn.Conv2d(64, 32, 3, stride=1, padding=1),
            # nn.ReLU(),
            # Flatten()
        )

        self.deconv_main = nn.Sequential(
            nn.Conv2d(256, 128, 1, stride=1, padding=1),
            nn.ReLU(),
            # nn.AvgPool2d(2),
            nn.Conv2d(128, 64, 1, stride=1, padding=1),
            nn.ReLU(),
            # nn.AvgPool2d(2),
            nn.Conv2d(64, 32, 1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1, stride=1, padding=1),
            nn.ReLU(),
        )

        # self.linear1 = nn.Linear(out_size * 32 + 8, hidden_size) #outsize is 15^2 (7208 total)
        #self.linear2 = nn.Linear(hidden_size, 256)
        #self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(73, 256)  # 73 object categories
        self.goal_lin = nn.Linear(256, 128)
        self.goal_lin2 = nn.Linear(128, 128)
        self.softmax = nn.Softmax(dim=1)
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.train()

    def forward(self, inputs, goal_cats):
        x = self.main(inputs)
        #print("x shape is ", x.shape)
        #orientation_emb = self.orientation_emb(extras[:,0])
        goal_emb = self.goal_emb(goal_cats).view(-1, 256)  # goal name
        goal_emb = self.goal_lin(goal_emb)
        goal_emb = self.relu(goal_emb)
        goal_emb = self.goal_lin2(goal_emb)
        goal_emb = self.relu(goal_emb)

        # Tile goal_emb

        #print("goal emb shape is ", goal_emb.shape)

        x = torch.cat((x, goal_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))

        x = x.view(-1, 1, 16, 16)
        x = self.deconv_main(x)  # WIll get Nx1x8x8
        x = self.flatten(x)
        #x = self.softmax(x)
        return


class MLM(nn.Module):

    def __init__(self, input_shape, output_shape, score_fname, small2indx=None, large2indx=None, options=list()):
        super(MLM, self).__init__()

        # input shape is (240, 240)
        self.input_h, self.input_w = input_shape
        self.output_h, self.output_w = output_shape
        self.options = options

        self.scores = pd.read_csv(score_fname)
        if small2indx is None:
            self.small2indx = {'AlarmClock': 0, 'Apple': 1, 'AppleSliced': 2, 'BaseballBat': 3, 'BasketBall': 4, 'Book': 5, 'Bowl': 6, 'Box': 7, 'Bread': 8, 'BreadSliced': 9, 'ButterKnife': 10, 'CD': 11, 'Candle': 12, 'CellPhone': 13, 'Cloth': 14, 'CreditCard': 15, 'Cup': 16, 'DeskLamp': 17, 'DishSponge': 18, 'Egg': 19, 'Faucet': 20, 'FloorLamp': 21, 'Fork': 22, 'Glassbottle': 23, 'HandTowel': 24, 'HousePlant': 25, 'Kettle': 26, 'KeyChain': 27, 'Knife': 28, 'Ladle': 29, 'Laptop': 30, 'LaundryHamperLid': 31, 'Lettuce': 32, 'LettuceSliced': 33, 'LightSwitch': 34, 'Mug': 35, 'Newspaper': 36,
                               'Pan': 37, 'PaperTowel': 38, 'PaperTowelRoll': 39, 'Pen': 40, 'Pencil': 41, 'PepperShaker': 42, 'Pillow': 43, 'Plate': 44, 'Plunger': 45, 'Pot': 46, 'Potato': 47, 'PotatoSliced': 48, 'RemoteControl': 49, 'SaltShaker': 50, 'ScrubBrush': 51, 'ShowerDoor': 52, 'SoapBar': 53, 'SoapBottle': 54, 'Spatula': 55, 'Spoon': 56, 'SprayBottle': 57, 'Statue': 58, 'StoveKnob': 59, 'TeddyBear': 60, 'Television': 61, 'TennisRacket': 62, 'TissueBox': 63, 'ToiletPaper': 64, 'ToiletPaperHanger':65, 'ToiletPaperRoll': 66, 'Tomato': 67, 'TomatoSliced': 68, 'Towel': 69, 'Vase': 70, 'Watch': 71, 'WateringCan': 72, 'WineBottle': 73}
            self.large2indx = {'ArmChair': 0, 'BathtubBasin': 1, 'Bed': 2, 'Cabinet': 3, 'Cart': 4, 'CoffeeMachine': 5, 'CoffeeTable': 6, 'CounterTop': 7, 'Desk': 8, 'DiningTable': 9, 'Drawer': 10,
                               'Dresser': 11, 'Fridge': 12, 'GarbageCan': 13, 'Microwave': 14, 'Ottoman': 15, 'Safe': 16, 'Shelf': 17, 'SideTable': 18, 'SinkBasin': 19, 'Sofa': 20, 'StoveBurner': 21, 'TVStand': 22, 'Toilet': 23}
        else:
            # self.small2indx = {' '.join(re.findall("[A-Z][a-z]+", k)).lower():v
            #                    for k, v in small2indx.items()}
            # self.large2indx = {' '.join(re.findall("[A-Z][a-z]+", k)).lower():v
            #                    for k, v in large2indx.items()}
            self.small2indx = small2indx
            self.large2indx = large2indx

        self.all2indx = {**self.small2indx, **{k: v+len(self.small2indx) for k, v in self.large2indx.items()}}

        self.indx2small = {v: k for k, v in self.small2indx.items()}
        self.indx2large = {v: k for k, v in self.large2indx.items()}

        self.scores["recep_indx"] = self.scores["receps"].map(self.large2indx)
        self.scores["object_indx"] = self.scores["object"].map(self.all2indx)
        self.scores = self.scores[self.scores.notnull().all(1)]
        self.scores["recep_indx"] = self.scores["recep_indx"].astype(int)
        self.scores["object_indx"] = self.scores["object_indx"].astype(int)
        self.score_mat = self.scores.pivot(
            index="recep_indx", columns="object_indx", values="scores")
        self.score_mat = torch.tensor(
            self.score_mat.values.astype(np.float32)).to(device="cuda")

        self.softmax_fn = torch.nn.Softmax(dim=0)
        # self.score_mat = softmax_fn(self.score_mat)
        # self.score_mat = torch.exp(self.score_mat)

    def split4Map(self, data):
        return data[:, 0, :, :], data[:, 1, :, :], data[:, 2, :, :], data[:, 3, :, :], data[:, 4:, :, :]

    def plotSample(self, pl, fname, plot_type=None, names=None, wrap_sz=None, img_sz=(1280, 720), zmax=1):
        dname = os.path.split(fname)[0]
        os.makedirs(dname, exist_ok=True)

        if plot_type == "recep":
            names = [self.indx2large[indx]
                     for indx in range(len(self.large2indx))]
            wrap_sz = 6
        elif plot_type == "object":
            names = [self.indx2small[indx]
                     for indx in range(len(self.small2indx))]
            wrap_sz = 15

        # import cv2
        # for img, name in zip(pl, names):
            # cv2.imwrite(os.path.join(os.path.dirname(fname), f"{name}.png"), img.to('cpu').detach().numpy().copy() * 255)

        fig_recep = px.imshow(
            pl, facet_col=0, facet_col_wrap=wrap_sz, zmin=0, zmax=zmax)
        # fig_recep = px.imshow(pl, facet_col=0, facet_col_wrap=wrap_sz)
        fig_recep.for_each_annotation(lambda a: a.update(
            text=names[int(a.text.split("=")[-1])]))

        if os.path.splitext(fname)[-1] == ".html":
            config = dict({'scrollZoom': True})
            fig_recep.write_html(fname, config=config)
        else:
            fig_recep.write_image(fname, width=img_sz[0], height=img_sz[1])

    def forward(self, inputs, target_name, out_dname=None, steps_taken=None, temperature=1):
        # NOTE: lower the temperature, sharper the distribution will be

        # @inputs is (batch)x(4+24 receptacle categories)x(@input_shape)x(@input_shape)
        # Ex. 1x28x240x240
        # First 4 elements of 28 are:
        # 1. Obstacle Map
        # 2. Exploread Area
        # 3. Current Agent Location
        # 4. Past Agent Locations

        roi_indx = self.all2indx[target_name]
        score_roi = self.score_mat[:, roi_indx:roi_indx+1]

        # larger this value is, faster the softmax output will approach to a uniform distribution
        temperature_ratio = 0.5
        for opt_name in self.options:
            if "tempRatio" in opt_name:
                temperature_ratio = float(opt_name.replace("tempRatio", ""))

        temperature = 1 + (temperature - 1) * temperature_ratio
        score_mat_n = self.softmax_fn(score_roi / temperature)

        obstacles, explored, curr_loc, past_loc, large_pred = self.split4Map(
            inputs)

        # create score map
        if (self.input_h, self.input_w) == (self.output_h, self.output_w):
            large_pred_redux = large_pred.clone()
        else:
            large_pred_redux = einops.reduce(
                large_pred, "b r (s1 h) (s2 w) -> b r s1 s2", "max", s1=self.output_h, s2=self.output_w)

        # b r h w -> b h w r
        large_tmp = large_pred_redux.permute((0, 2, 3, 1))

        # normalize spatially per receptacle class, to reduce the effect of receptacle's physical size
        epsilon = torch.finfo(torch.float32).tiny
        if "spatial_norm" in self.options:
            large_tmp /= (large_tmp.sum(keepdim=True, dim=[0, 1, 2]) + epsilon)

        # combine large object predictions with object-object relationship score by...
        if "aggregate_sum" in self.options:
            # sum (i.e. matrix multiplication)
            # matrix multiplication is equivalent to: torch.sum(large_tmp.unsqueeze(-1) * score_mat_n, dim=-2)
            prob_scores = torch.matmul(large_tmp, score_mat_n)
        elif "aggregate_max" in self.options:
            # max
            prob_scores = torch.max(
                large_tmp.unsqueeze(-1) * score_mat_n, dim=-2)[0]
        elif "aggregate_sample" in self.options:
            recep_exist_indices = torch.nonzero(large_tmp.sum(dim=[0, 1, 2]))[:, 0]

            if len(recep_exist_indices) == 0:
                # an edge case in which no receptacles are observed yet
                select_recep_index = random.randint(0, len(score_mat_n) - 1)
            else:
                one_hot = torch.zeros_like(score_mat_n)
                one_hot[recep_exist_indices] = 1
                score_exist = one_hot * score_mat_n
                prob_exist = score_exist / score_exist.sum()

                # sample a receptacle index
                select_recep_index = torch.multinomial(prob_exist.squeeze(), 1)
            prob_scores = large_tmp[:, :, :, select_recep_index:select_recep_index+1]

        # b h w o -> b o h w
        prob_scores = prob_scores.permute((0, 3, 1, 2))

        # remove explored locations from the score
        exp_map = None
        if "explored" in self.options:
            exp_map = explored
        elif "past_loc" in self.options:
            exp_map = past_loc

        if exp_map is None:
            scores = prob_scores
        else:
            exp_map_s = einops.reduce(
                exp_map, "b (s1 h) (s2 w) -> b s1 s2", "mean", s1=self.output_h, s2=self.output_w)
            scores = prob_scores * (1 - exp_map_s)

        # scale the scores (should sum to 1 for h x w dimenstions)
        scores += epsilon
        scores_n = scores / torch.sum(scores, dim=[2, 3], keepdim=True)
        # scores_n = scores

        # the output should be 1x73x(self.output_h)x(self.output_w)
        # scores_redux = einops.reduce(scores_n, "b r (s1 h) (s2 w) -> b r s1 s2", "sum", s1=self.output_h, s2=self.output_w)
        scores_redux = scores_n

        # visualize the result
        if out_dname is not None:
            self.plotSample(
                large_pred[0].cpu(), os.path.join(
                    out_dname, "pred_receptacles", f"{steps_taken}_receptacles.html"),
                img_sz=(1920, 1080), plot_type="recep", zmax=1)

            # self.plotSample(
            #     large_tmp[0].permute((2, 0, 1)).cpu(), os.path.join(
            #         out_dname, "pred_receptacles", f"{steps_taken}_receptacles_norm.html"),
            #     img_sz=(1920, 1080), plot_type="recep", zmax=0.01)

            # plot the extraneous information included in map_learned
            self.plotSample(
                inputs[0, :4, :, :].cpu(), os.path.join(
                    out_dname, "extra_info", f"{steps_taken}.png"),
                plot_type=None, names=["obstacles", "explored", "current location", "past location"], wrap_sz=4)

        # import cv2
        # target_prob_name = os.path.join(out_dname, "pred_receptacles", f"{steps_taken}_{target_name}.png")
        # cv2.imwrite(target_prob_name, (scores_redux / scores_redux.max() * 255)[0, 0].to('cpu').detach().numpy().copy())

        return scores_redux
