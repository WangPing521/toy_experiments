import torch
from deepclustering2.configparser import ConfigManger
from deepclustering2.utils import set_environment, write_yaml
from torch.utils.tensorboard import SummaryWriter
from architectures.enet import Enet
from dataset_loader.symetry_loader import SymetryData, DataloaderIter
from torch.utils.data import DataLoader
from loss_functions.consAwareVat import consVATLoss
from loss_functions.crossentropy import SimplexCrossEntropyLoss
from tool.independent_functions import fix_all_seed, class2one_hot, simplex, average_list, plot_joint_matrix
from tqdm import tqdm

device = 'cuda'
config = ConfigManger("config/config_toyseg.yaml").config
fix_all_seed(config['seed'])
dir = f'symetry_run/{config["save_dir"]}'
writer = SummaryWriter(f'{dir}/tensorboard')

config0 = config.copy()
config0.pop("Config", None)
write_yaml(config0, save_dir=dir, save_name="config.yaml")
set_environment(config.get("Environment"))

net1 = Enet(input_dim=1, num_classes=config['num_classes'])
optim1 = torch.optim.Adam(net1.parameters(), config['lr'])
crossentropy = SimplexCrossEntropyLoss()
vatloss = consVATLoss(consweight=config['weights']['cons_weight'], mode=config['train_mode'])

#todo: label and unlebel
lab_set = SymetryData(img_dir='dataset/symetry_images/train', ratio=config['ratio'], indicator='lab_train')
unlab_set = SymetryData(img_dir='dataset/symetry_images/train', ratio=config['ratio'], indicator='unlab_train')
val_set = SymetryData(img_dir='dataset/symetry_images/val')


lab_dataloader = DataloaderIter(DataLoader(lab_set, batch_size=4, shuffle=True))
unlab_dataloader = DataloaderIter(DataLoader(unlab_set, batch_size=4, shuffle=True))
test_dataloader = DataLoader(val_set, batch_size=4, shuffle=True)

max_epoch = 100
# # sup + cons (without vat)
if __name__ == '__main__':

    for cur_epoch in range(max_epoch):
        net1.to(device)
        if cur_epoch < config['weight_epoch']:
            weight1, weight2 = 0, 0
        else:
            weight1 = config['weights']['weight_adv']
            weight2 = config['weights']['weight_cons']

        suplosslist, ldslist, conslist = [], [], []
        batch_indicator = tqdm(range(200))
        batch_indicator.set_description(f"Training Epoch {cur_epoch:03d}")
        for batch_id, lab_data, unlab_data in zip(batch_indicator, lab_dataloader, unlab_dataloader):
            lds, cons = 0, 0
            lab_img, lab_target, lab_filename = (
                lab_data[0].to(device),
                lab_data[1].to(device),
                lab_data[2]
            )

            onehot_target = class2one_hot(lab_target.squeeze(1), C=2)

            unlab_img, unlab_target, unlab_filename = (
                unlab_data[0].to(device),
                unlab_data[1].to(device),
                unlab_data[2]
            )

            pred_lab = net1(lab_img).softmax(1)
            suploss = crossentropy(pred_lab, onehot_target)

            # unlab
            with torch.no_grad():
                unlab_pred = torch.softmax(net1(unlab_img), dim=1)
            assert simplex(unlab_pred)
            if cur_epoch > 15:
                all_shapes_list, shape_error_list, lds, cons = vatloss(net1, unlab_img)

                #todo:visulization
                if batch_id == 0:
                    joint1 = torch.cat([unlab_target[-1][0].unsqueeze(0), unlab_pred.max(1)[1][-1].unsqueeze(0)], 0)
                    joint1 = torch.cat([joint1, all_shapes_list[-1][0].unsqueeze(0)], 0)
                    joint1 = torch.cat([joint1, shape_error_list[-1][0].unsqueeze(0)], 0)
                    joint1 = joint1.unsqueeze(0)
                    # gt, seg, allshape, +error_symmetry
                    sample1 = plot_joint_matrix(unlab_filename[-1], joint1)
                    writer.add_figure(tag=f"symmetry_vis", figure=sample1, global_step=cur_epoch, close=True)

            optim1.zero_grad()
            loss = suploss + weight1 * lds + weight2 * cons
            suplosslist.append(suploss)
            ldslist.append(lds)
            conslist.append(cons)

            loss.backward()
            optim1.step()

        supl = average_list(suplosslist)
        lml = average_list(ldslist)
        consl = average_list(conslist)

        writer.add_scalar('loss/sup', supl.item(), cur_epoch)
        try:
            writer.add_scalar('loss/lds', lml.item(), cur_epoch)
        except:
            writer.add_scalar('loss/lds', lml, cur_epoch)
        try:
            writer.add_scalar('loss/cons', consl.item(), cur_epoch)
        except:
            writer.add_scalar('loss/cons', consl, cur_epoch)

        print(f"Training Epoch {cur_epoch}: suploss: {supl} lds: {lml} cons: {consl}")

#       # val
#         for val_data in enumerate(test_dataloader):
#             val_img, val_target, val_filename = (
#                 val_data[0][0],
#                 val_data[0][1],
#                 val_data[1]
#             )




