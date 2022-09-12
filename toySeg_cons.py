import torch
from deepclustering2.configparser import ConfigManger
from deepclustering2.utils import set_environment, write_yaml
from torch.utils.tensorboard import SummaryWriter
from architectures.enet import Enet
from dataset_loader.symetry_loader import SymetryData, DataloaderIter
from torch.utils.data import DataLoader
from loss_functions.consAwareVat import consVATLoss, Local_cons
from loss_functions.constraint_loss import symmetry_error
from loss_functions.crossentropy import SimplexCrossEntropyLoss
from tool.independent_functions import fix_all_seed, class2one_hot, simplex, average_list, plot_joint_matrix
from tqdm import tqdm

from tool.save_images import save_toyseg

config = ConfigManger("config/config_toyseg.yaml").config
fix_all_seed(config['seed'])
dir = f'symetry_run/{config["save_dir"]}'
writer = SummaryWriter(f'{dir}/tensorboard')

config0 = config.copy()
config0.pop("Config", None)
write_yaml(config0, save_dir=dir, save_name="config.yaml")
set_environment(config.get("Environment"))
device = config['device']

net1 = Enet(input_dim=1, num_classes=config['num_classes'])
optim1 = torch.optim.Adam(net1.parameters(), config['lr'])

crossentropy = SimplexCrossEntropyLoss()
vatloss = consVATLoss(consweight=config['weights']['cons_weight'], mode=config['train_mode'])
symmetry_cons = Local_cons()

lab_set = SymetryData(img_dir='dataset/symetry_images/train', ratio=config['ratio'], indicator='lab_train')
unlab_set = SymetryData(img_dir='dataset/symetry_images/train', ratio=config['ratio'], indicator='unlab_train')
val_set = SymetryData(img_dir='dataset/symetry_images/val')

lab_dataloader = DataloaderIter(DataLoader(lab_set, batch_size=4, shuffle=True))
unlab_dataloader = DataloaderIter(DataLoader(unlab_set, batch_size=4, shuffle=True))
test_dataloader = DataLoader(val_set, batch_size=1, shuffle=True)

max_epoch = 100
if __name__ == '__main__':

    for cur_epoch in range(0, max_epoch):
        net1.to(device)
        if cur_epoch < config['weight_epoch']:
            weight1, weight2 = 0, 0
        else:
            weight1 = config['weights']['weight_adv']
            weight2 = config['weights']['weight_cons']

        suplosslist, ldslist, conslist = [], [], []
        batch_indicator = tqdm(range(50))
        batch_indicator.set_description(f"Training Epoch {cur_epoch:03d}")
        train_dice_list, train_error_list, train_ublabdice_list, train_unlaberror_list = [], [], [], []
        for batch_id, lab_data, unlab_data in zip(batch_indicator, lab_dataloader, unlab_dataloader):
            lds, cons = 0, 0
            lab_img, lab_target, lab_filename = (
                lab_data[0].to(device),
                lab_data[1].to(device),
                lab_data[2]
            )

            onehot_target = class2one_hot(lab_target.squeeze(1).cpu(), C=2).to(device).to(device)

            unlab_img, unlab_target, unlab_filename = (
                unlab_data[0].to(device),
                unlab_data[1].to(device),
                unlab_data[2]
            )

            pred_lab = net1(lab_img).softmax(1)
            suploss = crossentropy(pred_lab, onehot_target)

            train_dice, train_error= symmetry_error(pred_lab.max(1)[1], lab_target.squeeze(1))
            train_dice_list.append(train_dice)
            train_error_list.append(train_error)
            # unlab
            if cur_epoch > config['weight_epoch']:
                if config['train_mode'] in ['cons_unlab']:
                    lds = 0
                    unlab_pred = torch.softmax(net1(unlab_img), dim=1)
                    all_shapes_list, shape_error_list, cons_loss = symmetry_cons(unlab_pred)
                else:
                    all_shapes_list, shape_error_list, lds, cons, unlab_pred = vatloss(net1, unlab_img)

                train_ublab_dice, train_unlab_error = symmetry_error(unlab_pred.max(1)[1], unlab_target.squeeze(1))
                train_ublabdice_list.append(train_ublab_dice)
                train_unlaberror_list.append(train_unlab_error)
                #visulization
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

        lab_dsc = average_list(train_dice_list)
        lab_error = average_list(train_error_list)
        try:
            unlab_dsc = average_list(train_ublabdice_list)
            unlab_error = average_list(train_unlaberror_list)
        except:
            pass

        writer.add_scalar('loss/sup', supl.item(), cur_epoch)
        writer.add_scalar('train/dice', lab_dsc[1], cur_epoch)
        writer.add_scalar('train/symmetry_error', lab_error, cur_epoch)

        try:
            writer.add_scalar('loss/lds', lml.item(), cur_epoch)
        except:
            writer.add_scalar('loss/lds', lml, cur_epoch)
        try:
            writer.add_scalar('loss/cons', consl.item(), cur_epoch)
        except:
            writer.add_scalar('loss/cons', consl, cur_epoch)

        try:
            writer.add_scalar('train/unlab_dsc', unlab_dsc, cur_epoch)
            writer.add_scalar('train/unlab_error', unlab_error, cur_epoch)
        except:
            pass

        print(f"Training Epoch {cur_epoch}: suploss: {supl} lds: {lml} cons: {consl}")

#       # val
        val_dsc, symmetry_errorlist = [], []
        for batchid, val_data in enumerate(test_dataloader):
            val_img, val_target, val_filename = (
                val_data[0].to(device),
                val_data[1].to(device),
                val_data[2]
            )
            pred_val = net1(val_img).softmax(1)

            dice, error= symmetry_error(pred_val.max(1)[1], val_target.squeeze(1))
            val_dsc.append(dice)
            symmetry_errorlist.append(error)
            if cur_epoch == 99:
                save_toyseg(pred_val.max(1)[1], names=val_filename, root=dir, mode='predictions')

        dsc = average_list(val_dsc)
        symmetry_erroravg = average_list(symmetry_errorlist)
        writer.add_scalar('val/dice1', dsc[1], cur_epoch)
        writer.add_scalar('val/error', symmetry_erroravg, cur_epoch)










