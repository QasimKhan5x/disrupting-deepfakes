import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from detection.network.models import model_selection
from garbage.resnet18 import Detection_Model as resnet18
from garbage.resnet50 import Detection_Model as resnet50
from loss import DisruptionLoss
from perturbation.unet2d import UNet2D
from stargan.model import Generator


class Disruptor(nn.Module):
    
    def __init__(self, config, celeba_loader):
        super().__init__()
        
        self.celeba_loader = celeba_loader
        self.dataset = 'CelebA'
        self.device = torch.device("cuda")
        
        # initialize generator
        self.G = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num)
        self.G.load_state_dict(torch.load(config.gen_ckpt, map_location="cuda"))
        
        # initialize detector
        checkpoint = torch.load(config.detector_path, map_location="cuda")
        if config.detector == "xception":
            self.D, *_ = model_selection(modelname='xception', num_out_classes=2)
            self.D.load_state_dict(checkpoint)
        elif config.detector == "resnet18":
            self.D = resnet18()
            self.D.load_state_dict(checkpoint["state_dict"])
        elif config.detector == "resnet50":
            self.D = resnet50()
            self.D.load_state_dict(checkpoint["state_dict"])
        # self.D.eval()
        
        #  initialize perturbation generator
        self.P = UNet2D(n_channels=3, n_classes=3)
        # gradnorm weights
        self.gn_weights = torch.ones(size=(3, ), requires_grad=True, device=self.device)
        
        for param in self.G.parameters():
            param.requires_grad = False
        for param in self.D.parameters():
            param.requires_grad = False
        
        self.criterion = DisruptionLoss(config)
        self.optim1 = torch.optim.AdamW(params=self.parameters(), lr=config.lr)
        self.optim2 = torch.optim.AdamW(params=[self.gn_weights], lr=0.025)
        self.scheduler1 = torch.optim.lr_scheduler.OneCycleLR(self.optim1, max_lr=0.1, 
                                                              total_steps=config.epochs * len(self.celeba_loader))
        self.scheduler2 = torch.optim.lr_scheduler.OneCycleLR(self.optim2, max_lr=0.1, 
                                                              total_steps=config.epochs * len(self.celeba_loader))

        self.c_dim = config.c_dim
        self.alpha = config.alpha
        self.model_save_dir = config.model_save_dir
        self.epochs = config.epochs
        self.selected_attrs = config.selected_attrs
        self.resume = config.resume
        self.log_step = config.log_step
        if not config.disable_tensorboard:
            self.logger = SummaryWriter(config.log_dir)
        else:
            self.logger = None
        self.sample_step = config.sample_step
        self.sample_dir = config.sample_dir
        
    def renormalize(self, initial_sum=3):
        '''
        initial_sum = sum of weights at iteration 0 (3 by default)
        '''
        # n_tasks = self.gn_weights.size(0)
        # normalize_coeff = n_tasks / torch.sum(self.gn_weights, dim=0)
        # self.gn_weights.grad.data = self.gn_weights.grad.data * normalize_coeff
        self.gn_weights = (self.gn_weights / self.gn_weights.sum() * initial_sum).detach()

    def gradNorm(self, model, task_loss, initial_task_loss):
        shared_layer = list(model.children())[-1]
        # compute the L2 norm of the gradients for each task
        gw = []
        for i, loss in enumerate(task_loss):
            dl = torch.autograd.grad(
                self.gn_weights[i] * loss, shared_layer.parameters(), retain_graph=True, create_graph=True)[0]
            gw.append(torch.norm(dl))
        gw = torch.stack(gw)
        # compute loss ratio per task
        loss_ratio = task_loss.detach() / initial_task_loss
        # compute the relative inverse training rate per task
        rt = loss_ratio / loss_ratio.mean()
        # compute the average gradient norm
        gw_avg = gw.mean().detach()
        # compute the GradNorm loss
        constant = (gw_avg * rt ** self.alpha).detach()
        gradnorm_loss = torch.abs(gw - constant).sum()
        return gradnorm_loss
    
    # def gradNorm(self, model, task_loss, initial_task_loss):
    #     '''
    #     https://github.com/brianlan/pytorch-grad-norm/blob/master/train.py#96
        
    #     model: perturbation model
    #     task_loss: unweighted loss for perturbation, 
    #                detection_loss_fake, detection_loss_real
    #     initial_task_loss: task_loss at epoch 0 (should be numpy array)
    #     '''
    #     if not isinstance(initial_task_loss, np.ndarray):
    #         initial_task_loss = initial_task_loss.detach().cpu().numpy()
    #     shared_layer = list(model.children())[-1]
    #     # get the gradient norms for each of the tasks
    #     # G^{(i)}_w(t)
    #     norms = []
    #     for i in range(len(task_loss)):
    #         # get the gradient of this task loss with respect to the shared parameters
    #         gygw = torch.autograd.grad(
    #             task_loss[i], shared_layer.parameters(), retain_graph=True)
    #         # compute the norm
    #         norms.append(torch.norm(
    #             torch.mul(self.gn_weights[i], gygw[0])))
    #     norms = torch.stack(norms)
    #     # print('G_w(t): {}'.format(norms))

    #     # compute the inverse training rate r_i(t)
    #     # \curl{L}_i
    #     initial_task_loss[initial_task_loss == 0] = 1e-5
    #     if torch.cuda.is_available():
    #         loss_ratio = task_loss.detach().cpu().numpy() / initial_task_loss
    #     else:
    #         loss_ratio = task_loss.detach().numpy() / initial_task_loss
    #     # r_i(t)
    #     inverse_train_rate = loss_ratio / np.mean(loss_ratio)
    #     # print('r_i(t): {}'.format(inverse_train_rate))

    #     # compute the mean norm \tilde{G}_w(t)
    #     if torch.cuda.is_available():
    #         mean_norm = np.mean(norms.detach().cpu().numpy())
    #     else:
    #         mean_norm = np.mean(norms.detach().numpy())
    #     # print('tilde G_w(t): {}'.format(mean_norm))

    #     # compute the GradNorm loss
    #     # this term has to remain constant
    #     constant_term = torch.tensor(
    #         mean_norm * (inverse_train_rate ** self.alpha), requires_grad=False)
    #     if torch.cuda.is_available():
    #         constant_term = constant_term.cuda()
    #     # print('Constant term: {}'.format(constant_term))
    #     # this is the GradNorm loss itself
    #     grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
    #     # print('GradNorm loss {}'.format(grad_norm_loss))

    #     # compute the gradient for the weights
    #     self.gn_weights.grad = torch.autograd.grad(
    #         grad_norm_loss, self.gn_weights)[0]

    #     return grad_norm_loss
      
    def detect(self, x):
        logits = self.D(x)
        probabilities = softmax(logits, dim=1)
        # argmin because detector predicts 1 as fake and 0 as real
        return probabilities.argmin(1, keepdim = False).float() 
    
    def save_model(self, epoch, model, optimizer, l0):
        print("Saving model on epoch", epoch, "in", self.model_save_dir)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'l0': l0,
            }, f'{self.model_save_dir}/checkpoint.ckpt')
    
    def restore_model(self):
        ckpt = torch.load(f'{self.model_save_dir}.pth', map_location="cuda")
        epoch = ckpt['epoch']
        print(f"Resuming training at epoch {epoch}...",)
        self.P.load_state_dict(ckpt['model_state_dict'])
        self.optim.load_state_dict(ckpt['optimizer_state_dict'])
        l0 = ckpt['l0']
        return epoch, l0
    
    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    def train(self):
        """P training on selected attributes"""
        # Set data loader.
        data_loader = self.celeba_loader

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = iter(data_loader).__next__()
        x_fixed = x_fixed.to(self.device)
        label_fixed = c_org[torch.randperm(c_org.size(0))].to(self.device)

        # Start training from scratch or resume training.
        start = 0
        l0 = None
        if self.resume:
            start, l0 = self.restore_model()
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        best_loss = float('inf')
        first_it = True
        for epoch in range(start, self.epochs):
            
            total_losses = {
                'objective_loss': 0,
                'distortion_loss': 0,
                'perturbation_loss': 0,
                'detection_loss_fake': 0,
                'detection_loss_real': 0,
                'gradnorm_loss': 0
            }
            
            torch.cuda.empty_cache()
            
            pbar = tqdm(data_loader, total=len(data_loader))
            for i, (x_real, label_org) in enumerate(pbar):
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                c_org = label_org.clone()
                c_trg = label_trg.clone()

                x_real = x_real.to(self.device)           # Input images.
                c_org = c_org.to(self.device)             # Original domain labels.
                c_trg = c_trg.to(self.device)             # Target domain labels.
                # Labels for computing classification loss.
                label_org = label_org.to(self.device)
                # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)
                
                # =================================================================================== #
                #                             2. Train the Perturber                              #
                # =================================================================================== #
                
                """ 
                1. x_fake = G(x)
                2. perturbation = P(x)
                3. x_perturbed = x + P(x)
                4. x_perturbed_fake = G(x + P(x))
                5. x_perturbed_fake_detected = D(G(x + P(x)))
                6. x_perturbed_detected = D(x + P(x))
                """
                # fake image
                x_fake = self.G(x_real, c_trg)[0]
                # Peturbed image
                perturbation = self.P(x_real)
                x_perturbed = x_real + perturbation
                # fake img gen from perturbed image
                x_perturbed_fake = self.G(x_perturbed, c_trg)[0]
                # detected fake and real images
                x_perturbed_fake_detected = self.detect(x_perturbed_fake)
                x_perturbed_detected = self.detect(x_perturbed)
                
                losses = self.criterion(x_perturbed_fake, x_fake,
                                        perturbation, x_perturbed_fake_detected,
                                        x_perturbed_detected, weights=self.gn_weights)

                if first_it:
                    l0 = losses['task_loss'].detach()
                    first_it = False

                self.optim1.zero_grad()
                losses['objective_loss'].backward(retain_graph=True)
                self.optim2.zero_grad()
                losses['gradnorm_loss'] = self.gradNorm(model=self.P,
                                                        task_loss=losses['task_loss'],
                                                        initial_task_loss=l0)
                losses['gradnorm_loss'].backward()
                # update loss weights
                self.optim2.step()
                self.scheduler2.step()
                # update network weights
                self.optim1.step()
                self.scheduler1.step()
                # Logging
                for loss in total_losses:
                    total_losses[loss] += losses[loss].item()
                self.renormalize()                
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (epoch+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Epoch [{}/{}]".format(
                    et, epoch+1, self.epochs)
                for tag, value in total_losses.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print("==========================================")
                print(log)
                print("==========================================")
                if self.logger is not None:
                    for tag, value in total_losses.items():
                        self.logger.add_scalar(tag, value, epoch+1)
                    for tag, value in zip(['c1', 'c2', 'c3'], self.gn_weights):
                        self.logger.add_scalar(tag, value, epoch+1)
                        
            # Debug images
            if (epoch + 1) % self.sample_step == 0:
                with torch.no_grad():
                    x_perturbed = x_fixed + self.P(x_fixed)
                    x_fake, _ = self.G(x_fixed, label_fixed)
                    x_perturbed_fake, _ = self.G(x_perturbed, label_fixed)
                if not os.path.isdir(f'{self.sample_dir}/{epoch+1}'):
                    os.mkdir(f'{self.sample_dir}/{epoch+1}')
                save_image(self.denorm(x_fixed.cpu()),
                            f'{self.sample_dir}/{epoch+1}/orig.jpg')
                save_image(self.denorm(x_perturbed.cpu()),
                            f'{self.sample_dir}/{epoch+1}/perturbed.jpg')
                save_image(self.denorm(x_fake.cpu()),
                            f'{self.sample_dir}/{epoch+1}/fake.jpg')
                save_image(self.denorm(x_perturbed_fake.cpu()),
                            f'{self.sample_dir}/{epoch+1}/attacked.jpg')

            # Save model checkpoints.
            if total_losses['objective_loss'] < best_loss:
                best_loss = total_losses['objective_loss']
                save = f"{self.model_save_dir}/best"
                torch.save(self.P.state_dict(), f"{save}.ckpt")
            self.save_model(epoch, self.P, self.optim1, l0)   
            
