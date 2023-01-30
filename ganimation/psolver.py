import datetime
import os
import time

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from detection.network.models import model_selection
from ganimation.model import Generator
from loss import DisruptionLoss
from perturbation.unet2d import UNet2D


class Disruptor(nn.Module):
    
    def __init__(self, config, celeba_loader):
        super().__init__()
        
        self.celeba_loader = celeba_loader
        self.dataset = 'CelebA'
        
        # initialize generator
        self.G = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num)
        self.G.load_state_dict(torch.load(config.gen_ckpt, map_location="cuda"))
        # initialize detector
        self.D, *_ = model_selection(modelname='xception', num_out_classes=2)
        self.D.load_state_dict(torch.load(
            config.detector_path, map_location="cuda"))
        #  initialize perturbation generator
        self.P = UNet2D(n_channels=3, n_classes=3)            
        
        for param in self.G.parameters():
            param.requires_grad = False
        for param in self.D.parameters():
            param.requires_grad = False
        
        self.criterion = DisruptionLoss(config)
        
        self.model_save_dir = config.model_save_dir
        self.device = torch.device("cuda")
        self.epochs = config.epochs
        self.resume = config.resume
        self.optim = torch.optim.AdamW(self.P.parameters(), lr=config.lr, 
                                       betas=(config.beta1, config.beta2))
        self.log_step = config.log_step
        if not config.disable_tensorboard:
            self.logger = SummaryWriter(config.log_dir)
        else:
            self.logger = None
        self.sample_step = config.sample_step
        self.sample_dir = config.sample_dir
        
    def detect(self, x):
        logits = self.D(x)
        probabilities = softmax(logits, dim=1)
        # argmin because detector predicts 1 as fake and 0 as real
        return probabilities.argmin(1, keepdim = False).float()
    
    def imFromAttReg(self, att, reg, x_real):
        """Mixes attention, color and real images"""
        return (1-att)*reg + att*x_real
    
    def generate(self, x_real, c_trg, c_org=None):
        att, reg = self.G(x_real, c_trg)
        x_fake = self.imFromAttReg(att, reg, x_real)
        if c_org is not None:
            reconstructed_attention_mask, \
                reconstructed_color_regression = self.G(x_fake, c_org)
            x_rec = self.imFromAttReg(
                reconstructed_attention_mask, reconstructed_color_regression, x_fake)
            return {
                "color_regression": reg,
                "x_fake": x_fake,
                "attention_mask": att,
                "x_rec": x_rec,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_color_regression": reconstructed_color_regression,
            }
        else:
            return {"x_fake": x_fake}
            
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
    
    def print_generations(self, x_real, epoch, generator_outputs_dict, is_perturbed=False):
        save_path = os.path.join(self.sample_dir, str(epoch))
        if is_perturbed:
            save_path = os.path.join(save_path, 'attacked')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        save_image(self.denorm(x_real), save_path + '/4real.png')
        save_image((generator_outputs_dict["color_regression"]+1)/2,
                   save_path + '/2reg.png')
        save_image(self.denorm(
            generator_outputs_dict["x_fake"]), save_path + '/3res.png')
        save_image(generator_outputs_dict["attention_mask"],
                   save_path + '/1attention.png')
        save_image(self.denorm(
            generator_outputs_dict["x_rec"]), save_path + '/5rec.png')

        save_image(generator_outputs_dict["reconstructed_attention_mask"],
                    save_path + '6rec_attention.png')
        save_image(self.denorm(
            generator_outputs_dict["reconstructed_color_regression"]), save_path + '7rec_reg.png')
    
    def train(self):
        """P training"""
        # Set data loader.
        data_loader = self.celeba_loader

        # Fetch fixed inputs for debugging.
        x_fixed, c_org_fixed = iter(data_loader).__next__()
        c_org_fixed = c_org_fixed.to(self.device)
        x_fixed = x_fixed.to(self.device)
        label_fixed = c_org_fixed[torch.randperm(c_org_fixed.size(0))].to(self.device)

        # Start training from scratch or resume training.
        start = 0
        l0 = None
        if self.resume:
            start, l0 = self.restore_model()
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        best_loss = float('inf')
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
            
            self.criterion.set_weights(self.P.weights)
            
            
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
                x_fake = self.generate(x_real, c_trg)["x_fake"]
                # Peturbed image
                perturbation = self.P(x_real)
                x_perturbed = x_real + perturbation
                # fake img gen from perturbed image
                # x_perturbed_fake = self.G(x_perturbed, c_trg)[0]
                x_perturbed_fake = self.generate(x_perturbed, c_trg)["x_fake"]
                # detected fake and real images
                x_perturbed_fake_detected = self.detect(x_perturbed_fake)
                x_perturbed_detected = self.detect(x_perturbed)
                
                losses = self.criterion(x_perturbed_fake, x_fake, 
                                        perturbation, x_perturbed_fake_detected, 
                                        x_perturbed_detected)
                
                if epoch == 0:
                    l0 = losses['task_loss']
                
                self.optim.zero_grad()
                losses['objective_loss'].backward(retain_graph=True)
                self.criterion.zero_grad()
                losses['gradnorm_loss'] = self.criterion.gradNorm(
                    model=self.P, task_loss=losses['task_loss'], initial_task_loss=l0)
                # Logging
                for loss in total_losses:
                    total_losses[loss] += losses[loss].item()
            self.optim.step()
            self.criterion.renormalize()

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
                        
            # Debug images
            if (epoch + 1) % self.sample_step == 0:
                with torch.no_grad():
                    x_perturbed = x_fixed + self.P(x_fixed)
                    original_fake_generations = self.generate(x_fixed, label_fixed, c_org_fixed)
                    attacked_fake_generations = self.generate(
                        x_perturbed, label_fixed, c_org_fixed)
                    self.print_generations(
                        x_fixed, epoch + 1, original_fake_generations)
                    self.print_generations(
                        x_perturbed, epoch + 1, attacked_fake_generations)

            # Save model checkpoints.
            if total_losses['objective_loss'] < best_loss:
                best_loss = total_losses['objective_loss']
                save = f"{self.model_save_dir}/best"
                torch.save(self.P.state_dict(), f"{save}.ckpt")
            self.save_model(epoch, self.P, self.optim, l0)   
            
