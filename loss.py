import numpy as np
import torch
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F


class DisruptionLoss(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.eps = config.eps
        self.order = config.order
        
    
    def forward(self, x_perturbed_fake, x_fake, perturbation,
                x_perturbed_fake_detected, x_perturbed_detected,
                weights):
        '''
        x_perturbed_fake = G(x + P(x))
        x_fake = G(x)
        perturbation = P(x)
        x_perturbed_fake_detected = D(G(x + P(x)))
        x_perturbed_detected = D(x + P(x))
        '''
    
        l1 = distortion_loss(x_perturbed_fake, x_fake, order=self.order)
        l2 = perturbation_loss(perturbation, eps=self.eps)
        l3 = detection_loss_fake(x_perturbed_fake_detected)
        l4 = detection_loss_real(x_perturbed_detected)
        
        task_loss = torch.stack([l2, l3, l4])
        objective_loss = l1 + weights @ task_loss
        
        return {
            'objective_loss': objective_loss,
            'task_loss': task_loss,
            'distortion_loss': l1,
            'perturbation_loss': l2,
            'detection_loss_fake': l3,
            'detection_loss_real': l4
        }
              
     
def distortion_loss(x_perturbed_fake, x_fake, order):
    '''
    x_perturbed_fake = G(x + P(x))
    x_fake = G(x)
    '''
    B = x_fake.size(0)
    x_perturbed_fake = x_perturbed_fake.view(B, -1)
    x_fake = x_fake.view(B, -1)
    diff = x_perturbed_fake - x_fake
    return LA.norm(diff, axis=1, ord=order).mean()


def perturbation_loss(p_x, eps):
    '''
    p_x = P(x)
    '''
    p_x = p_x.view(p_x.size(0), -1)
    norm = LA.norm(p_x, axis=1)
    diff = norm - eps
    hinge = F.relu(diff)
    return hinge.mean()
    

def detection_loss_fake(z):
    '''
    x_perturbed_fake = G(x + P(x))
    z = D(x_fake)
    ^ minimize predicted logits
    '''
    return z.mean()


def detection_loss_real(z):
    '''
    z = D(x + P(x))
    '''
    return (1 - z).mean()


def disruption_loss(attacked_x, gen_x, perturbed_x, 
                    x_fake_perturbed, x_real_perturbed,
                    c1, c2, c3, 
                    eps, order):
    l1 = distortion_loss(attacked_x, gen_x, order=order)
    l2 = perturbation_loss(perturbed_x, eps=eps)
    l3 = detection_loss_fake(x_fake_perturbed)
    l4 = detection_loss_real(x_real_perturbed)
    
    return -l1 + (c1 * l2) + (c2 * l3) + (c3 * l4)

if __name__ == "__main__":
    from types import SimpleNamespace

    import torch
    
    config = SimpleNamespace(**{
        'order': 2,
        'eps': 0.05,
        'c1': 1.0,
        'c2': 1.0,
        'c3': 1.0,
        'alpha': 0.1
    })
    
    # order = 2
    # eps = 0.05
    # c1 = c2 = c3 = 1.0
    
    B = 8
    C = 3
    H, W = 256, 256
    
    attacked_x = torch.randn((B, C, H, W))
    gen_x = torch.randn((B, C, H, W))
    perturbed_x = torch.randn((B, C, H, W))
    x_fake_perturbed = torch.randn((B, C, H, W))
    x_real_perturbed = torch.randn((B, C, H, W))
    
    criterion = DisruptionLoss(config)
    loss = criterion(attacked_x, gen_x, perturbed_x, x_fake_perturbed, x_real_perturbed)
    print(loss.item())
    # print(criterion.we)
    
    # loss = disruption_loss(attacked_x, gen_x, perturbed_x, 
    #                        x_fake_perturbed, x_real_perturbed,
    #                        c1, c2, c3, eps, order)
    # print(loss)