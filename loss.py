import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA


class DisruptionLoss(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.eps = config.eps
        self.order = config.order
        self.alpha = config.alpha
        self.weights = None
        
    def set_weights(self, weights):
        self.weights = weights
    
    def forward(self, x_perturbed_fake, x_fake, perturbation,
                x_perturbed_fake_detected, x_perturbed_detected):
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
        
        # print("l1=",l1)
        # print("l2=",l2)
        # print("l3=",l3)
        # print("l4=",l4)
        # print("weights=", self.weights)
        
        task_loss = torch.stack([l2, l3, l4])
        objective_loss =  -l1 + self.weights @ task_loss
        return objective_loss, task_loss
        
    def zero_grad(self):
        self.weights.grad = self.weights.grad * 0.0
    
    def renormalize(self):
        n_tasks = self.weights.size(0)
        normalize_coeff = n_tasks / torch.sum(self.weights, dim=0)
        self.weights.grad.data = self.weights.grad.data * normalize_coeff
    
    def gradNorm(self, model, task_loss, initial_task_loss):
        '''
        https://github.com/brianlan/pytorch-grad-norm/blob/master/train.py#96
        
        model: perturbation model
        task_loss: unweighted loss for perturbation, 
                   detection_loss_fake, detection_loss_real
        initial_task_loss: task_loss at epoch 0 (should be numpy array)
        '''
        if not isinstance(initial_task_loss, np.ndarray):
            initial_task_loss = initial_task_loss.detach().cpu().numpy()
        shared_layer = list(model.children())[-1]
        # get the gradient norms for each of the tasks
        # G^{(i)}_w(t)
        norms = []
        for i in range(len(task_loss)):
            # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(
                task_loss[i], shared_layer.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.norm(
                torch.mul(self.weights[i], gygw[0])))
        norms = torch.stack(norms)
        # print('G_w(t): {}'.format(norms))

        # compute the inverse training rate r_i(t)
        # \curl{L}_i
        if torch.cuda.is_available():
            loss_ratio = task_loss.detach().cpu().numpy() / initial_task_loss
        else:
            loss_ratio = task_loss.detach().numpy() / initial_task_loss
        # r_i(t)
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)
        # print('r_i(t): {}'.format(inverse_train_rate))

        # compute the mean norm \tilde{G}_w(t)
        if torch.cuda.is_available():
            mean_norm = np.mean(norms.detach().cpu().numpy())
        else:
            mean_norm = np.mean(norms.detach().numpy())
        # print('tilde G_w(t): {}'.format(mean_norm))

        # compute the GradNorm loss
        # this term has to remain constant
        constant_term = torch.tensor(
            mean_norm * (inverse_train_rate ** self.alpha), requires_grad=False)
        if torch.cuda.is_available():
            constant_term = constant_term.cuda()
        # print('Constant term: {}'.format(constant_term))
        # this is the GradNorm loss itself
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
        # print('GradNorm loss {}'.format(grad_norm_loss))

        # compute the gradient for the weights
        self.weights.grad = torch.autograd.grad(
            grad_norm_loss, self.weights)[0]
        
        return grad_norm_loss

    
def distortion_loss(x_perturbed_fake, x_fake, order):
    '''
    x_perturbed_fake = G(x + P(x))
    x_fake = G(x)
    '''
    x_perturbed_fake = x_perturbed_fake.view(-1)
    # print(x_fake.shape)
    x_fake = x_fake.view(-1)
    difference = x_perturbed_fake - x_fake
    # print(x_fake.shape)
    # print("difference:", difference.mean())
    norm = LA.norm(difference, ord=order)
    return norm.mean()


def perturbation_loss(p_x, eps):
    '''
    p_x = P(x)
    '''
    p_x = p_x.view(-1)
    return F.relu(LA.norm(p_x) - eps)


def detection_loss_fake(z):
    '''
    x_perturbed_fake = G(x + P(x))
    z = D(x_fake)
    ^ minimize predicted logits
    '''
    return z.view(-1).mean()


def detection_loss_real(z):
    '''
    z = D(x + P(x))
    '''
    return (1 - z.view(-1)).mean()


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
    print(criterion.we)
    
    # loss = disruption_loss(attacked_x, gen_x, perturbed_x, 
    #                        x_fake_perturbed, x_real_perturbed,
    #                        c1, c2, c3, eps, order)
    # print(loss)