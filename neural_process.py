import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from random import randint
from torch.distributions.kl import kl_divergence
from IPython.display import clear_output



def context_target_split(x,y,num_context,num_extra_target):
	# select the indices that represent the context and the targets for every batch
	locations = np.random.choice(x.shape[1],size=num_context + num_extra_target,replace=False)
	# select the context points for every batch
	x_context = x[:, locations[:num_context], :]
	y_context = y[:, locations[:num_context], :]
	# select the target points for every batch
	x_target = x[:, locations, :]
	y_target = y[:, locations, :]
	return x_context, y_context, x_target, y_target


class Encoder(nn.Module):
	## The respective x_i,y_i pairs are concatenated and treated as a single input
	## Each of these represenations is mapped to a represenatation r_i by the same neural net
    # x_dim: x dimension
    # y_dim: y dimension
    # h_dim: dimensionality of the NN layer
    # r_dim: dimensionality of the output representation of the NN
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        ## concatenate x and y
        ## map them through a NN to a represenation space
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)


class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.
    """
    ## r_dim: representation space
    ## z_dim: latent dimensionality 
    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class Decoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.
    """
    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        layers = [nn.Linear(x_dim + z_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x, z):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        z : torch.Tensor
            Shape (batch_size, z_dim)
        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # latent represenation z has to be concatenated with every x so it has to be repeated
        # instead of (batch_size, z_dim) to (batch_size, num_points, z_dim)

        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat), dim=1)
        hidden = self.xz_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
       
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return mu, sigma


class NeuralProcess(nn.Module):
    
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim):
        super(NeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Initialize networks
        self.xy_to_r = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = Decoder(x_dim, z_dim, h_dim, y_dim)

    def aggregate(self, r_i):
        return torch.mean(r_i, dim=1)

    def xy_to_mu_sigma(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # Return parameters of distribution
        return self.r_to_mu_sigma(r)

    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.
        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.
        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)
        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)
        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.
        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        batch_size, num_context, x_dim = x_context.size()
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()

        if (y_target != None):
            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target)
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from encoded distribution using reparameterization trick
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_target.rsample()
            # Get parameters of output distribution
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred

class NeuralProcessTrainer():
  
	def __init__(self, device, neural_process, optimizer, num_context_range,num_extra_target_range, print_freq=100):
        
		self.device = device
		self.neural_process = neural_process
		self.optimizer = optimizer
		self.num_context_range = num_context_range
		self.num_extra_target_range = num_extra_target_range
		self.print_freq = print_freq

		self.steps = 0
		self.epoch_loss_history = []

	def train(self, data_loader, epochs):

		loss_list = []
        
		for epoch in range(epochs):
			epoch_loss = 0.
			for i, data in enumerate(data_loader):
				self.optimizer.zero_grad()

				# Sample number of context and target points
				num_context = randint(*self.num_context_range)
				num_extra_target = randint(*self.num_extra_target_range)

				x, y = data

				'''print(x.shape)
				print(y.shape)
				print(num_context)
				print(num_extra_target)'''

				x_context, y_context, x_target, y_target = context_target_split(x, y, num_context, num_extra_target)
				p_y_pred, q_target, q_context = self.neural_process(x_context.float(), y_context.float(), x_target.float(), y_target.float())

				loss = self._loss(p_y_pred, y_target, q_target, q_context)
				loss.backward()
				self.optimizer.step()

				epoch_loss += loss.item()

				self.steps += 1

			print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
			clear_output(wait=True)
			self.epoch_loss_history.append(epoch_loss / len(data_loader))


		return self.epoch_loss_history

	def _loss(self, p_y_pred, y_target, q_target, q_context):    
		log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
		kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
		return -log_likelihood + kl