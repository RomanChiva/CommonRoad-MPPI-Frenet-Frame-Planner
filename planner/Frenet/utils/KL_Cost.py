import torch



def KL_Cost(plan_means, prediction, sigma, plan_sigma, n_samples, discount = 0.5):

        # Convert prediction and sigma to tensors
        prediction = torch.tensor(prediction, dtype=torch.float32)
        sigma = torch.tensor(sigma, dtype=torch.float32)
        

        # Transform Prediction to the correct FOrmat repeat tensor n_samples times
        prediction = prediction.unsqueeze(0).expand(n_samples, -1, -1, -1)

        # Generate Samples
        samples = GenerateSamplesReparametrizationTrick(plan_means, plan_sigma, n_samples)


        # Score
        score_pred = multivariate_normal_log_prob(samples, prediction, sigma)
        score_plan = multivariate_normal_log_prob_scalar_sigma(samples, plan_means, plan_sigma)

        # Compute KL Divergence and discount it
        kl_div = torch.mean(score_plan - score_pred, dim=0)
        kl_div = kl_div.T
        kl_div = discount_weighted_horizon(kl_div, discount)
        kl_div = kl_div.T
        kl_div = torch.sum(kl_div, dim=1)        


        return kl_div





def GenerateSamplesReparametrizationTrick(means, sigma, n_samples):
    # means: tensor of shape (n_distributions, n_components, n_dimensions)
    # covariances: tensor of shape (n_distributions, n_components, n_dimensions, n_dimensions)
    # n_samples: int, number of samples to generate

    # Expand the means tensor to match the number of samples
    expanded_means = means.unsqueeze(0).expand(n_samples, -1, -1, -1)
    
    # Sample noise from a standard normal distribution
    noise = torch.randn_like(expanded_means)

    # Scale the noise by the Cholesky decomposition and add the mean
    samples = expanded_means + sigma * noise

    return samples



def multivariate_normal_log_prob(x, means, covariances):
    # x: tensor of shape (n_samples, n_distributions, n_dimensions)
    # means: tensor of shape (n_distributions, n_dimensions)
    # covariances: tensor of shape (n_distributions, n_dimensions, n_dimensions)

    means = means[0,0,:,:]

    n_distributions = means.shape[0]
    log_probs = []

    for i in range(n_distributions):
        mean = means[i]
        covariance = covariances[i]
        distribution = torch.distributions.MultivariateNormal(mean, covariance)
        log_prob = distribution.log_prob(x[:,:,i,:])
        log_probs.append(log_prob)


    log_probs = torch.stack(log_probs, dim=1)
    log_probs = log_probs.permute(0, 2, 1)


    return log_probs



def multivariate_normal_log_prob_scalar_sigma(x, means, sigma, log=True):
    # x: tensor of shape (..., n_dimensions)
    # means: tensor of shape (..., n_dimensions)
    # sigma: float, standard deviation of the distributions

    if log:
         # Convert sigma to a tensor
        sigma = torch.tensor(sigma, dtype=torch.float32)

        # Compute the log probability of a multivariate normal distribution
        diff = x - means
        exponent = -0.5 * (diff / sigma) ** 2
        log_det = -x.shape[-1] * torch.log(sigma)
        log_2pi = -0.5 * x.shape[-1] * torch.log(torch.tensor(2.0 * 3.1415))

        # Sum the log probabilities over the last dimension
        log_prob = torch.sum(exponent + log_det + log_2pi, dim=-1)

        return log_prob
    
    else:

        # Convert sigma to a tensor
        sigma = torch.tensor(sigma, dtype=torch.float32)

        # Compute the probability of a multivariate normal distribution
        diff = x - means
        exponent = torch.exp(-0.5 * (diff / sigma) ** 2)
        
        normalization = 1 / (torch.sqrt(2 * torch.tensor(3.1415) * sigma) ** x.shape[-1])

        # Multiply the probabilities over the last dimension
        prob = torch.prod(exponent * normalization, dim=-1)

        return prob


def discount_weighted_horizon(costs, discount):
        # Number of elements based on the costs tensor
        time_horizon = costs.shape[0]

        # Generate indices tensor from 0 to time_horizon - 1
        indices = torch.arange(time_horizon, device=costs.device)

        # Generate weights using the discount factor for each timestep
        weights = discount ** indices

        # Multiply costs by weights along the time horizon for each sample
        weighted_costs = costs * weights[:, None]

        return weighted_costs