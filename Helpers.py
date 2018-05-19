# Calculate the perplexity of a given estimator on the validation data.
def calculate_perplexity(likelihood, validation):
    log_likelihood = likelihood
    # Take the negative average of the log likelihood.
    log_likelihood /= -len(validation)
    # Return 2 in power of the log likelihood to retrieve the perplexity.
    return pow(2, log_likelihood)

