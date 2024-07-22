import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import numpy as np
import os
import seaborn as sns

import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# tfp particulars
tfd = tfp.distributions
root = tfd.JointDistributionCoroutine.Root

# Split the data into train and test data
def split_train_test_data(sub_image_df, path_to_df, n_forecasting):
    # join the paths
    complete_path_to_df = os.path.join(path_to_df, sub_image_df)
    # read the csv 
    read_df = pd.read_csv(complete_path_to_df)

    # split the data into train and test
    train_df = read_df.iloc[:-n_forecasting, :]
    print(train_df.shape)
    test_df = read_df.iloc[-n_forecasting:,:]
    print(test_df.shape)

    # get the obs data
    train_y = train_df['tassel_count']
    test_y = test_df['tassel_count']

    # make these float 32 for bayes ts implementation
    train_y = train_y.astype("float32")
    test_y = test_y.astype("float32")

    # these needs to be returned

    # also split the covariate data
    # but add an intercept before the split?
    read_df.insert(0, 'intercept', np.repeat(1, read_df.shape[0]))
    # make this float32 for bayes ts implementation
    read_df['intercept'] = read_df['intercept'].astype("float32")

    # now can extract the covariate data
    X_preds = read_df.drop(['tassel_count'], axis = 1).astype("float32")
    X_preds = X_preds.values
    print(X_preds.shape)
    n_preds = X_preds.shape[-1]
    return(train_y, test_y, X_preds, n_preds) 
    
# redefine the plot function
def plot_tassel_count_data(train_data, test_data, df_no, fig, ax):
    # if not fig_ax:
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # else:
    #     fig, ax = fig_ax
    ax.plot(train_data, color = 'blue', label="training data")
    ax.plot(test_data, color = 'lightcoral', label="testing data")
    ax.legend()
    ax.set(
        ylabel="Tassel counts" ,
        xlabel="Time",
        title = "Tassel count distribution for sub image " + str(df_no)
    )
    fig.autofmt_xdate()
    fig.show()
    return fig, ax

def get_prioirs_and_x_beta(X_pred, n_pred, beta_mu, beta_sigma, noise_scale):
    beta = yield root(tfd.Sample(
        tfd.Normal(beta_mu, beta_sigma),
        sample_shape=n_pred,
        name='beta'))
    x_beta = tf.einsum('ij,...j->...i', X_pred, beta)

    noise_sigma = yield root(tfd.HalfNormal(scale=noise_scale, name='noise_sigma'))

    intercept_data = X_pred[:,0]

    return (x_beta, intercept_data, noise_sigma)

# define the number of train time periods
train_tp = 13

def generate_model_ar_latent(preds_data, n_pred, ar_scale, rho_lower, beta_mu, beta_sigma, noise_scale, training=True):

    @tfd.JointDistributionCoroutine
    def model_with_latent_ar():
        x_beta, intercept_data, noise_sigma = yield from get_prioirs_and_x_beta(preds_data, n_pred, beta_mu, beta_sigma, noise_scale)
        
        # Latent AR(1)
        ar_sigma = yield root(tfd.HalfNormal(ar_scale, name='ar_sigma'))
        rho = yield root(tfd.Uniform(rho_lower, 1., name='rho'))
        def ar_fun(y):
            loc = tf.concat([tf.zeros_like(y[..., :1]), y[..., :-1]],
                            axis=-1) * rho[..., None]
            return tfd.Independent(
                tfd.Normal(loc=loc, scale=ar_sigma[..., None]),
                reinterpreted_batch_ndims=1)
        temporal_error = yield tfd.Autoregressive(
            distribution_fn=ar_fun,
            sample0=tf.zeros_like(intercept_data),
            num_steps=intercept_data.shape[-1],
            name='temporal_error')

        # Linear prediction
        y_hat = x_beta + temporal_error
        if training:
            y_hat = y_hat[..., :train_tp]

        # Likelihood
        observed = yield tfd.Independent(
            tfd.Normal(y_hat, noise_sigma[..., None]),
            reinterpreted_batch_ndims=1,
            name='observed'
        )

    return model_with_latent_ar

# define a function to plot the X_beta and temporal errors
def preds_and_temoral_error(mcmc_samples_data, preds_data, n_total_time_points, nchains):
    # plot components
    fig, ax = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    beta = mcmc_samples_data[0]
    seasonality_posterior = tf.einsum('ij,...j->...i', preds_data, beta)
    temporal_error = mcmc_samples_data[-1]

    for i in range(nchains):
        ax[0].plot(np.arange(n_total_time_points), seasonality_posterior[-100:, i, :].numpy().T, alpha=.05);
        ax[1].plot(np.arange(n_total_time_points), temporal_error[-100:, i, :].numpy().T, alpha=.05);

    ax[0].set_title('X_beta effect')
    ax[1].set_title('Temporal error')
    ax[1].set_xlabel("Day")
    fig.autofmt_xdate()

# Plot the forecated and actual values
def forecasted_and_actual_values_plot(ppc_sample_data, train_counts, test_counts, df_no, fig, ax):
    fitted_with_forecast = ppc_sample_data[-1].numpy()
    
    ax.plot(np.arange(20), fitted_with_forecast[:250, 0, :].T, color='gray', alpha=.1);
    ax.plot(np.arange(20), fitted_with_forecast[:250, 1, :].T, color='gray', alpha=.1);
    
    plot_tassel_count_data(train_counts, test_counts, df_no, fig, ax)
    average_forecast = np.mean(fitted_with_forecast, axis=(0, 1)).T
    ax.plot(np.arange(20), average_forecast, ls='--', label='latent AR forecast', color = 'red', alpha=.5);
    plt.xticks(np.arange(20))
    plt.legend()
    plt.show()

# create a function to get the posteriors and the trace plots with az
def get_nuts_values_and_posterior_plots(mcmc_samples_bts, sampler_stats_bts):
    nuts_trace_ar_latent = az.from_dict(posterior={k:np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples_bts._asdict().items()},
    sample_stats = {k:np.swapaxes(sampler_stats_bts[k], 1, 0)for k in ["target_log_prob", "diverging", "accept_ratio", "n_steps"]})

    axes = az.plot_trace(nuts_trace_ar_latent, var_names=['beta', 'ar_sigma', 'rho', 'noise_sigma'], compact=True);

    plt.tight_layout()
    return(nuts_trace_ar_latent)

def posterior_vals(nuts_model):
    rho_values = nuts_model.posterior.rho
    print(rho_values.shape)
    ar_sigma_values = nuts_model.posterior.ar_sigma
    print(ar_sigma_values.shape)
    noise_sigma_values = nuts_model.posterior.noise_sigma
    print(noise_sigma_values.shape)
    beta_vals_all = nuts_model.posterior.beta
    print(beta_vals_all.shape)

    return(rho_values, ar_sigma_values, noise_sigma_values, beta_vals_all)

# colors for traceplots
color_list = ['cornflowerblue', 'lightsteelblue', 'blue', 'mediumblue', 'cyan', 'deepskyblue', 'steelblue', 'dodgerblue', 'lightslategray', 'mediumslateblue',
             'lightblue', 'teal', 'royalblue', 'indianred', 'deepskyblue', 'honeydew', 'lightseagreen', 'turquoise', 'cadetblue', 'tan', 'moccasin', 'burlywood',
             'peachpuff', 'powderblue', 'mediumaquamarine', 'powderblue', 'thistle', 'lavender', 'lightcyan', 'darkseagreen', 'honeydew', 'lightsteelblue', 'cadetblue']
len(color_list)

# color palattes for freq polygons
color_palletes_betas = ['Greys','Purples','Blues','Greens','Oranges','Reds','YlOrBr','YlOrRd','OrRd','PuRd','RdPu','BuPu','GnBu','PuBu','YlGnBu',
 'PuBuGn','BuGn','YlGn','Greys','Purples','Blues','Greens','Oranges','Reds','YlOrBr','YlOrRd','OrRd','PuRd','RdPu','BuPu','GnBu','PuBu', 'YlGnBu']
len(color_palletes_betas)

# define a function for this?

# for a chosen parameter - param

def get_trace_plots(ax, param, color, string_params):
    ax.plot(param.T, color = color, alpha = 0.5)
    ax.set_title("Trace plot for " + string_params, fontsize=10, fontweight="bold")

# Create a function for this?
def get_freq_curves(ax, param, color_palette, string_params):
    sns.kdeplot(data=param.T, fill=False, ax=ax, legend = False, palette = color_palette)
    ax.set_title("Frequency plot for " + string_params, fontsize=10, fontweight="bold")

def fit_Bayes_TS_model(csv_file, forecasting_steps, path_to_preprocessed_dfs, sub_image_number, n_features, nchains, ar_scale, rho_lower, beta_mu, beta_sigma, noise_scale):
    # get the counts and predictors
    Train_Y, Test_Y, X_preds_only, n_predictors = split_train_test_data(csv_file, path_to_preprocessed_dfs, forecasting_steps)
    # plot the train and test counts
    ## fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ## plot_tassel_count_data(Train_Y, Test_Y, sub_image_number, fig, ax)
    # define the mcmc function
    run_mcmc = tf.function(tfp.experimental.mcmc.windowed_adaptive_nuts, autograph=False, jit_compile=True)
    # define the latet ar model
    gam_with_latent_ar = generate_model_ar_latent(X_preds_only, n_predictors, ar_scale, rho_lower, beta_mu, beta_sigma, noise_scale, training=True)
    # plot samples from prior predictive distribution
    ## plt.figure(figsize = (10,8))
    ## plt.plot(tf.transpose(gam_with_latent_ar.sample(500)[-1]))
    ## plt.show()
    # run the mcmc with nuts sampler
    mcmc_samples, sampler_stats = run_mcmc(1000, gam_with_latent_ar, n_chains=4, num_adaptation_steps=1000, seed=tf.constant([36245, 734565], dtype=tf.int32), observed=Train_Y.T)
    # get the posterior values for parameters and the posterioir predictions
    gam_with_latent_ar_full = generate_model_ar_latent(X_preds_only, n_predictors, ar_scale, rho_lower, beta_mu, beta_sigma, noise_scale, training=False)
    posterior_dists, ppc_samples = gam_with_latent_ar_full.sample_distributions(value=mcmc_samples)
    # plot the posteriors ? for betas and temporal errors
    # preds_and_temoral_error(mcmc_samples, X_preds_only, n_total_time_points = 20, nchains = 4)
    # Plot the forecated and actual values
    ## fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ## forecasted_and_actual_values_plot(ppc_samples, Train_Y, Test_Y, sub_image_number, fig, ax)
    # Get the posterior plots
    nuts_output = get_nuts_values_and_posterior_plots(mcmc_samples, sampler_stats)
    # recreate this plot
    rho_values, ar_sigma_values, noise_sigma_values, beta_vals_all = posterior_vals(nuts_output)
    # get the betas gathered and plot them all in a single plot
    all_betas = [beta_vals_all[: ,: , i] for i in range(n_features)]
    # # recreate the plot
    # fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    # get_trace_plots(axs[0, 0], rho_values, 'palevioletred', 'rho')
    # get_trace_plots(axs[1, 0], noise_sigma_values, 'blue', 'noise sigma')
    # get_trace_plots(axs[2, 0], ar_sigma_values, 'indianred', 'ar sigma')
    # for i in range(33):
    #     get_trace_plots(axs[3, 0], all_betas[i], color_list[i], 'betas')
    # fig.tight_layout()
    # get_freq_curves(axs[0, 1], rho_values, 'Blues', 'rho')
    # get_freq_curves(axs[1, 1], noise_sigma_values, 'Purples', 'noise sigma')
    # get_freq_curves(axs[2, 1], ar_sigma_values, 'Greens', 'ar sigma')
    # for i in range(33):
    #     get_freq_curves(axs[3, 1], all_betas[i], color_palletes_betas[i], 'betas')
    # fig.tight_layout()
    # # plt.plot(all_betas[i].T, color = color_list[i], alpha = 0.5)
    # # figure_path_and_name = figure_folder_path + '/' + 'all_trace_plots_sub_' + str(sub_image_number) + '.png'
    # # plt.savefig(figure_path_and_name)
    # plt.show()
    # get the forecasted values
    forecasted_values = ppc_samples[-1].numpy()
    averaged_forecast = np.mean(forecasted_values, axis=(0, 1)).T
    print(forecasted_values.shape, averaged_forecast.shape)
    # we may need to store the averaged forecasts - extract these first for the test set only
    test_averaged_forecast = averaged_forecast[-forecasting_steps:]
    test_all_forecasts = forecasted_values[:,:,-forecasting_steps:]
    # create a dataframe for true and averaged forecasts for test data
    final_forecasted_values = pd.DataFrame(zip(Test_Y, test_averaged_forecast), columns = ['True_value', 'Forecasted_value'])
    # save the avrage and all forecasts for future use
    # avg_frcst_file_name = forecasts_folder_path + '/' + 'averaged_forecasts_sub_' + str(sub_image_number) + '.csv'
    # final_forecasted_values.to_csv(avg_frcst_file_name, index = False)
    # all_frcst_file_name = forecasts_folder_path + '/' + 'all_forecasts_sub_' + str(sub_image_number) + '.npy'
    # np.save(all_frcst_file_name, test_all_forecasts)
    # get the parameter summary
    parameter_summary = az.summary(nuts_output)
    # parameter_summary_file_name = forecasts_folder_path + '/' + 'posterior_parameter_summary_sub_' + str(sub_image_number) + '.csv'
    # parameter_summary.to_csv(parameter_summary_file_name, index = False)
    # save the posterior predicted values for other parameters
    # rho_file_name = forecasts_folder_path + '/' + 'rho_values_sub_' + str(sub_image_number) + '.npy'
    # np.save(rho_file_name, rho_values)
    # noise_sigma_file_name = forecasts_folder_path + '/' + 'noise_sigma_values_sub_' + str(sub_image_number) + '.npy'
    # np.save(noise_sigma_file_name, noise_sigma_values)
    # ar_sigma_file_name = forecasts_folder_path + '/' + 'ar_sigma_values_sub_' + str(sub_image_number) + '.npy'
    # np.save(ar_sigma_file_name, ar_sigma_values)
    # beta_file_name = forecasts_folder_path + '/' + 'beta_values_sub_' + str(sub_image_number) + '.npy'
    # np.save(beta_file_name, all_betas)
    return(final_forecasted_values, test_all_forecasts)

# put all these in a function? - note this function has 12 in the for loop for the 12 sub images - this need to change when we have overlapping data
def end_to_end_metrics(sub_image_files_list, sub_image_numbers, forecasting_steps, path_to_preprocessed_dfs,  n_features, nchains, ar_scale, rho_lower, beta_mu, beta_sigma, noise_scale):
    # get the average and all forecasts
    all_dfs = []
    for i in np.arange(12):
        get_df = fit_Bayes_TS_model(sub_image_files_list[i], forecasting_steps, path_to_preprocessed_dfs, sub_image_numbers[i], n_features, nchains, ar_scale, rho_lower, beta_mu, beta_sigma, noise_scale)
        all_dfs.append(get_df)

    # collect only the average forecasts dfs
    df_only = []
    for i in range(len(all_dfs)):
        df = all_dfs[i][0]
        df_only.append(df)

    # combine all the dfs - to get the finalized average tassel counts
    combined_dfs = pd.concat(df_only, axis = 1)
    # seperate the true and forecasted values
    True_counts_df = combined_dfs[['True_value']]
    total_true_values = True_counts_df.sum(axis = 1)
    Forecasted_counts_df = combined_dfs[['Forecasted_value']]
    total_forecasted_values = Forecasted_counts_df.sum(axis = 1)

    # combine these two together
    true_and_forecasted_values_df = pd.concat((total_true_values, total_forecasted_values), axis = 1)
    true_and_forecasted_values_df.columns = ["True_count", "Forecasted_count"]
    
    # compute the metrics
    rmse = np.sqrt(mean_squared_error(total_true_values, total_forecasted_values))
    mae = mean_absolute_error(total_true_values, total_forecasted_values)
    corr = pearsonr(total_true_values, total_forecasted_values)[0]
    # for widths and coverages we need the following
    # collect all the forecasted values, and sum them up to get the totals
    all_forecasts_arrays = []
    for i in range(len(all_dfs)):
        all_forecasts = all_dfs[i][1]
        all_forecasts_arrays.append(all_forecasts)
    # sum all these up
    output = sum(all_forecasts_arrays)
    # reshape these
    final_array = output.reshape(4000,7)
    # compute percentiles
    li_train = np.percentile(final_array, axis = 0, q = (2.5, 97.5))[0,:].reshape(-1,1)    
    ui_train = np.percentile(final_array, axis = 0, q = (2.5, 97.5))[1,:].reshape(-1,1)
    # compute width
    width_train = ui_train - li_train
    avg_width_train = width_train.mean(0)[0]
    # compute the coverage
    y_true = true_and_forecasted_values_df[["True_count"]].values
    ind_train = (y_true >= li_train) & (y_true <= ui_train)
    coverage_train= ind_train.mean(0)[0]
    # return the metrics
    return(rmse, mae, corr, avg_width_train, coverage_train)

# From this point onwards the code needs to change to match the block we are working on
forecasting_steps = 7
path_to_preprocessed_dfs = '../all_preprocessed_data/Block_0103/TS_ready_data_frames/'
# sub_image_number = 0
n_features = 33
nchains = 4

sub_image_numbers = np.arange(12)

# hyper parameter combos
ar_scale_vals = [0.1, 5]
rho_lower_vals = [-1, 0.5]
beta_mu_vals = [0]
beta_sigma_vals = [1, 5]
noise_scale_vals = [1, 5]

# get rid of the checkpointing folder
sub_image_files = [file for file in os.listdir(path_to_preprocessed_dfs) if file[-3:] == 'csv']
sub_image_files.sort()
# add the 10, 11 at the end
im_files = ['extracted_features_sub_window_10.csv', 'extracted_features_sub_window_11.csv']
other_files = [i for i in sub_image_files if i not in im_files]
sub_image_files = other_files + im_files

# do the loop for the HPOs
catch_all_contents = []
for ar_scale in ar_scale_vals:
    print(ar_scale)
    for rho_lower in rho_lower_vals:
        print(rho_lower)
        for beta_mu in beta_mu_vals:
            print(beta_mu)
            for beta_sigma in beta_sigma_vals:
                print(beta_sigma)
                for noise_scale in noise_scale_vals:
                    print(noise_scale)
                    all_metrics = end_to_end_metrics(sub_image_files, sub_image_numbers, forecasting_steps, path_to_preprocessed_dfs,
                                                                                          n_features, nchains, ar_scale, rho_lower, beta_mu, beta_sigma, noise_scale)
                    catch_all_contents.append(all_metrics)


# conver the list of contents to a dataframe
metrics_df = pd.DataFrame(catch_all_contents)
metrics_df.columns = ['rmse', 'mae', 'corr', 'avg_width_train', 'coverage_train']
# convert the hyper parameters to a dataframe
all_HPO_combos = []
for ar_scale in ar_scale_vals:
    for rho_lower in rho_lower_vals:
        for beta_mu in beta_mu_vals:
            for beta_sigma in beta_sigma_vals:
                for noise_scale in noise_scale_vals:
                    hpo = [ar_scale, rho_lower, beta_mu, beta_sigma, noise_scale]
                    all_HPO_combos.append(hpo)

hpo_df = pd.DataFrame(all_HPO_combos)
hpo_df.columns = ['ar_scale', 'rho_lower', 'beta_mu', 'beta_sigma', 'noise_scale']
# combine the two dataframes
combined_df = pd.concat((hpo_df, metrics_df ), axis = 1)
combined_df.to_csv("Store_metric_results/hpo_metrics_for_block0103.csv", index = False)