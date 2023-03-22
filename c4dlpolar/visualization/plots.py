from datetime import datetime, timedelta
import os
import string

from matplotlib import colors, gridspec, lines, patches, pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np

from ..analysis import evaluation


def plot_frame(ax, frame, norm=None):
    im = ax.imshow(frame.astype(np.float32), norm=norm)
    ax.tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)
    return im

def replace_one(x):
    x = np.where(x==1,0,x)
    return x

def replace_zero(x,zero):
    x = np.where(x==zero,-6,x)
    return x


transform_TB = lambda x: x*10+250
transform_radiance = lambda x: x*100
transform_T = lambda x: x*7.2+290
input_transforms = {
    "Rain rate": lambda x: 10**(x*0.528-0.051),
    "CZC": lambda x: x*8.71+21.3,
    "EZC-20": lambda x: x*1.97,
    "EZC-45": lambda x: x*1.97,
    "HZC": lambda x: x*1.97,
    "LZC": lambda x: 10**(x*0.135-0.274),
    "Lightning": lambda x: x,
    "Light. dens.": lambda x: 10**(x*0.640-0.593),
    "Current dens.": lambda x: 10**(x*0.731-0.0718),
    "POH": lambda x: x,
    "$R > 10\\mathrm{mm\\,h^{-1}}$": lambda x: x,
    "KDP": lambda x: replace_zero(x,-0.38300282) * 0.1765 + 0.5776,
    "RHOHV": lambda x: replace_one(x),
    "ZV": lambda x: x*9.154 + 14.9768,
    "ZDR": lambda x: x*0.6675 + 0.297, 
    }
input_norm = {
    "Rain rate": colors.LogNorm(0.01, 100, clip=True),
    "LZC": colors.LogNorm(0.75, 100, clip=True),
    "Light. dens.": colors.LogNorm(0.01, 100, clip=True),
    "Current dens.": colors.LogNorm(0.01, 100, clip=True),
    "Lightning": colors.Normalize(0, 1),
    "POH": colors.Normalize(0, 1),
    "$R > 10\\mathrm{mm\\,h^{-1}}$": colors.Normalize(0, 1),
    "KDP": colors.Normalize(0,2.5, clip=True),
    "ZDR": colors.LogNorm(0.01,5,clip=True),
    "RHOHV": colors.Normalize(0,1),
}
input_ticks = {
    "Rain rate]": [0.1, 1, 10, 100],
    "Lightning": [0, 0.5, 1],
    "POH": [0, 0.5, 1],
    "$R > 10\\mathrm{mm\\,h^{-1}}$": [0, 0.5, 1],
    "KDP": [0.5,1.5,2.5],
    "ZDR": [0.1,5],
    "RHOHV": [0, 0.5, 1],
}


def plot_model_examples(X, Y, models, shown_inputs=(0,25,12,9),
    input_timesteps=(-4,-1), output_timesteps=(0,2,5,11),
    batch_member=0, interval_mins=5,
    input_names=("Rain rate", "KDP", "RHOHHV"),
    future_input_names=(),
    min_p=0.025, plot_scale=256
):
    num_timesteps = len(input_timesteps)+len(output_timesteps)
    gs_rows = 2 * max(len(models),len(shown_inputs))
    gs_cols = num_timesteps
    width_ratios = (
        [0.1, 0.19] +
        [1]*len(input_timesteps) +
        [0.1] +
        [1]*len(output_timesteps) +
        [0.19, 0.1]
    )
    gs = gridspec.GridSpec(gs_rows, gs_cols+5, wspace=0.02, hspace=0.05,
        width_ratios=width_ratios)
    batch = [x[batch_member:batch_member+1,...] for x in X]
    obs = [y[batch_member:batch_member+1,...] for y in Y]

    fig = plt.figure(figsize=(gs_cols*1.5, gs_rows/2*1.5))

    # plot inputs
    row0 = gs_rows//2 - len(shown_inputs)
    for (i,k) in enumerate(shown_inputs):
        row = row0 + 2*i        
        ip = batch[k][0,input_timesteps,:,:,0]
        ip = input_transforms[input_names[i]](ip)
        norm = input_norm[input_names[i]]
        for m in range(len(input_timesteps)):
            col = m+2
            ax = fig.add_subplot(gs[row:row+2,col])
            im = plot_frame(ax, ip[m,:,:], norm=norm)
            if i == 0:
                iv = (input_timesteps[m]+1) * interval_mins
                ax.set_title(f"${iv}\\,\\mathrm{{min}}$")
            if m == 0:
                ax.set_ylabel(input_names[i])
                cax = fig.add_subplot(gs[row:row+2,0])                
                cb = plt.colorbar(im, cax=cax)
                cb.set_ticks(input_ticks[input_names[i]])
                cax.yaxis.set_ticks_position('left')

    # plot outputs
    row0 = 0
    future_input_ind = 0
    norm_log = colors.LogNorm(min_p,1,clip=True)
    for (i,model) in enumerate(models):
        if model == "obs":
            Y_pred = obs[0]
            norm = norm_log
            label = "Observed"
        elif isinstance(model, str) and model.startswith("input-future"):
            var_ind = int(model.split("-")[-1])
            Y_pred = batch[var_ind]
            input_name = future_input_names[future_input_ind]
            Y_pred = input_transforms[input_name](Y_pred)
            norm = input_norm[input_name]
            future_input_ind += 1
            label = input_name
        else:
            Y_pred = model.predict(batch)
            norm = norm_log
            label = "Forecast"
        row = row0 + 2*i
        op = Y_pred[0,output_timesteps,:,:,0]        
        for m in range(len(output_timesteps)):
            col = m + len(input_timesteps) + 3
            ax = fig.add_subplot(gs[row:row+2,col])
            im = plot_frame(ax, op[m,:,:], norm=norm)
            if i==0:
                iv = (output_timesteps[m]+1) * interval_mins
                ax.set_title(f"$+{iv}\\,\\mathrm{{min}}$")
            if m == len(output_timesteps)-1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(label)
                if i == len(models)-1:
                    scalebar = AnchoredSizeBar(ax.transData,
                           op.shape[1],
                           f'{plot_scale} km',
                           'lower center', 
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=1,
                           bbox_transform=ax.transAxes,
                           bbox_to_anchor=(0.5,-0.27)
                    )
                    ax.add_artist(scalebar)

        if i==len(models)-1:
            r0 = row0 + 2*len(future_input_names)
            r1 = r0 + 4
            cax = fig.add_subplot(gs[r0:r1,-1])            
            cb = plt.colorbar(im, cax=cax)
            cb.set_ticks([min_p, 0.05, 0.1, 0.2, 0.5, 1])
            cb.set_ticklabels([min_p, 0.05, 0.1, 0.2, 0.5, 1])
            cax.set_xlabel("$p$", fontsize=12)
        elif i<len(future_input_names):
            cax = fig.add_subplot(gs[row:row+2,-1])            
            cb = plt.colorbar(im, cax=cax)
            cb.set_ticks(input_ticks[input_name])

    return fig


def plot_multiple_models(X, Y, models, model_names, shown_inputs=(0,25,12,9),
    input_timesteps=(-4,-1), output_timesteps=(0,2,5,11),
    batch_member=0, interval_mins=5,
    input_names=("Rain rate", "KDP", "ZDR"),
    future_input_names=(),
    min_p=0.025, plot_scale=256
):
    num_timesteps = len(input_timesteps)+len(output_timesteps)
    gs_rows = 2 * max(len(models.keys()),len(shown_inputs))
    gs_cols = num_timesteps
    width_ratios = (
        [0.1, 0.19] +
        [1]*len(input_timesteps) +
        [0.1] +
        [1]*len(output_timesteps) +
        [0.19, 0.1]
    )
    gs = gridspec.GridSpec(gs_rows, gs_cols+5, wspace=0.02, hspace=0.05,
        width_ratios=width_ratios)
    batch = [x[batch_member:batch_member+1,...] for x in X[model_names[0]]]
    obs = [y[batch_member:batch_member+1,...] for y in Y]

    fig = plt.figure(figsize=(gs_cols*1.5, gs_rows/2*1.5))

    # plot inputs
    row0 = gs_rows//2 - len(shown_inputs)
    for (i,k) in enumerate(shown_inputs):
        row = row0 + 2*i        
        ip = batch[k][0,input_timesteps,:,:,0]
        ip = input_transforms[input_names[i]](ip)
        norm = input_norm[input_names[i]]
        for m in range(len(input_timesteps)):
            col = m+2
            ax = fig.add_subplot(gs[row:row+2,col])
            im = plot_frame(ax, ip[m,:,:], norm=norm)
            if i == 0:
                iv = (input_timesteps[m]+1) * interval_mins
                ax.set_title(f"${iv}\\,\\mathrm{{min}}$")
            if m == 0:
                ax.set_ylabel(input_names[i])
                cax = fig.add_subplot(gs[row:row+2,0])                
                cb = plt.colorbar(im, cax=cax)
                cb.set_ticks(input_ticks[input_names[i]])
                cax.yaxis.set_ticks_position('left')

    # plot outputs
    row0 = 0
    future_input_ind = 0
    norm_log = colors.LogNorm(min_p,1,clip=True)
    i=0
    for (model,item) in models.items():
        if model == "obs":
            Y_pred = obs[0]
            norm = norm_log
            label = "Observed"
        elif isinstance(model, str) and model.startswith("input-future"):
            var_ind = int(model.split("-")[-1])
            Y_pred = batch[var_ind]
            input_name = future_input_names[future_input_ind]
            Y_pred = input_transforms[input_name](Y_pred)
            norm = input_norm[input_name]
            future_input_ind += 1
            label = input_name
        else:
            batch = [x[batch_member:batch_member+1,...] for x in X[model]]
            Y_pred = item.predict(batch)
            norm = norm_log
            label = f"{model.upper()}"
        row = row0 + 2*i
        op = Y_pred[0,output_timesteps,:,:,0]        
        for m in range(len(output_timesteps)):
            col = m + len(input_timesteps) + 3
            ax = fig.add_subplot(gs[row:row+2,col])
            im = plot_frame(ax, op[m,:,:], norm=norm)
            if i==0:
                iv = (output_timesteps[m]+1) * interval_mins
                ax.set_title(f"$+{iv}\\,\\mathrm{{min}}$")
            if m == len(output_timesteps)-1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(label)
                if i == len(models.keys())-1:
                    scalebar = AnchoredSizeBar(ax.transData,
                           op.shape[1],
                           f'{plot_scale} km',
                           'lower center', 
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=1,
                           bbox_transform=ax.transAxes,
                           bbox_to_anchor=(0.5,-0.27)
                    )
                    ax.add_artist(scalebar)
        i+=1

        if i==len(models.keys())-1:
            r0 = row0 + 2*len(future_input_names)
            r1 = r0 + 4
            cax = fig.add_subplot(gs[r0:r1,-1])            
            cb = plt.colorbar(im, cax=cax)
            cb.set_ticks([min_p, 0.05, 0.1, 0.2, 0.5, 1])
            cb.set_ticklabels([min_p, 0.05, 0.1, 0.2, 0.5, 1])
            cax.set_xlabel("$p$", fontsize=12)
        elif i<len(future_input_names):
            cax = fig.add_subplot(gs[row:row+2,-1])            
            cb = plt.colorbar(im, cax=cax)
            cb.set_ticks(input_ticks[input_name])

    return fig

source_colors = {
    "r": "tab:blue",
}


notation = {
    "r": "R",
    "p": "P",
    "q": "Q",
}

prefix_notation = {
    "lightning": "Lightning",
    "hail": "Hail",
    "rain": "Precipitation"
}


def exclusion_plot(metrics, metrics_names, fig=None, axes=None,
    variable_names=None, subplot_index=0, significant_digits=3):

    import seaborn as sns

    metric_notation = {
        "binary": "Error rate",
        "cross_entropy": "CE",
        "mae": "MAE",
        "rmse": "RMSE",
        "FL2": "FL $\\gamma=2$"
    }

    prefixes_names = [prefix_notation[k] for k in metrics]
    metrics_tables = {prefix: np.full((4,2), np.nan) for prefix in metrics}
    metric_pos = {
        frozenset(("q", "r", "p")): (0,0),
        frozenset(("q", "p")): (0,1),

        frozenset(("r", "p")): (1,0),
        frozenset(("p")): (1,1),

        frozenset(("q", "r")): (2,0),
        frozenset(("q")): (2,1),

        frozenset(("r",)): (3,0),
        frozenset(()): (3,1),

    }
    metric_pos_inv = {v: k for (k, v) in metric_pos.items()}

    for prefix in metrics:
        for subset in metrics[prefix]:
            subset_frozen = frozenset(subset)
            (i,j) = metric_pos[subset_frozen]
            metrics_tables[prefix][i,j] = metrics[prefix][subset]

    xlabels_show = frozenset(("r", ""))
    ylabels_show = frozenset(("p", "q"))

    with sns.plotting_context("paper"):
        if fig is None:
            fig = plt.figure(figsize=(3.125*len(metrics),7.5))
        
        for (i,prefix) in enumerate(metrics):
            xlabels = [
                "\n".join(sorted(notation[s] for s in metric_pos_inv[0,i] & xlabels_show))
                for i in range(metrics_tables[prefix].shape[1])
            ]
            ylabels = [
                "\n".join(sorted(notation[s] for s in metric_pos_inv[i,0] & ylabels_show))
                for i in range(metrics_tables[prefix].shape[0])
            ]

            ax = axes[i] if (axes is not None) else fig.add_subplot(1,len(metrics),i+1)
            heatmap = sns.heatmap(
                metrics_tables[prefix],
                xticklabels=xlabels,
                yticklabels=ylabels,
                annot=True,
                fmt='#.{}g'.format(significant_digits),
                square=True,
                ax=ax,
                cbar_kws={"orientation": "horizontal"}
            )
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, ha='right')
            heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha='right')
            ax.set_title("({}) {}{}".format(
                string.ascii_lowercase[i+subplot_index],
                prefixes_names[i]+" " if prefixes_names[i] else "",
                metric_notation[metrics_names[i]]+" " if metrics_names[i] else "",
            ))
            ax.tick_params(axis='both', bottom=False, left=False,
                labelleft=(i+subplot_index==0))

    return fig

def exclusion_ens_plot(metrics, metrics_names, fig=None, axes=None,
    subplot_index=0):

    import seaborn as sns


    metric_notation = {
        "binary": "Error rate",
        "cross_entropy": "CE",
        "mae": "MAE",
        "rmse": "RMSE",
        "FL2": "FL $\\gamma=2$"
    }

    prefixes_names = [prefix_notation[k] for k in metrics]
    metrics_tables = {prefix: np.full((4,2), np.nan) for prefix in metrics}
    sd_tables = {}

    metric_pos = {
        frozenset(("q", "r", "p")): (0,0),
        frozenset(("q", "p")): (0,1),

        frozenset(("r", "p")): (1,0),
        frozenset(("p")): (1,1),

        frozenset(("q", "r")): (2,0),
        frozenset(("q")): (2,1),

        frozenset(("r",)): (3,0),
        frozenset(()): (3,1),

    }
    metric_pos_inv = {v: k for (k, v) in metric_pos.items()}

    for prefix in metrics:
        sd_tables[prefix]={}
        for subset in metrics[prefix]:
            subset_frozen = frozenset(subset)
            (i,j) = metric_pos[subset_frozen]
            if subset == '':
                mean = "1.00000"
                metrics_tables[prefix][i,j] = 1.00000
            else:
                value = np.mean(metrics[prefix][subset])
                mean = str("{:.5f}".format(value))
                metrics_tables[prefix][i,j] = mean
            std_value = np.std(metrics[prefix][subset],ddof=1)
            sd_tables[prefix][mean] = "{:.3f}".format(std_value)

    xlabels_show = frozenset(("r",""))
    ylabels_show = frozenset(("p","q"))

    with sns.plotting_context("paper"):
        if fig is None:
            fig = plt.figure(figsize=(3.125*len(metrics),7.5))
        
        for (i,prefix) in enumerate(metrics):
            xlabels = [
                "\n".join(sorted(notation[s] for s in metric_pos_inv[0,i] & xlabels_show))
                for i in range(metrics_tables[prefix].shape[1])
            ]
            ylabels = [
                "\n".join(sorted(notation[s] for s in metric_pos_inv[i,0] & ylabels_show))
                for i in range(metrics_tables[prefix].shape[0])
            ]

            ax = axes[i] if (axes is not None) else fig.add_subplot(1,len(metrics),i+1)
            heatmap = sns.heatmap(
                metrics_tables[prefix],
                xticklabels=xlabels,
                yticklabels=ylabels,
                annot=True,
                fmt='#.{}f'.format(5),
                square=True,
                ax=ax,
                cbar_kws={"orientation": "horizontal"}
            )
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, ha='right')
            heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha='right')

            for t in heatmap.texts:
                current_text = t.get_text()
                mean_obs = current_text
                    
                def text_transform(current_text):
                    if mean_obs == '1.00000':
                        text = f"1.00"
                    else:
                        sd = sd_tables[prefix][mean_obs]
                        current_text = str(round(float(current_text),3))
                        if len(current_text) ==4 or len(current_text)==3:
                            current_text = current_text+"0"
                        
                        text = f"{(current_text)}\n $\pm$ {sd}"

                    return text

                t.set_text(text_transform(current_text))

            ax.set_title("({}) {}{}".format(
                string.ascii_lowercase[i+subplot_index],
                prefixes_names[i]+" " if prefixes_names[i] else "",
                metric_notation[metrics_names[i]]+" " if metrics_names[i] else "",
            ))
            ax.tick_params(axis='both', bottom=False, left=False,
                labelleft=(i+subplot_index==0))


    return fig


def shapley_values_full_legend(shapley_values_full, ax):
    val_sum_full = sum(shapley_values_full.values())
    labels = [
        f"{notation[s]}: {shapley_values_full[s]/val_sum_full:.03f}"
        for s in shapley_values_full
    ]
    custom_lines = [
        lines.Line2D([0], [0], color=source_colors[s], lw=1)
        for s in shapley_values_full
    ]
    ax.legend(custom_lines, labels, ncol=3, mode="expand")


colors_linestyles = {
("5"): ("#785EF0"),
("10"): ("#33a02c"),
("15"): ("#DC267F"),
("20"): ("#a6cee3"),
("30"): ("#FFB000"),
("60"): ("#648FFF"),
("r"): ("#1f78b4"),
("rp"): ("#ff7f0e"),
("rq"): ("#2ca02c"),
("rpq10"): ("#648FFF"),
("rpq30"): ("#648FFF"),
("rpq50"): ("#648FFF"),

}
linestyles = {
("rpq10"): ("solid"),
("rpq30"): ("dashed"),
("rpq50"): ("dotted"),
}

names_str = {
("5"): ("5 min"),
("10"): ("10 min"),
("15"): ("15 min"),
("20"): ("20 min"),
("30"): ("30 min"),
("60"): ("60 min"),
("r"): ("r"),
("rp"): ("rp"),
("rq"): ("rq"),
("rpq10"): ("10mm"),
("rpq30"): ("30mm"),
("rpq50"): ("50mm"),

}
def plot_CSI_prob(conf_matrix, names, prefix, colors_linestyles=None, show_auc=True,fig=None,ax=None,legend=True,xlabel=True):
    thresholds=np.linspace(0, 1, 101)
    if fig is None:
            fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()
    
    CSI = {}
    labels = {}
    for (k,cm) in conf_matrix.items():
        CSI[k] = evaluation.intersection_over_union(cm)
        m = CSI[k]
        m = m.max(axis=0)
        if prefix == "rain":
            m= m[0]
        else:
            m=m
        labels[k] = f"{names_str[k]}  : {m:.3f}"

    for model in CSI:
        if colors_linestyles is not None:
            ls=None
            c = colors_linestyles[model]
            if prefix == "rain":
                ls = linestyles[model]
        else:
            c = ls = None
        ax.plot(thresholds, CSI[model], label=labels[model],
            color=c, linestyle=ls)
    
    ylim = ax.get_ylim()
    ax.set_ylim((0,ylim[1]))
    ax.set_xlim((0,1))

    ax.legend()

    if xlabel:
        ax.set_xlabel("Threshold")
    if not xlabel:
        ax.set_xticks([])
    ax.set_ylabel("Critical Success Index")

    return (fig,ax)


def get_FSS_files(prefixes=("lightning","hail","rain"),run="run1"):
    files={}
    scales = {"2": 0, "4": 1, "8": 2}

    for p in prefixes:
        files[p] ={}
        if p == "rain":
            for thr in [10,30,50]:
                files[p][thr] = {}
                for src in ["r","rpq"]:
                    dir = f"../runs/{run}/test/"
                    path = os.path.join(dir,f"FSS-{p}{thr}-{src}.npy")
                    FSS_scores = np.load(path)
                    for scale,i in scales.items():
                        FSS = FSS_scores[i,1:,0].max(axis=0)

                        files[p][thr][f"{src}{str(scale)}"] = "{:.3f}".format(FSS)
        else:
            for src in ["r","rpq"]:
                dir = f"..runs/{run}/test/"
                path = os.path.join(dir,f"FSS-{p}-{src}.npy")
                FSS_scores = np.load(path)
                files[p][src] = {str(scale):  FSS_scores[i,1:,:].max(axis=0).round(3) for scale,i in scales.items()}
                
    return files

scale_colors = {
    "2": "#1b9e77",
    "4": "#d95f02",
    "8": "#7570b3",
}

def FSS_by_time(
        leadtimes,
        values,
        interval=timedelta(minutes=5),
        fig=None,
        ax=None,
        legend=True,
    ):

    interval_mins = interval.total_seconds() / 60
    leadtimes = leadtimes * interval_mins
    
    if ax is None:
        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot()

    for src_str in ["r","rpq"]:
        for (source, value) in values[src_str].items():
            if src_str == "rpq":
                ls = "solid"
                label=f"{source} km"
            else:
                ls = "dashed"
                label=None
            ax.plot(
                leadtimes, value, linewidth=1,
                label=label, c=scale_colors[source],linestyle=ls
            )
    if legend:
        ax.legend()
    else:
        rpq = mlines.Line2D([], [], color='black', ls='solid', label='RPQ')
        r = mlines.Line2D([], [], color='black', ls='dashed', label='R')
        ax.legend(handles=[rpq,r])

    ax.set_xlim((0, leadtimes[-1]))
    ax.set_xlabel("Lead time [min]")
    ax.set_ylabel("FSS")

    return fig

def FSS_legend(values, ax):
    source_colors = {
    "r2": "#1b9e77",
    "r4": "#d95f02",
    "r8": "#7570b3",
    "rpq2": "#1b9e77",
    "rpq4": "#d95f02",
    "rpq8": "#7570b3",
}
    ls = {
    "r2": "dashed",
    "r4": "dashed",
    "r8": "dashed",
    "rpq2": "solid",
    "rpq4": "solid",
    "rpq8": "solid",
}   
    values_r = {}
    values_r["rpq2"] = values["rpq2"]
    values_r["r2"] = values["r2"]
    values_r["rpq4"] = values["rpq4"]
    values_r["r4"] = values["r4"]
    values_r["rpq8"] = values["rpq8"]
    values_r["r8"] = values["r8"]
    labels = [f"{values_r[s]}"
        for s in values_r    ]    
    custom_lines = [
        lines.Line2D([0], [0], lw=1,color=source_colors[s],linestyle=ls[s])
        for s in values_r
    ]
    ax.legend(custom_lines, labels, ncol=3, mode="expand")

def plot_probab_curve(conf_matrix, names, prefix, colors_linestyles=None, show_auc=True,fig=None,ax=None,legend=True,xlabel=True):
    thresholds=np.linspace(0, 1, 101)

    if fig is None:
            fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()
    
    colors_linestyles = {
    ("5"): ("#785EF0"),
    ("10"): ("#33a02c"),
    ("15"): ("#DC267F"),
    ("20"): ("#a6cee3"),
    ("30"): ("#FFB000"),
    ("60"): ("#648FFF"),
    ("r"): ("#1f78b4"),
    ("rp"): ("#ff7f0e"),
    ("rq"): ("#2ca02c"),
    ("rpq10"): ("#648FFF"),
    ("rpq30"): ("#648FFF"),
    ("rpq50"): ("#648FFF"),

    }
    linestyles = {
    ("rpq10"): ("solid"),
    ("rpq30"): ("dashed"),
    ("rpq50"): ("dotted"),
    }

    names_str = {
    ("5"): ("5 min"),
    ("10"): ("10 min"),
    ("15"): ("15 min"),
    ("20"): ("20 min"),
    ("30"): ("30 min"),
    ("60"): ("60 min"),
    ("r"): ("r"),
    ("rp"): ("rp"),
    ("rq"): ("rq"),
    ("rpq10"): ("10mm"),
    ("rpq30"): ("30mm"),
    ("rpq50"): ("50mm"),

    }


    CSI = {}
    labels = {}
    for (k,cm) in conf_matrix.items():
        CSI[k] = evaluation.intersection_over_union(cm)
        m = CSI[k]
        m = m.max(axis=0)
        if prefix == "rain":
            m= m[0]
        else:
            m=m
        labels[k] = f"{names_str[k]}  : {m:.3f}"


    for model in CSI:
        if colors_linestyles is not None:
            ls=None
            c = colors_linestyles[model]
            if prefix == "rain":
                ls = linestyles[model]
        else:
            c = ls = None
        ax.plot(thresholds, CSI[model], label=labels[model],
            color=c, linestyle=ls,markevery=5)
    
    ylim = ax.get_ylim()
    ax.set_ylim((0,ylim[1]))
    ax.set_xlim((0,1))

    ax.legend()

    if xlabel:
        ax.set_xlabel("Threshold")
    if not xlabel:
        ax.set_xticks([])
    ax.set_ylabel("Critical Success Index")

    return (fig,ax)

def get_all_files(prefixes=("lightning","hail","rain"),run="run1",sources=["r","rpq"]):
    files={}
    scales = {"2": 0, "4": 1, "8": 2}

    for p in prefixes:
        files[p] ={}
        if p == "rain":
            for thr in [10,30,50]:
                files[p][thr] = {}
                for src in ["r","rpq"]:
                    dir = f"../runs/{run}/results/"
                    path = os.path.join(dir,f"FSS-{p}{thr}-{src}.npy")
                    FSS_scores = np.load(path)
                    for scale,i in scales.items():
                        FSS = FSS_scores[i,1:,0].max(axis=0)

                        files[p][thr][f"{src}{str(scale)}"] = "{:.3f}".format(FSS)
        else:

            for src in ["r","rpq"]:
                dir = f"../runs/{run}/results/"
                path = os.path.join(dir,f"FSS-{p}-{src}.npy")
                FSS_scores = np.load(path)
                files[p][src] = {str(scale):  FSS_scores[i,1:,:].max(axis=0).round(3) for scale,i in scales.items()}
    return files