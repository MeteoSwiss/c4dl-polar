import os
import string

from matplotlib import pyplot as plt, gridspec
import numpy as np
from scipy.stats import norm
import tensorflow as tf

from c4dlpolar.analysis import calibration, evaluation, shapley
from c4dlpolar.visualization import plots

from training import model_sources


def exclusion_plot(prefix="lightning", out_file=None, fig=None, axes=None):
    scores = shapley.load_scores(
        f"../results/{prefix}/test/",
        prefix=prefix
    )
    scores_norm = {s: v/scores[''] for (s,v) in scores.items()}
    loss = "FL2" if prefix == "lightning" else "cross_entropy"
    fig = plots.exclusion_plot({loss: scores_norm}, fig=fig, axes=axes)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)


def exclusion_plot_all(prefixes=("lightning", "hail", "rain"), out_file=None):
    scores = {
        p: shapley.load_scores(
            f"../results/{p}/test/",
            prefix=p
        )
        for p in prefixes
    }
    scores_norm = {
        p:
            {s: v/scores[p][''] for (s,v) in scores[p].items()}
        for p in prefixes
    }
    losses = [
        "FL2" if p == "lightning" else "cross_entropy"
        for p in prefixes
    ]
    fig = plots.exclusion_plot(scores_norm, losses)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)

null_scores = {
"rain": 2.134406e-02,
"hail": 1.583702e-03,
"lightning":1.847274e-02
}

def exclusion_plot_all_ens(prefixes=("lightning", "hail", "rain"),out_file=None):
    scores = {p: shapley.load_ens_scores(prefix=p)
        for p in prefixes}
    scores_norm = {
        p:
            {s: v/null_scores[p] for (s,v) in scores[p].items()}
        for p in prefixes
    }
    losses = [
    "FL2" if p == "lightning" else "cross_entropy"
    for p in prefixes
    ]

    fig = plots.exclusion_ens_plot(scores_norm, losses)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)

def plot_examples(
    batch_gen, model, batch_number=13,
    batch_member=30, out_file=None, 
    shown_inputs=("RZC", "KDP"),
    input_names=("Rain rate [mm h$^{-1}$]", "KDP"),
    shown_future_inputs=(),
    future_input_names=(),
    preprocess_rain=False,
    plot_kwargs=None
):
    if plot_kwargs is None:
        plot_kwargs = {}

    names = batch_gen.pred_names_past + batch_gen.pred_names_future
    shown_inputs = [names.index(ip) for ip in shown_inputs]
    shown_future_inputs = [names.index(ip) for ip in shown_future_inputs]
    future_input_codes = [f"input-future-{ip}" for ip in shown_future_inputs]

    (X,Y) = batch_gen.batch(batch_number, dataset='test')
    if preprocess_rain:
        ip = [tf.keras.Input(shape=x.shape[1:]) for x in X]
        pred = model(ip)
        pred = tf.expand_dims(pred, axis=1)
        pred = tf.reduce_sum(pred[...,1:], axis=-1, keepdims=True)
        model = tf.keras.Model(inputs=ip, outputs=pred)

        rr = 10**Y[0].astype(np.float32)
        rr = rr.mean(axis=1, keepdims=True)
        sig = np.sqrt(np.log(0.33**2+1))
        mu = np.log(rr) - 0.5*sig**2
        Y[0] = norm.sf(np.log(10.0), loc=mu, scale=sig)

    fig = plots.plot_model_examples(X, Y, future_input_codes+["obs", model],
        batch_member=batch_member, shown_inputs=shown_inputs,        
        input_names=input_names, future_input_names=future_input_names,
        **plot_kwargs)

    if preprocess_rain:
        for ax in fig.axes:
            if ax.get_title() == "$+5\\,\\mathrm{min}$":
                ax.set_title("Next $60\\,\\mathrm{min}$")
                break

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight', dpi=200)
        plt.close(fig)

def plot_two_models(
    batch_gens, models, model_names, prefix, batch_number=13,
    batch_member=30, out_file=None, 
    shown_inputs=("RZC", "occurrence-8-10", "KDP", "ZDR-CORR"),
    input_names=("Rain rate", "Lightning", "KDP", "ZDR"),
    shown_future_inputs=(),
    future_input_names=(),
    preprocess_rain=False,
    plot_kwargs=None
):
    """
    Visualizes specifc event, predictors, observed and forecasted    
    """
    if plot_kwargs is None:
        plot_kwargs = {}

    names = batch_gens[model_names[0]].pred_names_past + batch_gens[model_names[0]].pred_names_future
    shown_inputs = [names.index(ip) for ip in shown_inputs]
    shown_future_inputs = [names.index(ip) for ip in shown_future_inputs]

    Xs = {}
    (X,Y) = batch_gens[model_names[1]].batch(batch_number, dataset='test')

    if preprocess_rain:
        ip = [tf.keras.Input(shape=x.shape[1:]) for x in X]
        pred = models[model_names[1]](ip)
        pred = tf.expand_dims(pred, axis=1)
        pred = tf.reduce_sum(pred[...,1:], axis=-1, keepdims=True)
        model_1 = tf.keras.Model(inputs=ip, outputs=pred)
        
    else:
        model_1 = models[model_names[1]]

    Xs[model_names[1]] = X
    (X,Y) = batch_gens[model_names[0]].batch(batch_number, dataset='test')
    if preprocess_rain:
        ip = [tf.keras.Input(shape=x.shape[1:]) for x in X]
        pred = models[model_names[0]](ip)
        pred = tf.expand_dims(pred, axis=1)
        pred = tf.reduce_sum(pred[...,1:], axis=-1, keepdims=True)
        model_2 = tf.keras.Model(inputs=ip, outputs=pred)

        rr = 10**Y[0].astype(np.float32)
        rr = rr.mean(axis=1, keepdims=True)
        sig = np.sqrt(np.log(0.33**2+1))
        mu = np.log(rr) - 0.5*sig**2
        Y[0] = norm.sf(np.log(10.0), loc=mu, scale=sig)
    else:
        model_2 = models[model_names[0]]

    Xs[model_names[0]] = X
    

    fig = plots.plot_multiple_models(Xs, Y, {"obs": "obs", model_names[0]: model_2, model_names[1]: model_1},
        model_names,batch_member=batch_member, shown_inputs=shown_inputs,        
        input_names=input_names, future_input_names=future_input_names,
        **plot_kwargs)

    if preprocess_rain:
        for ax in fig.axes:
            if ax.get_title() == "$+5\\,\\mathrm{min}$":
                ax.set_title("Next $60\\,\\mathrm{min}$")
                break

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight', dpi=200)
        plt.close(fig)

def plot_sources_models(batch_gens, models, prefix="lightning",sample=(32,11),plot_kwargs=None,out_file=None):
    if prefix == "lightning":
        if plot_kwargs == None:
            plot_kwargs = {"min_p": 0.025}        
        plot_two_models(
        batch_gens, models, model_names=["rpq","r"],
        prefix=prefix,
        batch_number=sample[0],
        batch_member=sample[1], 
        shown_inputs=("RZC","KDP","ZDR-CORR"),
        input_names=("Rain rate", "KDP", "ZDR"),
        shown_future_inputs=(),
        future_input_names=(),
        out_file=out_file,
        plot_kwargs=plot_kwargs
        )

    elif prefix == "rain":
        if plot_kwargs == None:
            plot_kwargs = {"min_p": 5e-4, "output_timesteps": [0]}
        plot_two_models(
        batch_gens, models, model_names=["rpq","r"], 
        prefix=prefix,
        batch_number=sample[0],
        batch_member=sample[1], 
        shown_inputs=("RZC","KDP","ZDR-CORR"),
        input_names=("Rain rate", "KDP", "ZDR"),
        shown_future_inputs=(),
        future_input_names=(),
        plot_kwargs=plot_kwargs,
        out_file=out_file,
        preprocess_rain=True
        )

    elif prefix == "hail":
        if plot_kwargs == None:
            plot_kwargs = {"min_p": 5e-5}
        plot_two_models(
        batch_gens, models, model_names=["rpq","r"], 
        prefix=prefix,
        batch_number=sample[0],
        batch_member=sample[1], 
        shown_inputs=("BZC","RZC","KDP","ZDR-CORR"),
        input_names=("POH","Rain rate", "KDP", "ZDR"),
        shown_future_inputs=(),
        future_input_names=(),
        plot_kwargs=plot_kwargs,
        out_file=out_file,
    )
        
def plot_metrics_leadtime(
    metric=evaluation.intersection_over_union,
    prefixes=("lightning", "hail"), metric_name="CSI",
    sources_str="rpq", step_minutes=5,
    out_fn=None
):
    fig = plt.figure()
    ax = fig.add_subplot()

    for p in prefixes:
        fn = f"../results/{p}/test/conf_matrix_leadtime-{p}-{sources_str}.npy"
        conf_matrix = np.load(fn)
        m = metric(conf_matrix)
        if m.ndim > 1:
            m = m.max(axis=0)
        x = np.arange(1, len(m)+1) * step_minutes
        label = p.capitalize()
        ax.plot(x, m, label=label)

    ax.legend()
    ax.set_xlim((0, x[-1]))
    ax.set_ylim((0, ax.get_ylim()[1]))
    ax.set_xlabel("Lead time [min]")
    ax.set_ylabel(metric_name)

    if out_fn is not None:
        fig.savefig(out_fn, bbox_inches='tight')

    plt.close(fig)



metrics = [
    ("CSI", evaluation.intersection_over_union),
    ("ETS", evaluation.equitable_threat_score),
    ("HSS", evaluation.heidke_skill_score),
    ("PSS", evaluation.peirce_skill_score),
    ("ROC AUC", evaluation.roc_area_under_curve),
    ("PR AUC", evaluation.pr_area_under_curve),
]
def plot_metrics_leadtime_all():
    for (metric_name, metric) in metrics:
        fn_metric_name = metric_name.replace(" ", "_")
        fn = f"../figures/{fn_metric_name}-leadtime.pdf"
        plot_metrics_leadtime(metric, metric_name=metric_name, out_fn=fn)


def rain_metrics_table(thresholds=(10,30,50), sources_str="rpq", ):
    for (metric_name, metric) in metrics:
        print(metric_name, end=' ')
        for t in thresholds:
            fn = "../results/rain/test/" + \
                f"conf_matrix_leadtime-rain{t}-{sources_str}.npy"
            conf_matrix = np.load(fn)
            m = metric(conf_matrix)
            if m.ndim > 1:
                m = m.max(axis=0)
            m = m[0]
            print(f"{m:.3f}", end=' ')
        print()


# for hail and lightning
def shapley_leadtime_plots(run="run1",out_file=None, sources = 'rpq'):
    files = shapley.get_all_files(run=run)

    leadtimes = np.arange(1, 13)
    fig = plt.figure(figsize=(6,9))
    gs = gridspec.GridSpec(7, 1, height_ratios=(1,5,0.75,1,5,0.75,1),
        hspace=0.3)
    
    # plot Shapley values by lead time
    prefixes = ("lightning", "hail")
    subplots = [1, 4]
    for (k,prefix) in enumerate(prefixes):
        values = {source: np.zeros(len(leadtimes)) for source in sources}
        for (i,lt) in enumerate(leadtimes):
            scores = shapley.load_scores(files[prefix],
                score_index=lt
            )
            print(scores)
            for source in sources:
                values[source][i] = shapley.shapley_value(scores, source)
        
        ax = fig.add_subplot(gs[k*3+1,0])
        plots.shapley_by_time(leadtimes, values, 
            fig=fig, ax=ax, legend=False)
        
    prefixes = ("lightning", "hail", "rain")
    subplots = [0, 3, 6]
    for (k,prefix) in enumerate(prefixes):
        # plot legend with full shapley values
        scores_full = shapley.load_scores(files[prefix]
        )
        values_full = {s: shapley.shapley_value(scores_full, s) for s in sources}
        ax = fig.add_subplot(gs[subplots[k],0])
        plots.shapley_values_full_legend(values_full, ax)
        ax.axis("off")
        ax.set_title(f"({string.ascii_lowercase[k]}) {plots.prefix_notation[prefix]}")

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)


def plot_all_CSI(run="run1",src_str=["rpq"],prefixes=("lightning","hail","rain"),leadtime=True,out_file=None):
    
    fig = plt.figure(figsize=(6,8))

    dir = f"../runs/{run}/results/"
        
    for i,p in enumerate(prefixes):
        ax = fig.add_subplot(len(prefixes), 1, i+1)
        if p == "hail" and leadtime==True:
            src = src_str[0]
            path = os.path.join(dir,f"conf_matrix_leadtime-{p}-{src}.npy")
            conf_matrix = np.load(path)
            lts = {"5": 0, "15": 2, "30": 5, "60": 11}
            conf_matrix = {str(lt):  conf_matrix[:,:,:,i] for lt,i in lts.items()}
            names = {str(lt): str(lt) for lt in lts}
            
        elif p == "lightning" and leadtime==True:
            src = src_str[0]
            path = os.path.join(dir,f"conf_matrix_leadtime-{p}-{src}.npy")
            conf_matrix = np.load(path)
            lts = {"5": 0, "15": 2, "30": 5, "60": 11}
            conf_matrix = {str(lt):  conf_matrix[:,:,:,i] for lt,i in lts.items()}
            names = {str(lt): str(lt) for lt in lts}

        else:
            if len(src_str) == 1:
                print(src_str)
                src = src_str[0]
                conf_matrix_files = {f"{src}{thr}": f"conf_matrix_leadtime-{p}{thr}-{src}.npy" for thr in [10,30,50]}
                names = {src: src}
                print(names)
            if p == "rain":
                conf_matrix = {
                    k: np.load(os.path.join(dir, fn))
                    for (k,fn) in conf_matrix_files.items()}
            else:
                conf_matrix = {
                    k: np.load(os.path.join(dir, fn))
                    for (k,fn) in conf_matrix_files.items()}
            
        plots.plot_probab_curve(conf_matrix=conf_matrix, names=names,prefix=p,show_auc=True,fig=fig,ax=ax,xlabel=(i==len(prefixes)-1))

        ax.set_title("({}) {}".format(
                string.ascii_lowercase[i],
                plots.prefix_notation[p]+" " if plots.prefix_notation[p] else "",
            ),horizontalalignment='left', verticalalignment='top',x=0.4)
    
    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)


def FSS_plots(prefixes=("lightning","hail","rain"),run="run1",out_file=None):
    files = plots.get_all_files(prefixes=prefixes,run=run)
    leadtimes = np.arange(1, 13)
    fig = plt.figure(figsize=(6,9))
    gs = gridspec.GridSpec(8, 1, height_ratios=(1,5,1,5,1,0.75,0.75,0.75),
        hspace=0.3)
    
    # plot FSS by lead time
    prefixes = ("lightning", "hail")
    subplots = [0, 1]
    for (k,prefix) in enumerate(prefixes):
        values = files[prefix]
        leg=k==0
        ax = fig.add_subplot(gs[k*2+1,0])
        plots.FSS_by_time(leadtimes, values, 
            fig=fig, ax=ax, legend=leg)
        ax.set_title(f"({string.ascii_lowercase[k]}) {plots.prefix_notation[prefix]}")

    thresholds = (10,30,50)
    subplots = [5,6,7]
    prefix="rain"
    for (k,thr) in enumerate(thresholds):
        # plot legend with FSS values
        scores = files["rain"][thr]
        values_full = {s: scores[s] for s in scores}
        ax = fig.add_subplot(gs[subplots[k],0])
        print(values_full)
        plots.FSS_legend(values_full, ax)
        ax.axis("off")
        if k==0:
            ax.set_title(f"(c) {plots.prefix_notation[prefix]}")
        ax.text(x=-0.11,y=.12,s=f"{str(thr)} mm")
    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)



def get_models(prefix,model1=["rpq","run3"],model2=["r","run1"]):
    
    if prefix == "lightning":
        target = "occurrence-8-10"
    elif prefix == "rain":
        target = "CPCH"
    elif prefix == "hail":
        target = "BZC"

    (_, batch_gen_1, model_1, _) = model_sources(model1[0],
        target=target)

    (_, batch_gen_2, model_2, _) = model_sources(model2[0],
        target=target)
    model_2.load_weights(f"../runs/{model2[1]}/{prefix}-{model2[0]}.h5")
    model_1.load_weights(f"../runs/{model1[1]}/{prefix}-{model1[0]}.h5")
    if prefix == "lightning":
        p = np.linspace(0,1,101)
        occurrence_1 = np.load(f"../runs/{model1[1]}/calibration/calibration-lightning-{model1[0]}.npy")
        occurrence_2 = np.load(f"../runs/{model2[1]}/calibration/calibration-lightning-{model2[0]}.npy")

        calib_model_1 = calibration.calibrated_model(model_1, p, occurrence_1)
        calib_model_2 = calibration.calibrated_model(model_2, p, occurrence_2)

        batch_gens = {model2[0]: batch_gen_2,model1[0]: batch_gen_1}
        models = {model2[0]: calib_model_2,model1[0]: calib_model_1}
    else:
        batch_gens = {model2[0]: batch_gen_2,model1[0]: batch_gen_1}
        models = {model2[0]: model_2,model1[0]: model_1}
    return batch_gens,models