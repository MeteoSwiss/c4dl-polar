import argparse
from datetime import datetime, timedelta
import gc
import os

import dask
import numpy as np

from c4dlpolar.analysis import calibration, evaluation
from c4dlpolar.features import batch, regions, transform
from c4dlpolar.ml.models import models


def setup_batch_gen(
    file_dir, file_suffix="2020", primary="RZC",
    target="R10", batch_size=48, epoch=datetime(1970,1,1),
    sources=("rad", "qualindex", "polar",)
):
    pred_names = [
        "RZC", "CZC",
        "EZC-20", "EZC-45",
        "HZC", "LZC",
        "AREA57","BZC","CPCH",
        "occurrence-8-10",
        "KDP","RHOHV","ZV-CORR","ZDR-CORR",
        "obs-qual"
    ]
    pred_names = select_sources(pred_names, sources)

    for preds in ["CPCH","BZC","occurrence-8-10"]:
        if preds not in pred_names:
            pred_names.append(preds)

    if primary not in pred_names:
        pred_names.append(primary)
    if target not in pred_names:
        pred_names.append(target)

    files = [f'patches_{pred}_2020.nc' for pred in pred_names]

    files = {
        fn.split("_")[1]: os.path.join(file_dir,fn) for fn in files
    } # map variable name to file

    # raw data
    raw = {
        var_name: dask.delayed(regions.load_patches)(fn)
        for (var_name, fn) in files.items()
    }
    raw = dask.compute(raw, scheduler="processes")[0]

    #ignorant model
    if not sources:
        raw["zeros"] = {
            "patches": np.zeros(
                (0,)+raw[primary]["patches"].shape[1:],
                dtype=np.float32
            ),
            "patch_coords": np.empty((0,2), dtype=np.uint16),
            "patch_times": np.empty(0, dtype=np.int64),
            "zero_patch_coords": np.vstack((
                raw[primary]["patch_coords"],
                raw[primary]["zero_patch_coords"]
            )).T,
            "zero_patch_times": np.hstack((
                raw[primary]["patch_times"],
                raw[primary]["zero_patch_times"]
            )),
            "zero_value": np.float32(0.0)
        }

    if "qualindex" in sources: 
        var = "obs-qual"
        raw[var]["static"] = True

     

    # features and targets are defined by transforming the raw data
    transforms = {"zeros": {
            "source_vars": ["zeros"],
            "transform": lambda x: x
        },
        "BZC-target": {
            "source_vars": ["BZC"],
            "transform": transform.scale_norm(raw["BZC"]["scale"],
                std=100.0, dtype=np.float16)
        },
        "BZC": {
            "source_vars": ["BZC"],
            "transform": transform.scale_norm(raw["BZC"]["scale"],
                std=100.0, dtype=np.float16)
        },

        "R10-target": {
            "source_vars": ["CPCH"],
            "transform": transform.R_threshold(raw["CPCH"]["scale"], 10.0)
        },
        "R10": {
            "source_vars": ["CPCH"],
            "transform": transform.R_threshold(raw["CPCH"]["scale"], 10.0)
        },
        "CPCH": {
            "source_vars": ["CPCH"],
            "transform": transform.scale_log_norm(raw["CPCH"]["scale"],
                threshold=0.1, fill_value=0.01, mean=0.0, std=1.0,
                dtype=np.float16)
        },
        "CPCH-target": {
            "source_vars": ["CPCH"],
            "transform": transform.scale_log_norm(raw["CPCH"]["scale"],
                threshold=0.1, fill_value=0.01, mean=0.0, std=1.0,
                dtype=np.float16)
        },
        "occurrence-8-10-target": {
            "source_vars": ["occurrence-8-10"],
            "transform": transform.cast(np.uint8)
        },
        "occurrence-8-10": {
            "source_vars": ["occurrence-8-10"],
            "transform": transform.cast(np.uint8)
        }}
    
    if 'rad' in sources:
        transforms.update({
        "RZC": {
            "source_vars": ["RZC"],
            "transform": transform.scale_log_norm(raw["RZC"]["scale"],
                threshold=0.1, fill_value=0.01, mean=-0.051, std=0.528,
                dtype=np.float16)
        },
        "CZC": {
            "source_vars": ["CZC"],
            "transform": transform.scale_norm(raw["CZC"]["scale"],
                threshold=5.0, fill_value=-5.0, mean=21.3, std=8.71,
                dtype=np.float16)
        },
        "EZC-20": {
            "source_vars": ["EZC-20"],
            "transform": transform.scale_norm(raw["EZC-20"]["scale"],
                std=1.97, dtype=np.float16)
        },
        "EZC-45": {
            "source_vars": ["EZC-45"],
            "transform": transform.scale_norm(raw["EZC-45"]["scale"],
                std=1.97, dtype=np.float16)
        },
        "HZC": {
            "source_vars": ["HZC"],
            "transform": transform.scale_norm(raw["HZC"]["scale"],
                std=1.97, dtype=np.float16)
        },
        "LZC": {
            "source_vars": ["LZC"],
            "transform": transform.scale_log_norm(raw["LZC"]["scale"],
                threshold=0.75, fill_value=0.5, mean=-0.274, std=0.135,
                dtype=np.float16)
        },        
        "AREA57": {
            "source_vars": ["AREA57"],
            "transform": transform.normalize(std=14.0, dtype=np.float16)
        }})
    
    if 'polar' in sources:
        transforms.update({
        "KDP": {
            "source_vars": ["KDP","RZC"],
            "transform": transform.transform_polar(mean=0.0676,std=0.1765,fill_value=0,lb=-0.5,ub=5)
        },
        "RHOHV": {
            "source_vars": ["RHOHV","RZC"],
            "transform": transform.transform_polar(mean=0,std=1,fill_value=1,lb=0.7,ub=1)
        },
        "ZV-CORR": {
            "source_vars": ["ZV-CORR","RZC"],
            "transform": transform.transform_polar(mean=14.9768,std=9.154,fill_value=-10.01,lb=-10,ub=60)
        },

        "ZDR-CORR": {
            "source_vars": ["ZDR-CORR","RZC"],
            "transform": transform.transform_polar(mean=0.2979,std=0.6675,fill_value=0,lb=-1.5,ub=5)
        }})

    if 'qualindex' in sources:
        transforms.update({
        "obs-qual": {
            "source_vars": ["obs-qual"],
            "transform": lambda x: x ,
            "timeframe": "static"   
        }})    
    # predictors
    pred_names = [
        "RZC", "CZC",
        "EZC-20", "EZC-45",
        "HZC", "LZC",
        "KDP","RHOHV","ZV-CORR","ZDR-CORR",
        "obs-qual"
    ]

    if not ("CPCH" in transforms[target]["source_vars"]):
        pred_names.append(target)

    pred_names = select_sources(pred_names, sources)
    if not pred_names:
        pred_names = ["zeros"] # prediction with no input data

    predictors = {
        var_name: transforms[var_name]
        for var_name in pred_names
    }

    # targets
    target_names = [target+"-target"]
    targets = {var_name: transforms[var_name] for var_name in target_names}

    # we need one "primary" raw data variable
    # that determines the location of the data for all variables
    primary_patch_data = raw[primary]    
    (box_locs, t0) = regions.box_locations(
        primary_patch_data["patch_coords"],
        primary_patch_data["patch_times"],
        primary_patch_data["zero_patch_coords"],
        primary_patch_data["zero_patch_times"]
    )

    batch_gen = batch.BatchGenerator(predictors, targets, raw, box_locs,
        primary, valid_frac=0.1, test_frac=0.1, batch_size=batch_size,
        timesteps=(6,12), random_seed=1234)

    gc.collect()

    return batch_gen


def select_sources(pred_names, sources=()):
    pred_names_flt = []

    if sources:
        source_list = {
            "rad": [
                "RZC", "CZC", "EZC-20", "EZC-45", "HZC", "LZC",
                "R10", "CPCH", "BZC", "AREA57"
            ],
            "polar": [
                "KDP","RHOHV","ZV-CORR","ZDR-CORR"
            ],
            "qualindex": ["obs-qual"]
        }

        var_list = []
        for source in sources:
            var_list.extend(source_list[source])

        for pred in pred_names:
            for source_var in var_list:
                if (pred == source_var) or pred.startswith(source_var+"-"):
                    pred_names_flt.append(pred)
                    break

    return pred_names_flt



def model_sources(sources_str, target="occurrence-8-10"):
    all_sources = ("rad", "polar","qualindex")
    sources = [s for s in all_sources if s[0] in sources_str]
    sources_str = "".join(s[0] for s in sources)

    batch_gen = setup_batch_gen("../data/2020/", target=target,
        batch_size=48, sources=sources)

    kwargs = {}
    compile_kwargs = {
        "opt_kwargs": {"weight_decay": 1e-4},
        "event_occurrence": 0.5
    }
    if target == "BZC":
        compile_kwargs["loss"] = "prob_binary_crossentropy"
    if target == "CPCH":        
        bins = np.array(
            [10, 30, 50],
            dtype=np.float32
        )
        compile_kwargs["loss"] = models.make_rain_loss_hist(bins)        
        compile_kwargs["metrics"] = []
        kwargs["last_only"] = True
        kwargs["num_outputs"] = len(bins)+1
        kwargs["final_activation"] = "softmax"

    (model,strategy) = models.init_model(
        batch_gen,
        dropout=0.1, 
        compile_kwargs=compile_kwargs,
        **kwargs
    )

    return (sources_str, batch_gen, model, strategy)

def training_sources(sources_str, target="occurrence-8-10", fn_prefix="lightning"):
    if sources_str in ("", "null"):
        sources_str = ""
        sources_suffix = "null"
    else:
        sources_suffix = sources_str
    (sources_str, batch_gen, model, strategy) = model_sources(
        sources_str, target=target)

    models.train_model(model, strategy, batch_gen,
        weight_fn=f"../models/{fn_prefix}/{fn_prefix}-{sources_suffix}.h5")


def eval_sources(sources_str, target="occurrence-8-10", fn_prefix="lightning",
    dataset="test", separate_leadtimes=False):
    if sources_str in ("", "null"):
        sources_str = ""
        sources_suffix = "null"
    else:
        sources_suffix = sources_str
    (sources_str, batch_gen, model, strategy) = model_sources(
        sources_str, target=target)
    
    weight_fn = os.path.join("../models/", fn_prefix, f"{fn_prefix}-{sources_suffix}.h5")
    model.load_weights(weight_fn)
    result_dir = os.path.join("../results/", fn_prefix, dataset)
    batch_seq = batch.BatchSequence(batch_gen, dataset=dataset)

    if not separate_leadtimes:
        eval_result = model.evaluate(batch_seq)        
        gc.collect()
        eval_fn = os.path.join(result_dir,
            f"eval-{fn_prefix}-{sources_str}.csv")
        if np.ndim(eval_result) == 0:
            eval_result = [eval_result]
        np.savetxt(eval_fn, eval_result, delimiter=',', fmt='%.6e')
    else:
        def loss_timestep(loss, timestep):
            def l(y_true, y_pred):
                y_true = y_true[:,timestep:timestep+1,...]
                y_pred = y_pred[:,timestep:timestep+1,...]
                return loss(y_true, y_pred)
            l.__name__ = f"loss_{timestep}"
            return l        
        metrics = [loss_timestep(model.loss, i) for i in range(12)]
        with strategy.scope():
            model.compile(loss=model.loss, metrics=metrics, optimizer='sgd')
        eval_result = model.evaluate(batch_seq)
        eval_fn = os.path.join(result_dir,
            f"eval_leadtime-{fn_prefix}-{sources_str}.csv")
        np.savetxt(eval_fn, eval_result, delimiter=',', fmt='%.6e')


def conf_matrix(
    sources_str, target="occurrence-8-10", fn_prefix="lightning",
    dataset="test"
):
    if sources_str in ("", "null"):
        sources_str = ""
        sources_suffix = "null"
    else:
        sources_suffix = sources_str
    (sources_str, batch_gen, model, strategy) = model_sources(
        sources_str, target=target)
    
    weight_fn = os.path.join("../models/", fn_prefix, f"{fn_prefix}-{sources_suffix}.h5")
    model.load_weights(weight_fn)
    result_dir = os.path.join("../results/", fn_prefix, dataset)
    
    thresholds = np.linspace(0, 1, 101)
    if fn_prefix == "rain":
        num_leadtimes = 1
        rain_thresh = [(1,10), (2,30), (3,50)]
        for rt in rain_thresh:
            conf_matrix = evaluation.conf_matrix_leadtimes(
                model, batch_gen, dataset=dataset,
                thresholds=thresholds, num_leadtimes=num_leadtimes,
                target=fn_prefix, rain_thresh=rt
            )
            cm_fn = os.path.join(result_dir,
                f"conf_matrix_leadtime-{fn_prefix}{rt[1]}-{sources_str}.npy")
            np.save(cm_fn, conf_matrix)
    else:
        num_leadtimes = 12
        conf_matrix = evaluation.conf_matrix_leadtimes(
            model, batch_gen, dataset=dataset,
            thresholds=thresholds, num_leadtimes=num_leadtimes,
            target=fn_prefix
        )
        cm_fn = os.path.join(result_dir,
            f"conf_matrix_leadtime-{fn_prefix}-{sources_str}.npy")
        np.save(cm_fn, conf_matrix)        

def get_FSS(sources_str, target="occurrence-8-10", fn_prefix="lightning",
    dataset="test"
):
    run="run1"
    if sources_str in ("", "null"):
        sources_str = ""
        sources_suffix = "null"
    else:
        sources_suffix = sources_str
    (sources_str, batch_gen, model, strategy) = model_sources(
        sources_str, target=target)
    
    weight_fn = os.path.join("../models/", fn_prefix, f"{fn_prefix}-{sources_suffix}.h5")
    model.load_weights(weight_fn)
    result_dir = os.path.join("../results/", fn_prefix, dataset)
    
    thresholds = np.linspace(0, 1, 101)

    if fn_prefix == "rain":
        num_leadtimes = 1
        rain_thresh = [(1,10),(2,30),(3,50)]
        for rt in rain_thresh:
            conf_matrix = evaluation.FSS(
                model, batch_gen, dataset=dataset,
                thresholds=thresholds, num_leadtimes=num_leadtimes,
                target=fn_prefix, rain_thresh=rt
            )
            cm_fn = os.path.join(result_dir,
                f"FSS-{fn_prefix}{rt[1]}-{sources_str}.npy")
            np.save(cm_fn, conf_matrix)
    else:
        num_leadtimes = 12
        conf_matrix = evaluation.FSS(
            model, batch_gen, dataset=dataset,
            thresholds=thresholds, num_leadtimes=num_leadtimes,
            target=fn_prefix
        )
        cm_fn = os.path.join(result_dir,
            f"FSS-{fn_prefix}-{sources_str}.npy")
        np.save(cm_fn, conf_matrix)   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('--sources', type=str)
    parser.add_argument('--target', type=str, default="occurrence-8-10")
    parser.add_argument('--prefix', type=str, default="lightning")
    parser.add_argument('--overwrite', type=bool, default=False)
    args = parser.parse_args()

    task = args.task
    if task == "train_sources":
        sources_str = args.sources
        target = args.target
        fn_prefix = args.prefix
        overwrite = args.overwrite
        model_exists = os.path.isfile(
            f"../models/{fn_prefix}/{fn_prefix}-{sources_str}.h5"
        )
        if model_exists and not overwrite:
            return
        training_sources(sources_str, target=target, fn_prefix=fn_prefix)
    elif task == "eval_sources":
        sources_str = args.sources
        target = args.target
        fn_prefix = args.prefix
        eval_sources(sources_str, target=target, fn_prefix=fn_prefix)
    elif task == "eval_sources_leadtime":
        sources_str = args.sources
        target = args.target
        fn_prefix = args.prefix
        eval_sources(sources_str, target=target, fn_prefix=fn_prefix,
            separate_leadtimes=True)
    elif "conf_matrix":
        sources_str = args.sources
        target = args.target
        fn_prefix = args.prefix
        conf_matrix(sources_str, target=target, fn_prefix=fn_prefix)
    elif task == "FSS":
        sources_str = args.sources
        target = args.target
        fn_prefix = args.prefix
        get_FSS(sources_str, target=target, fn_prefix=fn_prefix)


if __name__ == "__main__":
    main()
