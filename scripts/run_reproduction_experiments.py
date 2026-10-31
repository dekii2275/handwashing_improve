import subprocess
import sys
import json
import time
from pathlib import Path
import pandas as pd

def run_experiment(name: str, script: str, config_path: str, output_dir: str):
    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENT: {name}")
    print(f"Script: {script}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-u", script,
        "--config", config_path,
        "--data-dir", "data/processed_w48_s4",
        "--output-dir", output_dir,
        "--groups", "shape", "motion",
        "--evaluate-test",
        "--resume", "auto",
    ]
    
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    for line in proc.stdout:
        print(line, end="", flush=True)
        
    proc.wait()
    elapsed = time.time() - t0
    
    if proc.returncode != 0:
        print(f"ERROR: {name} failed with exit code {proc.returncode}")
        return None
        
    print(f"\n{name} COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # Read metrics
    test_metrics_path = out_path / "test_metrics.json"
    val_metrics_path = out_path / "validation_metrics.json"
    result_path = out_path / "result.json"
    
    res = {"name": name, "elapsed_s": elapsed}
    if test_metrics_path.exists():
        with open(test_metrics_path, "r", encoding="utf-8") as f:
            res["test"] = json.load(f)
    if val_metrics_path.exists():
        with open(val_metrics_path, "r", encoding="utf-8") as f:
            res["val"] = json.load(f)
    if result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            res["result"] = json.load(f)
            
    return res

def main():
    experiments = [
        ("GRU_shape_motion", "scratch/friend_repo/train.py", "configs/train_shape_motion.yaml", "models/reproduced_gru_shape_motion"),
        ("LSTM_shape_motion", "scratch/friend_repo/train_lstm.py", "configs/train_shape_motion.yaml", "models/reproduced_lstm_shape_motion"),
    ]
    
    all_results = []
    for name, script, cfg, out in experiments:
        res = run_experiment(name, script, cfg, out)
        if res:
            all_results.append(res)
            
    print("\n" + "#"*70)
    print("ALL EXPERIMENTS SUMMARY:")
    print("#"*70)
    
    summary_rows = []
    for r in all_results:
        test = r.get("test", {})
        val = r.get("val", {})
        res_info = r.get("result", {})
        summary_rows.append({
            "Experiment": r["name"],
            "Best Epoch": res_info.get("best_epoch", "-"),
            "Val Macro-F1 (%)": f"{val.get('macro_f1', 0)*100:.2f}%",
            "Val Accuracy (%)": f"{val.get('accuracy', 0)*100:.2f}%",
            "Test Accuracy (%)": f"{test.get('accuracy', 0)*100:.2f}%",
            "Test Macro-F1 (%)": f"{test.get('macro_f1', 0)*100:.2f}%",
            "Test Balanced Acc (%)": f"{test.get('balanced_accuracy', 0)*100:.2f}%",
        })
        
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    
    # Save summary
    df_summary.to_csv("models/reproduction_summary.csv", index=False)
    print("\nSaved summary to models/reproduction_summary.csv")

if __name__ == "__main__":
    main()
