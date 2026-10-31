# Temporal CNN reproduction

The original CNN run is not a reliable paper comparison: the legacy cache
contains synthetic landmarks generated from labels, the saved run used 40
epochs, and its model contains batch normalization not described in the paper.
The original files are left intact.

## Separate first-pass experiment

1. `python scripts/prepare_baseline_landmarks.py --workers 4` copies legacy
   landmarks with no synthetic confidence fingerprint and re-extracts flagged
   files from the corresponding real videos into `data/landmarks_baseline/`.
   The manifest labels copied files `legacy_unverified`: lacking original
   extraction metadata, those files do **not** have verified provenance.
   Use `--all` to extract every file from video for stronger provenance; this
   is slower. The manifest includes SHA-256 hashes for generated artifacts.
2. `python scripts/run_paper_cnn.py --epochs 100` trains in an isolated results
   directory and saves the scaler, manifest and split hashes, checkpoint,
   training history, validation and test metrics. Outputs do not overwrite the
   old run. A second run must specify a different `--output` path.
   Use a Python environment with CUDA-enabled PyTorch: the script requires CUDA
   and fails rather than falling back to CPU. CPU is still used for data loading
   and preprocessing, with PyTorch CPU threads limited to two.

The CNN uses the architecture described in the paper's prose: two 64-channel
convolutions (kernel 3), ReLU, dropout 0.2, adaptive average pooling, and a
64-unit classifier head. The paper's table instead says max pooling; use
`--pooling max` as a separately reported sensitivity experiment. The code uses
unweighted cross entropy, Adam at 0.001 without scheduler or weight decay,
batch size 256, and validation macro F1 checkpoint selection. These are
explicit assumptions where the paper is silent. It uses the paper's 48-frame,
stride-one windows and final-frame labels. The original (pre-imputation) hand
presence is used to filter windows, an interpretation of the paper's detected
hand requirement; imputation and wrist normalization precede a scaler fit on
training frames. Exact video split and MediaPipe settings were not specified in
the paper, so this experiment is not an exact independent reproduction.
The alternative `--presence-filter imputed` filters using the post-interpolation
presence flags and should always use a separate `--output` directory. This is
a sensitivity analysis, since the interpretation is not specified clearly.

Both scripts fail when required input is missing; they never manufacture
synthetic landmarks. The cache preparation preserves the legacy originals.

The completed GPU sensitivity run (`presence-filter imputed`) reached test
accuracy 69.14% and macro F1 69.08% at the validation-selected epoch 70.
It uses 47 newly extracted and 165 legacy-unverified landmark files and was
run after observing the first experiment's test result, so it is exploratory.
The recorded script hash refers to the version used during the run; subsequent
formatting and output-directory safeguards change the committed script hash.
