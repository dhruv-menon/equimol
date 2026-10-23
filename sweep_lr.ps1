# LR sweep for the irrep-egnn l_max ladder on rMD17 aspirin, split 1.
# Runs 150 epochs at each LR for one rung, at C = 32 channels per l.
# Usage (from the repo root): powershell -ExecutionPolicy Bypass -File .\sweep_lr.ps1 -Rung l3

param([Parameter(Mandatory)][ValidateSet("l0", "l1", "l2", "l3")][string]$Rung)

$rungs = @{
  l0 = @("32x0e", "0e")
  l1 = @("32x0e + 32x1o", "0e + 1o")
  l2 = @("32x0e + 32x1o + 32x2e", "0e + 1o + 2e")
  l3 = @("32x0e + 32x1o + 32x2e + 32x3o", "0e + 1o + 2e + 3o")
}
New-Item -ItemType Directory -Force logs | Out-Null

foreach ($lr in "3e-4", "1e-3", "3e-3") {
  $name = "sweep_irrep_${Rung}_c32_lr$lr"
  uv run --no-sync python examples/trainers/md17_regression.py `
    --dataset rmd17 --datapath data/rmd17 --molecule aspirin --splits 1 `
    --model irrep-egnn --graph fully_connected --batch-size 5 --lr $lr `
    --irreps-hidden $rungs[$Rung][0] --irreps-edge $rungs[$Rung][1] `
    --epochs 150 --eval-every 1 --lambda-energy 1 --lambda-force 10 `
    --lr-patience 25 --min-lr 1e-6 --ema-decay 0.99 --num-workers 0 `
    --checkpoint-path "checkpoints/rmd17/$name.pt" `
    | Tee-Object -FilePath "logs/$name.log"
}
