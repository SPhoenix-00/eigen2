# Disk Space Requirements for Eigen2 Training

## Critical Issue: Replay Buffer Size

The replay buffer is the largest consumer of disk space during training. Understanding its growth is essential for proper cloud GPU configuration.

---

## Buffer Size Calculation

### Per Transition
Each transition in the buffer contains:
- **State**: (504, 108, 9) float32 = ~1.87 MB
- **Action**: (108, 2) float32 = ~0.001 MB
- **Reward**: 1 float32 = ~0.000004 MB
- **Next State**: (504, 108, 9) float32 = ~1.87 MB
- **Done**: 1 bool = ~0.000001 MB

**Total per transition: ~3.74 MB**

### Buffer Growth During Training

| Generation | Transitions Added | Total Transitions | Uncompressed Size | Compressed Size* |
|------------|-------------------|-------------------|-------------------|------------------|
| 1 | 2,000 | 2,000 | 7.5 GB | 2.5 GB |
| 5 | 2,000 | 10,000 | 37.4 GB | 12-15 GB |
| 10 | 2,000 | 20,000 | 74.8 GB | 25-30 GB |
| 15 | 2,000 | 30,000 | 112.2 GB | 37-45 GB |
| 20 | 2,000 | 40,000 | 149.6 GB | 50-60 GB |
| 25 | 2,000 | 50,000 | **187.0 GB** | **62-75 GB** |

*Compression achieved with gzip level 6 (estimates 60-70% reduction)

**Note:** Buffer capacity is 200,000 transitions but 25 generations only reach 50,000.

---

## Total Disk Space Breakdown

For a complete 25-generation training run:

| Component | Size | Notes |
|-----------|------|-------|
| **Operating System** | 10-15 GB | Ubuntu/container base |
| **PyTorch + CUDA** | 5-8 GB | Deep learning framework |
| **Python Dependencies** | 1-2 GB | numpy, pandas, etc. |
| **Training Data CSV** | 0.5 GB | Eigen2_Master_PY_OUTPUT.pkl |
| **Replay Buffer (compressed)** | 62-75 GB | At generation 25 |
| **Population Checkpoints** | 0.25 GB | 16 agents × ~15 MB each |
| **Best Agent Checkpoint** | 0.015 GB | Single model |
| **Trainer State** | 0.001 GB | JSON metadata |
| **Logs** | 0.1 GB | Training logs |
| **Temporary Files** | 2-5 GB | pip cache, etc. |
| **Safety Margin** | 20-30 GB | For temporary operations |
| | | |
| **TOTAL REQUIRED** | **~100-135 GB** | |
| **RECOMMENDED** | **250 GB** | Comfortable headroom |

---

## Compression Implementation

### Automatic Compression (Enabled by Default)

The replay buffer automatically uses gzip compression when saving:

```python
# In models/replay_buffer.py
buffer.save("replay_buffer.pkl", compress=True)  # Default
```

### Compression Benefits

- **Space Savings**: 60-70% reduction in disk usage
- **Upload Savings**: Smaller files = faster GCS uploads
- **Cost Savings**: Less cloud storage needed
- **Performance**: Minimal CPU overhead (gzip level 6)

### Decompression

Loading automatically detects compressed buffers:

```python
buffer = ReplayBuffer.load("replay_buffer.pkl")
# Auto-detects gzip by magic bytes (0x1f8b)
```

---

## Cloud GPU Provider Recommendations

### RunPod
- **Minimum Disk**: 250 GB
- **Recommended**: 250-300 GB
- **Cost Impact**: ~$0.10-0.15/hr additional for larger disk

### Vast.ai
- **Minimum Disk**: 250 GB
- **Recommended**: 300 GB
- **Cost Impact**: Varies by provider

### Lambda Labs
- **Minimum Disk**: 512 GB (standard offering)
- **No extra cost** - comes with instance

---

## What Happens if Disk is Too Small?

### At Generation 5 (~15 GB buffer)
- First buffer save
- If disk < 30 GB free: **Training will fail**
- Error: "No space left on device"

### At Generation 10 (~30 GB buffer)
- Second buffer save
- If disk < 50 GB free: **Training will fail**

### At Generation 25 (~75 GB buffer)
- Final buffer save
- If disk < 100 GB free: **Training will fail**

**Recovery**: Training can resume from last successful checkpoint, but you'll lose progress since then.

---

## Monitoring Disk Usage During Training

### Check Disk Space
```bash
# On RunPod/Vast.ai instance
df -h /workspace

# Watch in real-time
watch -n 60 df -h /workspace
```

### Expected Usage by Generation
```bash
# Generation 1: ~25 GB
# Generation 5: ~40 GB
# Generation 10: ~60 GB
# Generation 15: ~75 GB
# Generation 20: ~90 GB
# Generation 25: ~100 GB
```

### Alert if Space Low
```bash
# Add to training script
available=$(df /workspace | tail -1 | awk '{print $4}')
if [ $available -lt 20000000 ]; then
    echo "WARNING: Less than 20GB free!"
fi
```

---

## Optimization Options

### Option 1: Delete Old Buffers (Not Recommended)
Only keep latest buffer, delete previous saves:
- Saves disk space
- Loses ability to resume from earlier generations
- **Not implemented** - buffers already overwrite

### Option 2: Increase Compression (Available)
Use higher gzip level for more compression:
```python
# In replay_buffer.py, line 139
gzip.open(file_path + '.gz', 'wb', compresslevel=9)  # Max compression
```
- Saves additional 5-10%
- Increases save time by 30-50%
- Minimal benefit vs. cost

### Option 3: Reduce Buffer Capacity (Impacts Quality)
Lower `BUFFER_SIZE` in config.py:
```python
BUFFER_SIZE = 100000  # Instead of 200000
```
- Halves disk usage
- **May reduce training quality**
- Not recommended without testing

### Option 4: Stream to Cloud (Complex)
Save directly to GCS without local copy:
- Eliminates local disk usage
- Requires download on resume
- **Not currently implemented**

---

## Summary

**For 25-generation training runs:**
- Minimum disk: **150 GB** (tight, risky)
- Recommended disk: **250 GB** (safe)
- Comfortable disk: **500 GB** (plenty of room)

**With compression enabled (default):**
- Replay buffer: 62-75 GB
- System + checkpoints: 20-30 GB
- Safety margin: 20-30 GB
- **Total: ~100-135 GB used**

**Update your RunPod instance to 250+ GB disk before starting training!**
