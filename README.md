# YOLO CoreML Performance Benchmark

This project benchmarks the performance improvements of YOLO models when converted to CoreML format on Apple Silicon.

## Overview

The benchmark compares inference times between original PyTorch YOLO models and their CoreML counterparts across different model sizes (nano, small, medium, large, extra-large).

## Results

### Apple M1 MacBook Air (8GB RAM)

![Benchmark Results](benchmark_results_m1.png)

#### Performance Data

| Model Size | Original (s) | CoreML (s) | Speed Improvement |
|------------|--------------|------------|-------------------|
| n (nano)   | 0.0529       | 0.0162     | 3.27x faster      |
| s (small)  | 0.0980       | 0.0166     | 5.90x faster      |
| m (medium) | 0.1707       | 0.0245     | 6.97x faster      |
| l (large)  | 0.2178       | 0.0471     | 4.63x faster      |
| x (xlarge) | 0.4083       | 0.0468     | 8.73x faster      |

**M1 Key Findings:**
- **Average speed improvement**: 5.9x faster inference with CoreML
- **Best performance**: X-large model shows 8.73x improvement
- **Consistent gains**: All model sizes show significant performance improvements

### Apple M3 Ultra

![Benchmark Results](benchmark_results_m3_ultra.png)

#### Performance Data

| Model Size | Original (s) | CoreML (s) | Speed Improvement |
|------------|--------------|------------|-------------------|
| n (nano)   | 0.0263       | 0.0118     | 2.23x faster      |
| s (small)  | 0.0400       | 0.0130     | 3.08x faster      |
| m (medium) | 0.0614       | 0.0197     | 3.12x faster      |
| l (large)  | 0.0810       | 0.0210     | 3.86x faster      |
| x (xlarge) | 0.1186       | 0.0330     | 3.59x faster      |

**M3 Ultra Key Findings:**
- **Average speed improvement**: 3.18x faster inference with CoreML
- **Best performance**: Large model shows 3.86x improvement
- **Raw performance**: M3 Ultra is significantly faster overall, with even the slowest CoreML inference (0.033s) being faster than the fastest M1 CoreML inference (0.0162s for nano model)

### Cross-Generation Comparison

| Metric | M1 | M3 Ultra | Improvement |
|--------|----|----------|-------------|
| Fastest CoreML inference | 0.0162s (n) | 0.0118s (n) | 1.37x faster |
| Slowest CoreML inference | 0.0468s (x) | 0.0330s (x) | 1.42x faster |
| Average speed improvement | 5.9x | 3.18x | - |
| Raw performance gain | baseline | ~1.5-2x faster overall |

## Key Findings

- **Significant performance gains**: Both M1 and M3 Ultra show substantial improvements with CoreML
- **M3 Ultra raw performance**: ~1.4x faster than M1 across all models, despite lower relative speedup percentages
- **Consistent gains**: All model sizes show significant performance improvements on both platforms
- **Efficiency scaling**: M1 shows higher relative improvements due to baseline performance being slower
- **Real-world impact**: M3 Ultra can run inference in as little as 0.0118s (nano model) to 0.0330s (xlarge model)

## Usage

Run the benchmark with:
```bash
uv sync
uv run main.py
```

This will:
1. Load YOLO models in different sizes
2. Convert them to CoreML format (if not already done)
3. Run inference benchmarks
4. Generate performance comparison chart
5. Save detailed results to `benchmark_results.txt`

## Requirements

- Python 3.8+
- Ultralytics YOLO
- matplotlib
- Apple Silicon Mac (for CoreML acceleration)

## Notes

- Benchmark uses 10 warm-up runs per model for consistent measurements
- Test image: `bus.jpg` (must be present in working directory)
- CoreML models are saved as `.mlpackage` directories