from ultralytics import YOLO
from pathlib import Path
from time import perf_counter as time
import matplotlib.pyplot as plt

benchmark = {}
for size in ["n", "s", "m", "l", "x"]:
    pt_path = Path(f"yolo26{size}.pt")
    coreml_model_path = Path(f"yolo26{size}.mlpackage")
    coreml_model = YOLO(coreml_model_path, task='detect')
    # Load the YOLO26 model
    model = YOLO(pt_path)
    # First inference to warm up
    model("bus.jpg")

    # Export the model to CoreML format
    if not coreml_model_path.exists():
        model.export(format="coreml")  # creates 'yolo26{size}.mlpackage'
    # Load the exported CoreML model

    original_model_st = time()
    for _ in range(10):  # Warm-up runs
        # Run inference
        results_origin = model("bus.jpg")
    original_model_et = time()
    avg_time_original = (original_model_et - original_model_st) / 10
    print(f"Average inference time for original model yolo26{size}: {avg_time_original:.2f} seconds")
    
    del model  # free up memory for fair benchmarking

    # First inference to warm up
    coreml_model("bus.jpg")
    coreml_model_st = time()
    for _ in range(10):  # Warm-up runs
        results = coreml_model("bus.jpg")
    coreml_model_et = time()
    avg_time_coreml = (coreml_model_et - coreml_model_st) / 10
    print(f"Average inference time for CoreML model yolo26{size}: {avg_time_coreml:.2f} seconds")


    benchmark[size] = {
        "original_inference_time": avg_time_original,
        "coreml_inference_time": avg_time_coreml,
    }
    
    del coreml_model  # free up memory for next iteration

# Plotting the benchmark results
sizes = list(benchmark.keys())
original_times = [benchmark[size]["original_inference_time"] for size in sizes]
coreml_times = [benchmark[size]["coreml_inference_time"] for size in sizes]
x = range(len(sizes))
plt.bar(x, original_times, width=0.4, label='YOLO.pt', align='center')
plt.bar([i + 0.4 for i in x], coreml_times, width=0.4, label='YOLO.mlpackage', align='center')
plt.xlabel('Model Size')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time Comparison between Original and CoreML Models')
plt.xticks([i + 0.2 for i in x], sizes)
plt.legend()
plt.show()
