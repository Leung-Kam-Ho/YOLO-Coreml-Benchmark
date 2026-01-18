from ultralytics import YOLO
from pathlib import Path
from time import perf_counter as time
import matplotlib.pyplot as plt

benchmark = {}
for size in ["n", "s", "m", "l", "x"]:
    pt_path = Path(f"yolo26{size}.pt")
    model = YOLO(pt_path)
    
    image_url = "https://ultralytics.com/images/bus.jpg"
    coreml_model_path = Path(f"yolo26{size}.mlpackage")
    if not coreml_model_path.exists():
        model.export(format="coreml")  # creates 'yolo26{size}.mlpackage'
    # Load the YOLO26 model
    # First inference to warm up
    model(image_url)

    # Export the model to CoreML format
    # Load the exported CoreML model

    original_model_st = time()
    for _ in range(10):  # Warm-up runs
        # Run inference
        results_origin = model(image_url)
    original_model_et = time()
    avg_time_original = (original_model_et - original_model_st) / 10
    print(f"Average inference time for original model yolo26{size}: {avg_time_original:.2f} seconds")
    
    del model  # free up memory for fair benchmarking

    # First inference to warm up
    coreml_model = YOLO(coreml_model_path, task='detect')
    coreml_model(image_url)
    coreml_model_st = time()
    for _ in range(10):  # Warm-up runs
        results = coreml_model(image_url)
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
plt.savefig("benchmark_results.png")
# plt.show()

# save benchmark results to a text file
with open("benchmark_results.txt", "w") as f:
    for size in benchmark:
        f.write(f"Model Size: {size}\n")
        f.write(f"Original Inference Time: {benchmark[size]['original_inference_time']:.4f} seconds\n")
        f.write(f"CoreML Inference Time: {benchmark[size]['coreml_inference_time']:.4f} seconds\n\n")
    print("Benchmark results saved to benchmark_results.txt")
    del results  # free up memory for next iteration