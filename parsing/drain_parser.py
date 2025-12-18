import json
import logging
import time

from drain3 import TemplateMiner
from drain3.template_miner import TemplateMinerConfig
import os

# Configure logging (optional)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Configuration for the TemplateMiner
# Use default configuration, but you can customize parameters like sim_th, max_children, etc.
config = TemplateMinerConfig()
config_file = os.path.join("drain3.ini")
config.load(config_file)
config.profiling_enabled = True

# You can load configuration from a file (e.g., drain3.ini) if you have complex masking rules
# config.load('drain3.ini')

template_miner = TemplateMiner(config=config)

# Specify your log file path
log_file_path = os.path.join("..", "data","raw", "final_auth_modern.log")

logger.info(f"--- Starting training phase on {log_file_path} ---")

# Open and read the log file
line_count = 0
parsed_data = []
with open(log_file_path, 'r') as f:
    lines = f.readlines()

    start_time = time.time()
    batch_start_time = start_time
    batch_size = 10000

    line_again = []

    for line in lines:
        line = line.rstrip()
        timestamp = line.partition(": ")[0]
        line = line.partition(": ")[2]
        result = template_miner.add_log_message(line)
        parameters = template_miner.extract_parameters(result["template_mined"], line, True)
        line_count += 1
        # if line_count % batch_size == 0:
        #     time_took = time.time() - batch_start_time
        #     rate = batch_size / time_took
        #     # logger.info(f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
        #     #             f"{len(template_miner.drain.clusters)} clusters so far.")
        #     batch_start_time = time.time()
        if result["change_type"] != "none":
            result_json = json.dumps(result)
            logger.info(f"Input ({line_count}): {line}")
            logger.info(f"Result: {result_json}")

        parsed_line = {
            "timestamp": timestamp,
            "template_id": result["cluster_id"],
            "template_text": result["template_mined"],
            "parameters": parameters
            }
        parsed_data.append(parsed_line)
    time_took = time.time() - start_time
    rate = line_count / time_took

    logger.info(
        f"--- Done processing file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
        f"{len(template_miner.drain.clusters)} clusters")

    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
    for cluster in sorted_clusters:
        logger.info(cluster)

    print("Prefix Tree:")
    template_miner.drain.print_tree()

    template_miner.profiler.report(0)


    output_file = os.path.join("..", "data", "parsed", "parsed_data.json")
    with open(output_file, "w") as fw:
        json.dump(parsed_data, fw, indent=4) # type: ignore