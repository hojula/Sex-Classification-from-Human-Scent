import subprocess as sp
import os
import datetime
import yaml
from omegaconf import OmegaConf


class Partitions:
    gpufast = 'gpufast'
    gpu = 'gpu'
    gpulong = 'gpulong'
    gpuextralong = 'gpuextralong'
    gpudeadline = 'gpudeadline'
    amdgpufast = 'amdgpufast'
    amdgpu = 'amdgpu'
    amdgpulong = 'amdgpulong'
    amdgpuextralong = 'amdgpuextralong'
    amdgpudeadline = 'amdgpudeadline'

    max_times = {'gpufast': '0-04:00:00',
                 'gpu': '0-12:00:00',
                 'gpulong': '3-00:00:00',
                 'gpuextralong': '21-00:00:00',
                 'gpudeadline': '1-00:00:00',
                 'amdgpufast': '0-04:00:00',
                 'amdgpu': '0-8:00:00',
                 'amdgpulong': '3-00:00:00',
                 'amdgpuextralong': '21-00:00:00',
                 'amdgpudeadline': '1-00:00:00',
                 'qgpu': '1-00:00:00',
                 'qgpu_free': '18:00:00',
                 'standard-g': '12:00:00'
                 }

    num_workers = {'gpufast': 12,
                   'gpu': 12,
                   'gpulong': 12,
                   'gpuextralong': 12,
                   'gpudeadline': 12,
                   'amdgpufast': 1,
                   'amdgpu': 12,
                   'amdgpulong': 1,
                   'amdgpuextralong': 1,
                   'amdgpudeadline': 1,
                   }

    @staticmethod
    def to_amd(queue_name):
        return 'amd' + queue_name


def main():
    src_style_job_conf_path = os.path.join(os.getcwd(), 'config.yaml')
    config = OmegaConf.load(src_style_job_conf_path)
    workload_manager = 'sbatch'

    master_port = 9014

    node_count = 1
    gpus_per_node = 1

    cpus_per_task = gpus_per_node * 16
    mem_per_cpu = "2GB"

    script_dir = config.script_dir
    data_dir = config.root_dir
    experiments_dir = os.path.join(data_dir, 'experiments')
    cluster = 'rci'

    logs_dir = config.logs_dir

    confs_dir = os.path.join(script_dir, "confs")

    cluster_job_params = {"&NODES": node_count,
                          "&GPUS_PER_NODE": gpus_per_node,
                          "&LOGSDIR": logs_dir,
                          "&CPU_PER_TASK": cpus_per_task,
                          "&MEM_PER_CPU": mem_per_cpu
                          }

    partition = Partitions.gpu
    cluster_job_params.update({
        "&PARTITION": partition,
        "&TIME": Partitions.max_times[partition]
    })

    cluster_job_params.update({
        "&SCRIPT_PATH": os.path.join(script_dir, 'train_network_classification.py'),
    })
    batch_size = config.batch_size
    cluster_job_params.update({
        "&BATCH_SIZE": batch_size,
    })

    config.batch_size = batch_size
    config.experiments_dir = experiments_dir
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    datetime_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f'))
    identification_str = datetime_str + '_rest_png_' + str(batch_size)
    trn_job_conf_path = os.path.join(confs_dir, identification_str + '.yml')
    os.makedirs(os.path.dirname(trn_job_conf_path), exist_ok=True)
    with open(trn_job_conf_path, 'w') as file:
        OmegaConf.save(config, file)

    cluster_job_params.update({
        "&MASTER_PORT": master_port,
        "&LOGDIRNAME": identification_str,
    })

    script_args = [trn_job_conf_path, config.root_dir, config.data_dir, '/data/temporary/', config.save_dir,
                   config.txt_dir, config.master_mask_filepath, config.plots_dir]

    errs, trn_job_id = submit_job(workload_manager, cluster_job_params, script_dir, script_args, cluster)
    print(f'Submitted job_id: {trn_job_id}, errs: {errs}')

    master_port += 1


def prepare_sbatch_file(sbatch_source_filepath, sbatch_target_filepath, params):
    lines_to_write = []
    with open(sbatch_source_filepath, 'r') as f:
        for line in f.readlines():
            for key, value in params.items():
                line = line.replace(key, str(value))
            lines_to_write.append(line)

    with open(sbatch_target_filepath, 'w') as f:
        f.writelines(lines_to_write)


def submit_job(workload_manager, cluster_job_params, script_dir, script_args, cluster):
    workload_manager_source_filepath = os.path.join(script_dir, f'{cluster}_job.sh')
    workload_manager_filepath = os.path.join(script_dir, f'{cluster}_job.sh')
    prepare_sbatch_file(workload_manager_source_filepath, workload_manager_filepath,
                        cluster_job_params)
    print("Workload manager file path: ", workload_manager_filepath)
    p = sp.Popen([workload_manager, workload_manager_filepath] + script_args, stdout=sp.PIPE)
    outs, errs = p.communicate()
    job_id, errs = -1, ""
    if outs is not None:
        outs = outs.decode("utf8").strip()
        job_id = outs.split(' ')[-1]
    return errs, job_id


if __name__ == '__main__':
    main()
