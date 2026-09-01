import os
import glob
import json
import re
from transformers import AutoTokenizer
from dataclasses import asdict
from .data import QueryPrompt

class CONSTANTS:
    repoexec_benchmark = 'repoexec'
    codereval_python_benchmark = 'codereval-python'
    train_dataset = 'repost_train'
    token = 'token'
    
    repost_train_repos = ['AudioLDM-training-finetuning', 'auto-news', 'ComfyUI_essentials', 'CRM', 'EVE',
                        'finetune-anything', 'InfLLM', 'Music-Source-Separation-Training', 'OpenMusic', 'OpenVoice',
                        'pips2', 'PonderV2', 'pytorch-attention', 'rStar', 'stable-audio-tools',
                        'stable-fast', 'sync-notion', 'TensoIR', 'VectorDBBench', 'visualnav-transformer',
                        'so-vits-svc-4.0-v2', 'erpnext_china', 'magicoder', 'cace', 'cluster-health',
                        'SparseBEV', 'xtts-webui', 'BEVHeight', 'distil-whisper', 'pydantic-ai',
                        'npc_gzip', 'InfiniTransformer', 'arc-dsl', 'Scene-Diffuser', 'ShortGPT',
                        'Neural-Codec-and-Speech-Language-Models', 'mvsplat', 'selfcheckgpt', 'lighteval', 'VideoMAEv2',
                        'flowmap', 'IRRA', 'ViTMatte', 'all-in-one', 'bm25s', 'nougat', 'openai-forward', 'Gemini',
                        'baby-llama2-chinese', 'supabase-pydantic', 'nlf', 'MVDream', 'diffusionerf',
                        'Score-Entropy-Discrete-Diffusion', 'ChatGLM2-Voice-Cloning', 'tram', 'nerf2mesh', 'OpenFerro',
                        'UniTS', 'docling-core', 'inst-inpaint', 'Text-To-Video-Finetuning', 'LM4VisualEncoding',
                        'open-musiclm', 'wanda', 'ansible-dc-vxlan', 'FunCodec', 'unmasked_teacher',
                        'SOME', 'PythonFMU3', 'ConfLUNet', 'hamer', 'dolphin',
                        'python-audio-separator', 'StableTTS', 'BBDM', 'SparK', 'localrf',
                        'mlfz', 'jar3d_meta_expert', 'hitchhiking-rotations', 'S3IM-Neural-Fields', 'UniScene',
                        'BakedAvatar', 'pipelines', 'MeZO', 'blurry', 'Cutie',
                        'YOSO', 'alphaflow', 'dot', 'WinClip', 'Subject-Diffusion',
                        'UniversalFakeDetect', 'LLM2CLIP', 'functionary', 'WHAM', 'HexPlane',
                        'D3G', 'AnglE', 'bluffs', 'LightPHE']

    repoexec_repos = ['youtube-dl', 'python-semantic-release', 'sanic', 'pyMonet', 'pypara',
                      'fastapi', 'docstring_parser', 'httpie', 'tornado', 'cookiecutter',
                      'luigi', 'python-string-utils', 'flutes', 'py-backwards', 'PySnooper', 'thonny',
                      'pytutils', 'dataclasses-json', 'apimd', 'flutils', 'black', 'scrapy', 'typesystem']
    
    codereval_python_repos = [
        'rows', 'infoblox-client', 'atticmatic', 'lena', 'prestoplot',
        'o2sclpy', 'radiospectra', 'cloudmesh-common', 'neutron-lib', 'concert',
        'neo4j-python-driver', 'federation', 'shconfparser', 'pre-commit', 'cinder',
        'rdflib', 'py-seed', 'docopt-ng', 'boolean', 'infrared',
        'matplotlib', 'relman-auto-nag', 'ansible_collections', 'repoapi', 'os-zope',
        'Krake', 'swh-lister', 'planb', 'rdiffweb', 'pysolbase',
        'ocfl-py', 'borgmatic', 'makeprojects', 'gopad-python', 'flashbake',
        'apphelpers', 'python-sql-parameters', 'shortuuid', 'os-python-cachetools', 'os-python-dateutil',
        'lithium', 'eppy', 'packtools'
    ]

class FilePathBuilder:
    repoexec_benchmark_path = 'datasets/repoexec_target_function_prompts.jsonl'
    codereval_python_benchmark_path = 'datasets/codereval_python.jsonl'
    repoexec_repo_base_dir = 'repositories/repoexec/test-apps'
    codereval_python_repo_base_dir = 'repositories/codereval/python'
    repost_train_repo_base_dir = 'repositories/repost/train'
    repost_train_benchmark_path = 'datasets/repost_train_formatted.jsonl'
    result_save_dir = '/root/autodl-fs/test'
    position_weight_dir = 'weights'

    @staticmethod
    def make_needed_dir(file_path):
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    @staticmethod
    def get_repo_base_dir(benchmark_type: str = None):
        if benchmark_type is None:
            raise ValueError("benchmark_type is required")
        elif benchmark_type == CONSTANTS.repoexec_benchmark:
            return FilePathBuilder.repoexec_repo_base_dir
        elif benchmark_type == CONSTANTS.codereval_python_benchmark:
            return FilePathBuilder.codereval_python_repo_base_dir
        elif benchmark_type == CONSTANTS.train_dataset:
            return FilePathBuilder.repost_train_repo_base_dir
        else:
            raise ValueError(f"Invalid benchmark name: {benchmark_type}")

    @staticmethod
    def get_benchmark_path(benchmark_type: str = None):
        if benchmark_type is None:
            raise ValueError("benchmark_type is required")
        if benchmark_type == CONSTANTS.repoexec_benchmark:
            return FilePathBuilder.repoexec_benchmark_path
        elif benchmark_type == CONSTANTS.codereval_python_benchmark:
            return FilePathBuilder.codereval_python_benchmark_path
        elif benchmark_type == CONSTANTS.train_dataset:
            return FilePathBuilder.repost_train_benchmark_path
        else:
            raise ValueError(f"Invalid benchmark type: {benchmark_type}")

    @staticmethod
    def get_result_save_dir(benchmark_type: str,
                            task_type: str, 
                            model_name: str, 
                            sigma_ratio: float | None = None,
                            window_type: str | None = None):
        match task_type:
            case 'token':
                assert window_type == 'flexible'
                result_file_name = f'{benchmark_type}_{task_type}_{model_name}_sigma{sigma_ratio}.jsonl'
            case _:
                raise ValueError(f"Invalid task type: {task_type}")
        
        return os.path.join(FilePathBuilder.result_save_dir, result_file_name)
    
    @staticmethod
    def get_position_weight_path(sigma_ratio: float):
        return os.path.join(FilePathBuilder.position_weight_dir, f'weights_sigma_ratio_{sigma_ratio}.jsonl')

class DeepSeekTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/root/autodl-tmp/models/deepseek-coder-1.3b-base",
            revision="main",
            local_files_only=True
        )

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

        
class CodeLlamaTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/root/autodl-tmp/models/codellama-13b-hf",
            revision="main",
            local_files_only=True
        )

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


class Tools:
    @staticmethod
    def get_tokenizer(model_name: str):
        """Return the corresponding tokenizer based on the model name"""
        if 'deepseek-coder' in model_name:
            return DeepSeekTokenizer()
        elif 'codellama' in model_name:
            return CodeLlamaTokenizer()
        else:
            raise ValueError(f"Invalid model name: {model_name}")
    
    @staticmethod
    def get_input_dim(model_name: str):
        """Return the corresponding input_dim based on the model name"""
        if 'deepseek-coder-1.3b' in model_name:
            return 2048
        elif 'deepseek-coder-6.7b' in model_name:
            return 4096
        elif 'codellama-7b' in model_name:
            return 4096
        elif 'codellama-13b' in model_name:
            return 5120
        else:
            raise ValueError(f"Invalid model name: {model_name}, only support DS-1.3b, DS-6.7b, CL-7b, CL-13b")
    
    @staticmethod
    def get_function_body(prediction):
        """
        get function body from prediction

        Args:
            prediction: prediction string
            
        Returns:
            str: function body
        """
        if not prediction:
            return ""
        m = re.search(r'\n[^\s]', prediction)
        if not m:
            return prediction
        cut_index = m.start() + 1
        return prediction[:cut_index]
    
    @staticmethod
    def format_repoexec_predictions(prompt_results: list[QueryPrompt], output_file_path: str) -> None:
        """Format RepoExec predictions"""
        print(f"Loaded {len(prompt_results)} RepoExec predictions")
        output_data = []
        for prompt_result in prompt_results:
            prediction_data = prompt_result.metadata.prediction
            assert prediction_data is not None, "Prediction field is required"
            formatted_predictions = []
            for prediction in prediction_data:
                complete_function = prompt_result.metadata.target_function_prompt + Tools.get_function_body(prediction)
                formatted_predictions.append(complete_function)
            
            output_data.append(formatted_predictions)
            
        with open(output_file_path, 'w', encoding='utf8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
        print(f"Saved {len(output_data)} formatted RepoExec predictions to {output_file_path}")
    
    @staticmethod
    def format_codereval_python_predictions(prompt_results: list[QueryPrompt], output_file_path: str) -> None:
        """Format CoderEval Python predictions"""
        print(f"Loaded {len(prompt_results)} CoderEval Python predictions")
        
        output_data = []
        for prompt_result in prompt_results:
            prediction_data = prompt_result.metadata.prediction
            assert prediction_data is not None, "Prediction field is required"
            
            # Create complete function code for each prediction
            formatted_predictions = {}
            formatted_predictions["_id"] = prompt_result.metadata.task_id.split('/')[-1]
            function_predictions = []
            for prediction in prediction_data:
                complete_function = prompt_result.metadata.target_function_prompt + Tools.get_function_body(prediction)
                function_predictions.append(complete_function)
            formatted_predictions["generate_results"] = function_predictions
            
            output_data.append(formatted_predictions)

        with open(output_file_path, 'w', encoding='utf8') as f:
            for formatted_predictions in output_data:
                f.write(json.dumps(formatted_predictions) + '\n')
    
        print(f"Saved {len(output_data)} formatted CoderEval Python predictions to {output_file_path}")
    
    @staticmethod
    def get_repos(benchmark: str):
        """Return the corresponding repos list based on the benchmark"""
        if benchmark == CONSTANTS.repoexec_benchmark:
            return CONSTANTS.repoexec_repos
        elif benchmark == CONSTANTS.codereval_python_benchmark:
            return CONSTANTS.codereval_python_repos
        elif benchmark == CONSTANTS.train_dataset:
            return CONSTANTS.repost_train_repos
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}, only support repoexec, codereval-python, repost-train")
    
    @staticmethod
    def read_code(fname):
        with open(fname, 'r', encoding='utf8') as f:
            return f.read()
    
    @staticmethod
    def dump_dataclass(data_list, fname):
        data_dicts = [asdict(result) for result in data_list]
        Tools.dump_jsonl(data_dicts, fname)
    
    @staticmethod
    def dump_jsonl(data_list, fname):
        with open(fname, 'w', encoding='utf8') as f:
            for data in data_list:
                f.write(json.dumps(data) + '\n')
    
    @staticmethod
    def load_jsonl(fname):
        with open(fname, 'r', encoding='utf8') as f:
            return [json.loads(line) for line in f]
    
    @staticmethod
    def iterate_repository(base_dir, repo):
        pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.py")
        files = glob.glob(pattern, recursive=True)

        skipped_files = []
        loaded_code_files = dict()
        base_dir_list = os.path.normpath(base_dir).split(os.sep)
        for fname in files:
            try:
                code = Tools.read_code(fname)
                fpath_tuple = tuple(os.path.normpath(fname).split(os.sep)[len(base_dir_list):])
                loaded_code_files[fpath_tuple]= code
            except Exception as e:
                skipped_files.append((fname, e))
                continue

        if len(skipped_files) > 0:
            print(f"Skipped {len(skipped_files)} out of {len(files)} files due to I/O errors")
            for fname, e in skipped_files:
                print(f"{fname}: {e}")
        return loaded_code_files


