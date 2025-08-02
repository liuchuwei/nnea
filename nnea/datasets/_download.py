import requests
import os
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，使用简单的进度条
    def tqdm(iterable=None, **kwargs):
        if iterable is None:
            return DummyTqdm(**kwargs)
        return iterable
    
    class DummyTqdm:
        def __init__(self, **kwargs):
            self.total = kwargs.get('total', 0)
            self.unit = kwargs.get('unit', 'B')
            self.unit_scale = kwargs.get('unit_scale', True)
            self.unit_divisor = kwargs.get('unit_divisor', 1024)
            self.initial = kwargs.get('initial', 0)
            self.desc = kwargs.get('desc', 'Downloading')
            self.colour = kwargs.get('colour', 'green')
            self.current = self.initial
        
        def update(self, n):
            self.current += n
        
        def close(self):
            pass

try:
    from ..logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def download(dataset: str, output_dir: str) -> None:

    """
    Download datasets
    :param dataset: name of the dataset
    :param output_dir: directory to save the dataset
    :return: None
    """

    if not dataset:
        raise ValueError("dataset is required")

    if not output_dir:
        raise ValueError("output_dir is required")

    ## imm_melanoma
    if dataset == "imm_melanoma":

        exp_url = "https://figshare.com/ndownloader/files/56528579"
        phe_url = "https://figshare.com/ndownloader/files/56528576"
        exp_fl = os.path.join(output_dir, "imm_melanoma_exp.txt")
        phe_fl = os.path.join(output_dir, "imm_melanoma_phe.txt")

        if not os.path.exists(output_dir) & os.path.exists(exp_fl) & os.path.exists(phe_fl):
            logger.info("Downloading melanoma immunotherapy dataset...")
        else:
            logger.info("Datset already downloaded...")

        if not os.path.exists(output_dir):

            logger.info(f"Creating output directory {output_dir}")
            os.mkdir(output_dir)

        if not os.path.exists(exp_fl):
            request_fl_through_url(exp_url, exp_fl)

        if not os.path.exists(phe_fl):
            request_fl_through_url(phe_url, phe_fl)

    ## imm_bladder
    elif dataset == "imm_bladder":

        logger.info("Downloading bladder immunotherapy dataset...")
        exp_url = "https://figshare.com/ndownloader/files/56528570"
        phe_url = "https://figshare.com/ndownloader/files/56540711"
        exp_fl = os.path.join(output_dir, "imm_bladder_exp.txt")
        phe_fl = os.path.join(output_dir, "imm_bladder_phe.txt")

        if not os.path.exists(output_dir):
            logger.info(f"Creating output directory {output_dir}")
            os.mkdir(output_dir)

        if not os.path.exists(exp_fl):
            request_fl_through_url(exp_url, exp_fl)

        if not os.path.exists(phe_fl):
            request_fl_through_url(phe_url, phe_fl)

    ## imm_ccRCC
    elif dataset == "imm_ccRCC":

        logger.info("Downloading ccRCC immunotherapy dataset...")
        exp_url = "https://figshare.com/ndownloader/files/56528567"
        phe_url = "https://figshare.com/ndownloader/files/56528573"

        exp_fl = os.path.join(output_dir, "imm_ccRCC_exp.txt")
        phe_fl = os.path.join(output_dir, "imm_ccRCC_phe.txt")

        if not os.path.exists(output_dir):
            logger.info(f"Creating output directory {output_dir}")
            os.mkdir(output_dir)

        if not os.path.exists(exp_fl):
            request_fl_through_url(exp_url, exp_fl)

        if not os.path.exists(phe_fl):
            request_fl_through_url(phe_url, phe_fl)

    else:
        logger.error(f"Dataset {dataset} not supported！")


def request_fl_through_url(url=None, output_file=None):
    """
    Download file through url with progress bar and resume support
    :param url: file's url to request
    :param output_file: path of output file
    :return: None
    """

    headers = {
        "Referer": "https://figshare.com/articles/dataset/nnea/29635898",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    }

    # 断点续传初始化
    initial_bytes = 0
    if os.path.exists(output_file):
        initial_bytes = os.path.getsize(output_file)
        resume_header = {"Range": f"bytes={initial_bytes}-"}
    else:
        resume_header = {}

    try:
        response = requests.get(
            url,
            headers={**headers, **resume_header},
            stream=True,
            timeout=30,# 添加代理参数
        )
        response.raise_for_status()

        # 获取文件总大小（考虑续传场景）
        if resume_header and response.status_code == 206:  # 部分内容
            content_range = response.headers.get('Content-Range', '')
            total_size = int(content_range.split('/')[-1]) if content_range else None
        else:
            total_size = int(response.headers.get('content-length', 0)) or None

        # 初始化进度条（核心添加部分）
        progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            initial=initial_bytes,
            desc=f"Downloading {os.path.basename(output_file)}",
            colour='green'
        )

        # 文件写入模式（续传时追加）
        mode = "ab" if resume_header and response.status_code == 206 else "wb"


        with open(output_file, mode) as f:

            for chunk in response.iter_content(chunk_size=8192):
                if initial_bytes > 0:  # 续传时丢弃第一个不完整分块
                    chunk = chunk[initial_bytes % 8192:]
                    initial_bytes = 0  # 仅需处理一次
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))  # 更新进度条

        progress_bar.close()  # 确保关闭进度条
        logger.info(f"File saved to: {output_file}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if 'progress_bar' in locals():
            progress_bar.close()
