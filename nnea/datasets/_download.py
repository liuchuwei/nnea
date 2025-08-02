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
    下载数据集
    :param dataset: 数据集名称
    :param output_dir: 保存数据集的目录
    :return: None
    """

    if not dataset:
        raise ValueError("dataset参数不能为空")

    if not output_dir:
        raise ValueError("output_dir参数不能为空")

    ## imm_melanoma
    if dataset == "imm_melanoma":
        nadata_url = "https://figshare.com/ndownloader/files/56852135"
        nadata_fl = os.path.join(output_dir, "imm_melanoma_exp.txt")

        if not os.path.exists(output_dir):
            logger.info(f"正在创建输出目录 {output_dir}")
            os.mkdir(output_dir)

        if not os.path.exists(nadata_fl):
            logger.info("正在下载melanoma数据集...")
            request_fl_through_url(nadata_url, nadata_fl)
        else:
            logger.info("melanoma数据集已存在，无需重复下载。")



    ## imm_bladder
    elif dataset == "imm_bladder":
        nadata_url = "https://figshare.com/ndownloader/files/56852129"
        nadata_fl = os.path.join(output_dir, "imm_bladder_exp.txt")

        if not os.path.exists(output_dir):
            logger.info(f"正在创建输出目录 {output_dir}")
            os.mkdir(output_dir)

        if not os.path.exists(nadata_fl):
            logger.info("正在下载bladder数据集...")
            request_fl_through_url(nadata_url, nadata_fl)
        else:
            logger.info("bladder数据集已存在，无需重复下载。")



    ## imm_ccRCC
    elif dataset == "imm_ccRCC":
        nadata_url = "https://figshare.com/ndownloader/files/56852132"
        nadata_fl = os.path.join(output_dir, "imm_ccRCC_exp.txt")

        if not os.path.exists(output_dir):
            logger.info(f"正在创建输出目录 {output_dir}")
            os.mkdir(output_dir)

        if not os.path.exists(nadata_fl):
            logger.info("正在下载ccRCC数据集...")
            request_fl_through_url(nadata_url, nadata_fl)
        else:
            logger.info("ccRCC数据集已存在，无需重复下载。")



    else:
        logger.error(f"暂不支持的数据集: {dataset}！")


def request_fl_through_url(url=None, output_file=None):
    """
    通过url下载文件，支持进度条和断点续传
    :param url: 文件下载链接
    :param output_file: 输出文件路径
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
            timeout=30,
        )
        response.raise_for_status()

        # 获取文件总大小（考虑续传场景）
        if resume_header and response.status_code == 206:  # 部分内容
            content_range = response.headers.get('Content-Range', '')
            total_size = int(content_range.split('/')[-1]) if content_range else None
        else:
            total_size = int(response.headers.get('content-length', 0)) or None

        # 初始化进度条
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
        logger.info(f"文件已保存至: {output_file}")

    except requests.exceptions.RequestException as e:
        logger.error(f"下载失败: {e}")

    except Exception as e:
        logger.error(f"发生未知错误: {e}")
        if 'progress_bar' in locals():
            progress_bar.close()
