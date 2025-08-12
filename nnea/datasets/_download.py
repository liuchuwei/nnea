import requests
import os
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, use a simple progress bar
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
    Download dataset
    :param dataset: Dataset name
    :param output_dir: Directory to save the dataset
    :return: None
    """

    if not dataset:
        raise ValueError("dataset parameter cannot be empty")

    if not output_dir:
        raise ValueError("output_dir parameter cannot be empty")

    ## imm_melanoma
    if dataset == "imm_melanoma":
        nadata_url = "https://figshare.com/ndownloader/files/56852135"
        nadata_fl = os.path.join(output_dir, "imm_melanoma_exp.txt")

        if not os.path.exists(output_dir):
            logger.info(f"Creating output directory {output_dir}")
            os.mkdir(output_dir)

        if not os.path.exists(nadata_fl):
            logger.info("Downloading melanoma dataset...")
            request_fl_through_url(nadata_url, nadata_fl)
        else:
            logger.info("Melanoma dataset already exists, no need to download again.")



    ## imm_bladder
    elif dataset == "imm_bladder":
        nadata_url = "https://figshare.com/ndownloader/files/56852129"
        nadata_fl = os.path.join(output_dir, "imm_bladder_exp.txt")

        if not os.path.exists(output_dir):
            logger.info(f"Creating output directory {output_dir}")
            os.mkdir(output_dir)

        if not os.path.exists(nadata_fl):
            logger.info("Downloading bladder dataset...")
            request_fl_through_url(nadata_url, nadata_fl)
        else:
            logger.info("Bladder dataset already exists, no need to download again.")



    ## imm_ccRCC
    elif dataset == "imm_ccRCC":
        nadata_url = "https://figshare.com/ndownloader/files/56852132"
        nadata_fl = os.path.join(output_dir, "imm_ccRCC_exp.txt")

        if not os.path.exists(output_dir):
            logger.info(f"Creating output directory {output_dir}")
            os.mkdir(output_dir)

        if not os.path.exists(nadata_fl):
            logger.info("Downloading ccRCC dataset...")
            request_fl_through_url(nadata_url, nadata_fl)
        else:
            logger.info("ccRCC dataset already exists, no need to download again.")



    else:
        logger.error(f"Dataset not supported yet: {dataset}!")


def request_fl_through_url(url=None, output_file=None):
    """
    Download file through URL, supports progress bar and resume download
    :param url: File download link
    :param output_file: Output file path
    :return: None
    """

    headers = {
        "Referer": "https://figshare.com/articles/dataset/nnea/29635898",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    }

    # Resume download initialization
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

        # Get total file size (considering resume scenario)
        if resume_header and response.status_code == 206:  # Partial content
            content_range = response.headers.get('Content-Range', '')
            total_size = int(content_range.split('/')[-1]) if content_range else None
        else:
            total_size = int(response.headers.get('content-length', 0)) or None

        # Initialize progress bar
        progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            initial=initial_bytes,
            desc=f"Downloading {os.path.basename(output_file)}",
            colour='green'
        )

        # File write mode (append for resume)
        mode = "ab" if resume_header and response.status_code == 206 else "wb"

        with open(output_file, mode) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if initial_bytes > 0:  # Discard first incomplete chunk when resuming
                    chunk = chunk[initial_bytes % 8192:]
                    initial_bytes = 0  # Only need to process once
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))  # Update progress bar

        progress_bar.close()  # Ensure progress bar is closed
        logger.info(f"File saved to: {output_file}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")

    except Exception as e:
        logger.error(f"Unknown error occurred: {e}")
        if 'progress_bar' in locals():
            progress_bar.close()
