### Clone the repository
```shell
git clone git@github.com:liuchuwei/nnea.git
```

### Load Immunotherapy Dataset

```shell
mkdir datsaets
mkdir factory
mkdir factory/tumor_immunotherapy
cd factory/tumor_immunotherapy
wget --referer="https://figshare.com/" \
     --user-agent="Mozilla/5.0" \
     -c "https://figshare.com/ndownloader/files/56334518" -O exp.txt
wget --referer="https://figshare.com/" \
     --user-agent="Mozilla/5.0" \
     -c "https://figshare.com/ndownloader/files/56334512" -O phe.txt 
python run.py dataload --config/generate_dataset.toml
```