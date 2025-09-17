wget "https://civitai.com/api/download/models/2031577?type=Model&format=SafeTensor&size=pruned&fp=fp8" \
  --content-disposition \
  -P models/checkpoints/

# Download VAE
wget "https://civitai.com/api/download/models/2031577?type=VAE&format=SafeTensor" \
  --content-disposition \
  -P models/vae/