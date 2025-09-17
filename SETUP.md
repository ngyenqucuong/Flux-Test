wget "https://civitai.com/api/download/models/2031577?type=Model&format=SafeTensor&size=pruned&fp=fp8&token=0379cb908883e50ea081d3583d20552a" \
  --content-disposition \
  -P models/checkpoints/

wget "https://civitai.com/api/download/models/2031577?type=VAE&format=SafeTensor&token=0379cb908883e50ea081d3583d20552a" \
  --content-disposition \
  -P models/vae/