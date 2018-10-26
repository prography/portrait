Portrait! 
---
This application transfers style of images.

We support three style: watercolor, oilcolor, vector illustration



Usage
---

#### if you want to transfer image to watercolor painting style

```
python generate.py --convert_mode water --input_path path/to/input/image --output_path path/to/save/output/image
```

#### if you want to transfer image to oilcolor painting style

```
python generate.py --convert_mode oil --input_path path/to/input/image --output_path path/to/save/output/image
```

#### if you want to transfer image to vector illustration style

```
python generate.py --convert_mode vector --input_path path/to/input/image --output_path path/to/save/output/image
```

Examples
---
#### original to watercolor style
![](https://github.com/prography/portrait/blob/deep_dev/StyleTransfer-pytorch/imgs/input_004.jpg)
<img src="https://github.com/prography/portrait/blob/deep_dev/StyleTransfer-pytorch/imgs/water_output_004.png" width="178" height="218">


#### original to oilcolor style   
![](https://github.com/prography/portrait/blob/deep_dev/StyleTransfer-pytorch/imgs/input_004.jpg)
<img src="https://github.com/prography/portrait/blob/deep_dev/StyleTransfer-pytorch/imgs/oil_output_004.png" width="178" height="218">
