[TensorFlow] Protein Interface Prediction using Graph Convolutional Networks
=====

## Result

### Training
<div align="center">
  <p>
    <img src="./figures/loss.svg" width="800">
  </p>
  <p>Loss graph.</p>
</div>

### Test
<div align="center">
  <img src="./figures/0.png" width="350">
  <img src="./figures/1.png" width="350"></br>
  <img src="./figures/2.png" width="350">  
  <img src="./figures/3.png" width="350"></br>
  <img src="./figures/4.png" width="350">
  <img src="./figures/5.png" width="350"></br>  
  <img src="./figures/6.png" width="350">
  <img src="./figures/7.png" width="350"></br>  
  <img src="./figures/8.png" width="350">
  <img src="./figures/9.png" width="350"></br>  
  <p>Each figure shows input digit, attention map, and overlapped image sequentially.</p>
</div>

### Further usage
<div align="center">
  <img src="./figures/f0.png" width="800"></br>
  <img src="./figures/f1.png" width="800"></br>
  <img src="./figures/f2.png" width="800"></br>
  <p>The further usages. Detecting the location of digits can be conducted using an attention map.</p>
</div>

## Requirements
* TensorFlow 2.3.0  
* Numpy 1.18.5

## Additional Resources
[1] <a href="https://github.com/kjm1559/simple_attention">Simple attention mechanism test</a> by Myung Jin Kim
