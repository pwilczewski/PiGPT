## Memorizing digits of Pi
In _MemorizePi.py_ I estimate a decoder transformer that memorizes sequences of Pi. When the number of model parameters >> sequence length the model is able to memorize digits to arbitrary precision. After training the model for 400 epochs I'm able to exactly reproduce the first 1,280 digits of Pi.

## Quantizing a model of Pi
In _QuantizePi - cpu.py_ I estimate an MLP model that memorizes sequences of Pi. Then I quantize the model from fp32 to int8 with post-training dynamic quantization using PyTorch's prototype FX graph model quantization. Then I confirmed that the quantized model could still exactly predict the first 1,280 digits of Pi. After quantization the model state dictionary file _pi_mlp_int8.pth_ was 63% smaller on disc than the pre-quantized file _pi_mlp_f32.pth_

## Testing if Pi is a normal number
In the _NormalPi - Memory.py_ file I estimate a decoder transformer to see if I can find any evidence that Pi is not a normal number. A number is normal in base 10 if for every positive integer n, all possible strings n digits long have density 1/n^10. Using 1 million digits of Pi, a context length of 4 and a model with 12,800 parameters I train the model for 400 epochs. Using this model I test if given 4 digits the 5th digit can be predicted. The trained model accuracy on a test sample of 100,000 digits is 0.0993 using greedy selection or slightly worse than random chance.

Finally using the trained model I predicted the fifth digit for all sequences between 0000 and 9999. If I use greedy selection to predict the fifth digit, the distribution of predictions is quite uneven. The most commonly predicted digit (3) is predicted three times more frequently as the least commonly predicted digit (0).

![fifth_digit](https://user-images.githubusercontent.com/9024799/236638205-bc230dbd-42db-49b4-bb01-d853738848e7.png)

## Data
For this analysis I used the digits of Pi from: https://pi2e.ch/blog/2017/03/10/pi-digits-download/#download

My transformer code is adapted from Andrej Karpath's colab: https://colab.research.google.com/drive/1SiF0KZJp75rUeetKOWqpsA8clmHP6jMg?usp=sharing
