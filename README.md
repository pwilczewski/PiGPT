# PiGPT

## Memorizing digits of Pi
In MemorizePi.py I estimate a decoder transformer that memorizes sequences of Pi. When the number of model parameters >> sequence length the model is able to memorize digits to arbitrary precision. After training the model for 400 epochs I'm able to exactly reproduce the first 1,280 digits of Pi.

## Testing if Pi is a normal number
In the NormalPi.py files I estimate a decoder transformer to see if I can find any evidence that Pi is not a normal number. A number is normal in base 10 if for every positive integer n, all possible strings n digits long have density 1/n^{10}. Using 1 million digits of Pi, a context length of 4 and a model with 12,800 parameters I train the model for 400 epochs. Using this model I test if given 4 digits the 5th digit can be predicted. 

Finally I test to see the distribution of predictions for all possible context lengths.

## Data
For this analysis I used the digits of Pi from: https://pi2e.ch/blog/2017/03/10/pi-digits-download/#download

My transformer code is adapted from Andrej Karpath's colab: https://colab.research.google.com/drive/1SiF0KZJp75rUeetKOWqpsA8clmHP6jMg?usp=sharing
