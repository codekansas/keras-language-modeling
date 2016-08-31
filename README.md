# keras-language-modeling

Some code for doing language modeling with Keras, in particular for question-answering tasks. I wrote a very long blog post that explains how a lot of this works, which can be found [here](http://benjaminbolte.com/blog/2016/keras-language-modeling.html).

### Stuff that might be of interest

 - `attention_lstm.py`: Attentional LSTM, based on one of the papers referenced in the blog post and others. One application used it for [image captioning](http://arxiv.org/pdf/1502.03044.pdf). It is initialized with an attention vector which provides the attention component for the neural network.
 - `insurance_qa_eval.py`: Evaluation framework for the InsuranceQA dataset. To get this working, clone the [data repository](https://github.com/codekansas/insurance_qa_python) and set the `INSURANCE_QA` environment variable to the cloned repository. Changing `config` will adjust how the model is trained.
 - `keras-language-model.py`: The `LanguageModel` class uses the `config` settings to generate a training model and a testing model. The model can be trained by passing a question vector, a ground truth answer vector, and a bad answer vector to `fit`. Then `predict` calculates the similarity between a question and answer. Override the `build` method with whatever language model you want to get a trainable model. Examples are provided at the bottom, including the `EmbeddingModel`, `ConvolutionModel`, and `RecurrentModel`.

### Getting Started

````bash
# Install Keras (may also need dependencies)
git clone https://github.com/fchollet/keras
cd keras
sudo python setup.py install

# Clone InsuranceQA dataset
git clone https://github.com/codekansas/insurance_qa_python
export INSURANCE_QA=$(pwd)/insurance_qa_python

# Run insurance_qa_eval.py
git clone https://github.com/codekansas/keras-language-modeling
cd keras-language-modeling/
python insurance_qa_eval.py
````

Alternatively, I wrote a script to get started on a Google Cloud Platform instance (Ubuntu 16.04) which can be run via

````bash
cd ~
git clone https://github.com/codekansas/keras-language-modeling
cd keras-language-modeling
source install.py
````

I've been working on making these models available out-of-the-box. You need to install the Git branch of Keras (and maybe make some modifications) in order to run some of these models; the Keras project can be found [here](https://github.com/fchollet/keras).

The runnable program is `insurance_qa_eval.py`. This will create a `models/` directory which will store a history of the model's weights as it is created. You need to set an environment variable to tell it where the INSURANCE_QA dataset is.

Finally, my setup (which I think is pretty common) is to have an SSD with my operating system, and an HDD with larger data files. So I would recommend creating a `models/` symlink from the project directory to somewhere in your HDD, if you have a similar setup.

### Serving to a port

I added a command line argument that uses Flask to serve to a port. Once you've [installed Flask](http://flask.pocoo.org/docs/0.11/installation/), you can run:

````bash
python insurance_qa_eval.py serve
````

This is useful in combination with [ngrok](https://ngrok.com/) for monitoring training progress away from your desktop.

### Additionally

 - The official implementation can be found [here](https://github.com/white127/insuranceQA-cnn-lstm)

### Data

 - L6 from [Yahoo Webscope](http://webscope.sandbox.yahoo.com/)
 - [InsuranceQA data](https://github.com/shuzi/insuranceQA)
   - [Pythonic version](https://github.com/codekansas/insurance_qa_python)

