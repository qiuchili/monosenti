
# Multimodal Sentiment Analysis and Emotion Recognition

Contributor: Qiuchi Li

## Instructions to run the code

### Download the datasets

+ Dropbox Link: https://www.dropbox.com/s/7z56hf9szw4f8m8/cmumosi_cmumosei_iemocap.zip?dl=0
  + Containing CMUMOSI, CMUMOSEI and IEMOCAP datasets
  + Each dataset has the CMU-SDK version and Multimodal-Transformer version (with different input dimensionalities)

### Do a single run (train/valid/test) 

1. Set up the configurations in config/run.ini
2. python run.py -config config/run.ini

#### Configuration setup
  + **mode = run**
  + **dataset_type = multimodal**
  + **pickle_dir_path = /path/to/datasets/**. The absolute path of the folder storing the datasets.
  + **dataset_name in `{'cmumosei','cmumosi','iemocap'}`**. Name of the dataset.
  + **features in `{'acoustic', 'visual', 'textual'}`**. Multiple modality names should be joined by ','. 
  + **label in `{'sentiment','emotion'}`** ('sentiment' for CMUMOSI and CMUMOSEI, 'emotion for IEMOCAP).
  + **wordvec_path**. The relative path of the pre-trained word embedding file.
  + **embedding_trainable in `{'True','False'}`**. Whether you want to train the word embedding for textual modality. Usually set to be True.
  + **seed**. The random seed for the experiment.
  + **load_model_from_dir in `{'True','False'}`**. Whether the model is loaded from a saved file.
    + **dir_name**. The directory storing the model configurations and model parameters. Requires **load_model_from_dir = True**.
  + **fine_tune in `{'True','False'}**. Whether you want to train the model with the data. 
  + **model-specific parameters**. For running a model on the dataset, uncomment the respective area of the model and comment the areas for the other models. Please refer to the model implementations in /models/monologue/ for the meaning of each model specific parameter.
    + supported models include but are not limited to:
      + RAVEN 
      + Multimodal-Transformer 
      + EF-LSTM
      + LF-LSTM
      + TFN
      + MFN
      + MARN
      + Graph-MFN (Not available for the moment)
      + RMFN
      + MARN
      + QMF (Local-Mixture & QDNN)
      + Vanilla-LSTHM
  
### Grid Search for the best hyperparameters
1. Set up the configurations in config/grid_search.ini. Tweak a couple of fields in the single run configurations, as instructed below.
2. Write up the hyperparameter pool in config/grid_parameters/.
3. python run.py -config config/grid_search.ini

#### Configuration setup
+ **mode = run_grid_search**
+ **grid_parameters_file**. The name of file storing the parameters to be searched, under the folder /config/grid_parameters. 
  + the format of a file is:
    + [COMMON]
    + var_1 = val_1;val_2;val_3
    + var_2 = val_1;val_2;val_3
+ **search_times**. The number of times the program searches in the pool of parameters.
+ **output_file**.  The file storing the performances for each search in the pool of parameters. By default, it is eval/grid_search_`{dataset_name}`_`{network_type}`.csv






