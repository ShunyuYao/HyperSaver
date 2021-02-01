# HyperSaver
Training deep learning models and recording the results are tedious. HyperSaver is a little tool to automatically save your hyperparameter settings and model permormance to a local csv or a FeiShu online document. 

# How To Use
## Prepration
Just copy the code file to your project and import the HyperSaver!   
## Create a template file
Create a csv or xlsx format template file to tell the HyperSaver which parameter you want to monitor, see the [hyperSaverTemplate.xlsx]() for more details.    
#### Notice: The `ID` is necesarry for the template because it is the only identification for a training log.
## Set up the HyperSaver class
Here is the sample code:
```python
from HyperSaver import HyperSaver
hyperSaver = HyperSaver(
        init_template='./hyperSaverTemplate.xlsx', set_id_by_time=True, webhook_url='')

hyperSaver.get_config_from_class(args)
# Get model performance
model_perf = {
    'loss': 0.1,
    'accuracy': 98,
}
hyperSaver.save_config('./results.csv')
```
Please see the complete code in `HyperSaver.py`.

A detailed instruction will be published to [here]().

# TO DO