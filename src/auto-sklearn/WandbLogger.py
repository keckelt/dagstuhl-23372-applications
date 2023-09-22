import logging
import wandb
import re


isFirst=True
runInitialized=False
    
# Define a custom logging handler that sends logs to WandB
class WandbHandler(logging.Handler):
    
    def __init__(self )-> None:
        logging.Handler.__init__(self=self)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ create logger')
        
        
    def emit(self, record):
        # Format the log message
        log_message = self.format(record)
        with open("application.txt",'a',encoding = 'utf-8') as f:
            f.write(f"{log_message} \n")
            
        return
        
        global isFirst
        global runInitialized 

        # Start a run when a config is evaluated
        # e.g.,
        # 870 2023-09-14 12:21:14,601 - Client-TAE - INFO - Starting to evaluate configuration 9 
        # start a new wandb run to track this script
        
        # tmp file tmp_folder: /tmp/auto-sklearn_tmp_58ef5f20-52ff-11ee-a009-0242ac110002
        config = {
          "train_accuracy": -1,
          "test_accuracy": -1,
        }
        
        if 'Starting to evaluate configuration' in log_message:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ wandb.init')
            wandb.init(
                # set the wandb project where this run will be logged
                project="automl",
                entity='dagstuhl-23372',
                group='tmnt12',
                config=config
            )
            isFirst=True
            runInitialized=True
    

        # Check if a match was found
        if runInitialized is True:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ wandb.log')
            wandb.log({"custom_log": log_message})
            # Define a regular expression pattern to match the accuracy value
            pattern = r"accuracy=([0-9.]+)"

            # Use re.search to find the first occurrence of the pattern in the string
            match = re.search(pattern, log_message)
            
            if match:
                # Extract the accuracy value as a float
                accuracy = float(match.group(1))
                
                print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ acurracy {accuracy}')
                if isFirst:
                    wandb.log({"train_accuracy": accuracy})
                    wandb.config.update({"train_accuracy": accuracy})
                    isFirst = False
                else:
                    wandb.config.update({"test_accuracy": accuracy})
                    wandb.log({"test_accuracy": accuracy})
        
        
        # Finish the run when the config is done
        # e.g.,
        # 909 2023-09-14 12:21:19,988 - Client-TAE - INFO - Finished evaluating configuration 9
        if 'Finished evaluating configuration' in log_message and runInitialized is True:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ wandb.finish')
            wandb.finish()
            runInitialized=False
        
