# trainer.py

from transformers import Trainer
import torch 

# Define a custom trainer
class SentenceTrainer(Trainer):
    def compute_loss(self, model, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['label']
        tasks = inputs['task']
        
        outputs = model(input_ids, attention_mask, tasks) 
        return torch.nn.CrossEntropyLoss()(outputs, labels)
    
    def prediction_step(self, model, inputs, prediction_loss_only, both_tasks=False, ignore_keys=None):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        if both_tasks == True:
            tasks = ['class','sent']
        else:
            tasks = inputs['task'] 
        
        # We do not need to calculate gradients for validation
        with torch.no_grad():
            logits = []
            for i, task in enumerate(tasks):
                task_output = model(input_ids=input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0), task=task)
                logits.append(task_output)

        logits = torch.cat(logits)

        if prediction_loss_only:
            return (None, logits, None)

        # labels = torch.argmax(inputs['label'], dim=1)
        labels = inputs['label']
        return (None, logits, labels)

