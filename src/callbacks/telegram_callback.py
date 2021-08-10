import tensorflow as tf
import requests

class TelegramCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_interval=15, info_file='callbacks/credentials.txt'):
        super().__init__()
        self.epoch_interval = epoch_interval
        self.bot_token, self.chat_ids = self._get_chat_info(info_file)
        self.url = 'https://api.telegram.org/bot' + self.bot_token  \
                +  '/sendMessage'

    def _get_chat_info(self, filename):
        with open(filename, 'r') as info_file:
            lines = info_file.read().splitlines()
        return lines[0], lines[1:]
    
    def on_train_begin(self, logs):
        msg = '<pre>' + 33*'=' + f'\nTraining started for model {self.model.name}\n' + 33*'=' + '</pre>'
        for chat_id in self.chat_ids:
            requests.post(url=self.url, 
                data={'chat_id': chat_id, 'text':msg, 'parse_mode': 'HTML'})

    def on_epoch_end(self, epoch, logs):
        if epoch % self.epoch_interval == 0:
            msg = '<pre>' + f'Report at epoch {epoch + 1}:'
            msg += '\n' + 33 * '-' + "\n|{:^19s}|{:^11s}|".format('Metric', 'Value')
            msg += '\n' + 33 * '-'
            for key in logs:
                msg += "\n|{:^19s}|{:^11.4g}|".format(key, logs[key])
            print(msg)
            for chat_id in self.chat_ids:
                requests.post(url=self.url,
                    data={'chat_id': chat_id, 'text': msg + '</pre>', 'parse_mode': 'HTML'})
    
    def on_train_end(self, logs):
        msg = '<pre>' + 33*'=' + f"\nTraining ended for model {self.model.name}; total number of epochs: {len(logs['loss'])}\n" + 33*'=' + '</pre>'
        for chat_id in self.chat_ids:
            requests.post(url=self.url, 
                data={'chat_id': chat_id, 'text':msg, 'parse_mode': 'HTML'})
        
       