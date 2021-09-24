import tensorflow as tf
import random
import requests

EMOJIS = ['&#128052', '&#128017', '&#128042', '&#128024', '&#128036', '&#128039', 
        '&#129417', '&#128011', '&#128044', '&#128009', '&#128025', '&#129419',
        '&#129409', '&#128030', '&#128042', '&#129413', '&#129414', '&#129416',
        '&#129408', '&#128032', '&#128033', '&#128375', '&#128010', '&#128330',
        '&#128060', '&#129423', '&#128023', '&#129412', '&#128041', '&#129421']

class TelegramCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_interval=10, info_file='callbacks/credentials.txt'):
        super().__init__()
        self.epoch_interval = epoch_interval
        self.bot_token, self.chat_ids = self._get_chat_info(info_file)
        self.url = 'https://api.telegram.org/bot' + self.bot_token  \
                +  '/sendMessage'
        self.emoji = random.choice(EMOJIS)

    def _get_chat_info(self, filename):
        with open(filename, 'r') as info_file:
            lines = info_file.read().splitlines()
        return lines[0], lines[1:]
    
    def on_train_begin(self, logs):
        msg = '<pre>' + 33*'=' + f'\nTraining started for model {self.model.name} {self.emoji}\n' + 33*'=' + '</pre>'
        for chat_id in self.chat_ids:
            requests.post(url=self.url, 
                data={'chat_id': chat_id, 'text':msg, 'parse_mode': 'HTML'})

    def on_epoch_end(self, epoch, logs):
        print(logs)
        if epoch % self.epoch_interval == 0:
            msg = '<pre>' + f'Report at epoch {epoch} {self.emoji}:'
            msg += '\n' + 33 * '-' + "\n|{:^19s}|{:^11s}|".format('Metric', 'Value')
            msg += '\n' + 33 * '-'
            for key in logs:
                print(key)
                msg += "\n|{:^19s}|{:^11.4g}|".format(key, logs[key])
            for chat_id in self.chat_ids:
                requests.post(url=self.url,
                    data={'chat_id': chat_id, 'text': msg + '</pre>', 'parse_mode': 'HTML'})
    
    def on_train_end(self, logs):
        msg = '<pre>' + 33*'=' + f"\nTraining ended for model {self.model.name} {self.emoji}.\n"
        msg += '\n' + 33 * '-' + "\n|{:^19s}|{:^11s}|".format('Metric', 'Value')
        msg += '\n' + 33 * '-'
        for key in logs:
            msg += "\n|{:^19s}|{:^11.4g}|".format(key, logs[key])
        msg += '\n' + 33 * '-' + '\n\n' + 33*'=' + '</pre>'
        for chat_id in self.chat_ids:
            requests.post(url=self.url, 
                data={'chat_id': chat_id, 'text':msg, 'parse_mode': 'HTML'})
        
       