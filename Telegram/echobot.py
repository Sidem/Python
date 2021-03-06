#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Simple Bot to reply to Telegram messages. This is built on the API wrapper, see
# echobot2.py to see the same example built on the telegram.ext bot framework.
# This program is dedicated to the public domain under the CC0 license.
import logging
import telegram
from telegram.error import NetworkError, Unauthorized
from time import sleep


update_id = None

def main():
    global update_id
    # Telegram Bot Authorization Token
    bot = telegram.Bot('400297899:AAFWYq4SHK6Sk6RM9ROoN9PPUodTbYDuo5g')

    # get the first pending update_id, this is so we can skip over it in case
    # we get an "Unauthorized" exception.
    try:
        update_id = bot.get_updates()[0].update_id
    except IndexError:
        update_id = None

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('Test')
    fh = logging.FileHandler('spam.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)  
    while True:
        try:
            echo(bot, ch)
        except NetworkError:
            sleep(1)
        except Unauthorized:
            # The user has removed or blocked the bot.
            update_id += 1


def echo(bot, logger):
    global update_id
    # Request updates after the last update_id
    for update in bot.get_updates(offset=update_id, timeout=10):
        update_id = update.update_id + 1

        if update.message:  # your bot can receive updates without messages
            # Reply to the message
            update.message.reply_text(update.message.text)
            logger.info(update.message.text)


if __name__ == '__main__':
    main()
