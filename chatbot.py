#--- imports begin ---#
from telebot import TeleBot
from dotenv import dotenv_values

from google_trans_new import google_translator

from tensorflow.keras.models import load_model
import numpy as np
import pickle
import json
import random

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from model.translationModel import get_prediction, translate

import re

import database as db
#--- imports end ---#


#--- initialize begin ---#
lemmatizer = WordNetLemmatizer()

translator = google_translator()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot.h5')
#--- initialize end ---#

#--- global variables begin ---#
finaltag = ''
mode = -1
CID = 0
password = ''
accountNo = ''
vendor = 0
index = 0
PIN = 0
loggedIn = False
flagMessage = False
storeFinalTag = ''
botReply = ''
toAccountNo = ''
amount = 0
visited = False
noOfVendors = 0
noOfMonthlySubscription = 0
#--- global variables end ---#


#--- telegram bot begin ---#
API_KEY = dotenv_values('.env')['API_KEY']
bot = TeleBot('nchatbot')
bot.config['api_key'] = API_KEY
#--- telegram bot end ---#


#--- chatbot functions begin ---#
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    result = ''
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            global finaltag
            result = random.choice(i['responses'])
            finaltag = i['tag']
            break
    return result


def end() -> None:
    global finaltag, mode, CID, password, accountNo, vendor, index, PIN, loggedIn, flagMessage, storeFinalTag, botReply, toAccountNo, amount, visited, noOfVendors, noOfMonthlySubscription
    finaltag = ''
    mode = -1
    CID = 0
    password = ''
    accountNo = ''
    vendor = 0
    index = 0
    PIN = 0
    loggedIn = False
    flagMessage = False
    storeFinalTag = ''
    botReply = ''
    toAccountNo = ''
    amount = 0
    visited = False
    noOfVendors = 0
    noOfMonthlySubscription = 0
#--- chatbot functions end ---#


#--- responses begin ---#
def humanagent():
    return('\nWe are sorry our automated bot could not help you out.\nFeel free to contact or leave a feedback through any of the following channels:\nWEB: www.abfc.com\nMOB:9867512345\nEMAIL:abfc@financials.help\nThank You for your patience\nIs there anything else we can do for you?')


def enquire():
    return('\nYou may choose these frequent topics or may write your own query:\nAccounts\nVendors\nMonthly Subscription\nAccount Statements\nTransfer Money\nDeposits\nLoans')


def depositintro():
    return('\nThese are important deposit account:\nSaving Account\nFixed Deposits\nRecurring Deposit\nWe have raised a query, you will be personally contacted for more details.')


def loanintro():
    return('\nThese are main types of loan :\Personal Loan\nHome Loan\nCar Loan\nEducation Loan\nCredit Card Loan\nWe have raised a query, you will be personally contacted for more details.')
#--- responses end ---#


#--- driver code begin ---#
def getReply(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res
#--- driver code end ---#


#--- bot implementation begin ---#


#--- route all messages ---#
@bot.route('')
def any(message):
    global finaltag, mode, CID, password, accountNo, vendor, index, PIN, loggedIn, flagMessage, storeFinalTag, botReply, toAccountNo, amount, visited, noOfVendors, noOfMonthlySubscription

    userMessage = message['text'].lower()
    userID = message['chat']['id']

    botReply = ''

    if mode != -1:
        botReply = getReply(userMessage)

    #--- check login status ---#
    if CID != 0 and accountNo != '':
        loggedIn = True

    #--- /start ---#
    if userMessage == '/start':
        botReply = 'Hey '+message['from']['first_name']+'!\n' + \
            'Welcome to ABFC Financials Ltd. I am Ava, How may I help you?'
        mode = 0

    #--- /login ---#
    elif userMessage == '/login':
        if not loggedIn:
            botReply = ''
            loggedIn = True
            mode = 1
        else:
            botReply = 'You are already logged in\n\nSend /end to end the session'

    #--- /logout ---#
    elif userMessage == '/logout':
        if not loggedIn:
            botReply = 'You are not logged in.\n\nSend /login to login.'
        else:
            end()
            botReply = 'Logged Out Successfully!\n\nSend /end to end session'
            mode = -1

    #--- /end - logout with message ---#
    elif userMessage == '/end' or finaltag == 'goodbye':
        end()
        botReply = '\n\nThank you for choosing ABFC Financials Ltd.\nIt was my pleasure to help you.\n(Quitting....)'
        mode = -1

    if mode == 0:
        flagMessage = False
        visited = False

        #--- talk to operator ---#
        if finaltag == 'handoff':
            botReply += humanagent()

        #--- options ---#
        elif finaltag == 'enquire':
            botReply += enquire()

        #--- deposits ---#
        elif finaltag == 'deposits':
            botReply += depositintro()

        #--- loans ---#
        elif finaltag == 'loans1':
            botReply += loanintro()

        #--- check balance ---#
        elif finaltag == 'fetch_balance':
            storeFinalTag = finaltag
            if not loggedIn:
                mode = 1
            else:
                mode = 3

        #--- merchant payment ---#
        elif finaltag == 'checkOut_vendor':
            storeFinalTag = finaltag
            if not loggedIn:
                mode = 1
            else:
                mode = 3

        #--- monthly subscription ---#
        elif finaltag == 'checkOut_monthlySubscription':
            storeFinalTag = finaltag
            if not loggedIn:
                mode = 1
            else:
                mode = 3

        #--- show transaction ---#
        elif finaltag == 'show_transaction':
            storeFinalTag = finaltag
            if not loggedIn:
                mode = 1
            else:
                mode = 3

        #--- account to account transfer ---#
        elif finaltag == 'acc_to_acc':
            storeFinalTag = finaltag
            if not loggedIn:
                mode = 1
            else:
                mode = 12

    if mode == 1:
        #--- ask for CID ---#
        if not flagMessage:
            botReply += '\n8-digit Customer ID'
            flagMessage = True

        else:
            #--- fetch CID ---#
            if db.findCID(int(userMessage)):
                CID = int(userMessage)
                mode = 2
                flagMessage = False
            else:
                botReply = 'Invalid CID!\nTry again.'
                flagMessage = True

    if mode == 2:
        #--- ask for account number ---#
        if not flagMessage:
            botReply = 'Account Number'
            flagMessage = True

        else:
            #--- fetch account number ---#
            if db.findAcc(CID, userMessage):
                accountNo = userMessage
                mode = 3
                flagMessage = False
            else:
                botReply = 'CID-Account Number mismatch\nTry again.'
                flagMessage = True

    if mode == 3:
        #--- ask for password ---#
        if not flagMessage:
            botReply = 'Password'
            flagMessage = True

        else:
            #--- fetch password ---#
            if db.verifyPass(CID, accountNo, userMessage):
                botReply = 'Verified✔'
                botReply += '\n\nIs there anything else I can help you with?'
                mode = 0
                password = userMessage
                flagMessage = False
                if storeFinalTag == 'fetch_balance':
                    mode = 4
                elif storeFinalTag == 'checkOut_vendor':
                    mode = 5
                elif storeFinalTag == 'checkOut_monthlySubscription':
                    mode = 10
                elif storeFinalTag == 'show_transaction':
                    mode = 11
                elif storeFinalTag == 'acc_to_acc':
                    if not visited:
                        visited = True
                        mode = 12
                    else:
                        db.acc_to_acc(CID, password, accountNo,
                                      toAccountNo, amount)
                        botReply = 'Transaction Successful!'
                        botReply += '\n\nIs there anything else I can help you with?'
                        mode = 0
            else:
                botReply = 'Wrong Password :(\nTry again.'
                botReply += '\n\nIs there anything else I can help you with?'
                mode = 0

    if mode == 4:
        #--- balance from database ---#
        balance = db.fetch_balance(CID, password, accountNo)
        botReply = 'Your balance is '+str(balance)
        botReply += '\n\nIs there anything else I can help you with?'
        mode = 0

    if mode == 5:
        #--- ask for vendor name ---#
        if not flagMessage:
            botReply = 'Vendor Name'
            flagMessage = True

        else:
            #--- fetch vendor name ---#
            if db.check_vendor(CID, userMessage):
                vendor = userMessage
                mode = 6
                flagMessage = False
            else:
                botReply = 'No such Vendor exists.\nTry again'
                flagMessage = True

    if mode == 6:
        #--- vendor details from database ---#
        vendorDetails = db.fetch_vendor(CID, vendor)
        botReply = 'Vendor Details:\n\n'
        noOfVendors = len(vendorDetails)
        i = 0
        for everyvendor in vendorDetails:
            i += 1
            botReply += '['+str(i)+'] '+'Name: ' + \
                everyvendor[2] + '\nAmount: '+str(everyvendor[3])+'\n\n'
        mode = 7

    if mode == 7:
        #--- ask if user wants to do payment ---#
        if not flagMessage:
            botReply += 'Do you want to make payment? \nIf yes, send item number \n\nOtherwise send 0'
            flagMessage = True

        else:
            #--- fetch index ---#
            index = int(userMessage)-1
            if index == -1:
                botReply = '\n\nIs there anything else I can help you with?'
                mode = 0
            # elif storeFinalTag =='checkOut_vendor':
            #     if index > noOfVendors:
            #         botReply = 'No such item number\nTry again'
            #         flagMessage = True
            # elif storeFinalTag=='checkOut_monthlySubscription':
            #     if index>noOfMonthlySubscription:
            #         botReply = 'No such item number\nTry again'
            #         flagMessage = True
            else:
                mode = 8
                flagMessage = False

    if mode == 8:
        #--- ask for confirmation ---#
        if not flagMessage:
            botReply = 'Are you sure? [Y/N]'
            flagMessage = True

        else:
            #--- fetch confirmation  ---#
            confirmation = userMessage
            if storeFinalTag == 'checkOut_vendor' or storeFinalTag == 'checkOut_monthlySubscription':
                if re.match('^y', confirmation):
                    flagMessage = False
                    mode = 9
                elif re.match('^n', confirmation):
                    flagMessage = True
                    botReply = 'Do you want to make payment? \nIf yes, send item number \n\nOtherwise send 0'
                    mode = 7
                else:
                    botReply = 'Please send correct choice'
                    flagMessage = True
            elif storeFinalTag == 'acc_to_acc':
                if re.match('^y', confirmation):
                    flagMessage = True
                    botReply = 'Password'
                    mode = 3
                elif re.match('^n', confirmation):
                    botReply = 'Transaction Cancelled!'
                    botReply += '\n\nIs there anything else I can help you with?'
                    mode = 0
                else:
                    botReply = 'Please send correct choice'
                    flagMessage = True

    if mode == 9:
        #--- ask for PIN ---#
        if not flagMessage:
            botReply = 'PIN'
            flagMessage = True

        else:
            #--- fetch PIN ---#
            PIN = int(userMessage)

            #--- checkout vendor payment ---#
            if storeFinalTag == 'checkOut_vendor':
                vendorDetails = db.fetch_vendor(CID, vendor)
                if PIN != vendorDetails[index][1]:
                    botReply = 'Invalid PIN! You have to repeat the process!'
                else:
                    Vendor = vendorDetails[index][2]
                    Payment = vendorDetails[index][3]
                    if Payment > db.fetch_balance(CID, password, accountNo):
                        botReply = 'Balance Insufficient!'
                    else:
                        db.checkout_vendor(
                            CID, PIN, Vendor, Payment, password, accountNo)
                        botReply = 'Payment Done!'
                botReply += '\n\nIs there anything else I can help you with?'
                mode = 0

            #--- checkout monthly subscription ---#
            elif storeFinalTag == 'checkOut_monthlySubscription':
                monthlySubscriptionDetails = db.check_monthlySubscription(CID)
                if PIN != monthlySubscriptionDetails[index][1]:
                    botReply = 'Invalid PIN! You have to repeat the process!'
                else:
                    Monthly_Subs = monthlySubscriptionDetails[index][2]
                    Payment = monthlySubscriptionDetails[index][3]
                    if Payment > db.fetch_balance(CID, password, accountNo):
                        botReply = 'Balance Insufficient!'
                    else:
                        db.checkout_monthlySubscription(
                            CID, PIN, Monthly_Subs, Payment, password, accountNo)
                        botReply = 'Payment Done!'
                botReply += '\n\nIs there anything else I can help you with?'
                mode = 0

    if mode == 10:
        #--- monthly subscription details from database ---#
        monthlySubDetails = db.check_monthlySubscription(CID)
        botReply = 'Monthly Subscription Details:\n\n'
        noOfMonthlySubscription = len(monthlySubDetails)
        if len(monthlySubDetails) != 0:
            i = 0
            for everymerchant in monthlySubDetails:
                i += 1
                botReply += '['+str(i)+'] '+'Name: ' + \
                    everymerchant[2] + '\nAmount: ' + \
                    str(everymerchant[3])+'\n\n'
            botReply += 'Do you want to make payment? \nIf yes, send item number \n\nOtherwise send 0'
            flagMessage = True
            mode = 7
        else:
            botReply = 'There are no Monthly Subscriptions!'
            botReply += '\n\nIs there anything else I can help you with?'
            mode = 0

    if mode == 11:
        #--- show last 5 transactions from database ---#
        transactionDetails = db.show_transactions(CID, password, accountNo)
        if len(transactionDetails) != 0:
            botReply = 'Last 5 Transaction Details:\n\n'
            i = 0
            for eachTransaction in transactionDetails:
                i += 1
                botReply += '['+str(i)+'] '+'Payment Type: ' + str(eachTransaction[1]) + '\n      Payment Date: ' + str(
                    eachTransaction[2])+'\n      Payment Time: '+str(eachTransaction[3])+'\n      Payment: '+str(eachTransaction[4])+'\n\n'
                if i == 5:
                    break
        else:
            botReply = 'There have been no transactions'
        botReply += '\n\nIs there anything else I can help you with?'
        mode = 0

    if mode == 12:
        #--- account to account transaction ---#
        visited = True
        if not flagMessage:
            if finaltag != 'acc_to_acc':
                botReply = 'Recipient Account Number'
            else:
                botReply += '\n\nRecipient Account Number'
            flagMessage = True

        else:
            #--- fetch to Account number ---#
            if userMessage == accountNo:
                botReply = 'Cannot send to same account!'
                flagMessage = True
            if db.findAccount(userMessage):
                toAccountNo = userMessage
                mode = 13
                flagMessage = False
            else:
                botReply = 'No such account exists.\nTry again'
                flagMessage = True

    if mode == 13:
        #--- ask for amount ---#
        if not flagMessage:
            botReply = 'Amount to Transfer'
            flagMessage = True

        else:
            #--- fetch amount ---#
            if int(userMessage) > db.fetch_balance(CID, password, accountNo):
                botReply = 'Balance Insufficient!'
                botReply += '\n\nIs there anything else I can help you with?'
                mode = 0
            else:
                amount = int(userMessage)
                botReply = 'Are you sure? [Y/N]'
                flagMessage = True
                mode = 8

    #--- translate ---#
    botReply_Hindi = get_prediction(botReply)
    botReply = botReply_Hindi+'\n\n'+botReply

    bot.send_message(userID, botReply)


bot.poll(debug=True)
#--- bot implementation end ---#
