# The YooKassa API PHP Client Library

[![Latest Stable Version](https://img.shields.io/packagist/v/yoomoney/yookassa-sdk-php?label=stable)](https://packagist.org/packages/yoomoney/yookassa-sdk-php)
[![Total Downloads](https://img.shields.io/packagist/dt/yoomoney/yookassa-sdk-php)](https://packagist.org/packages/yoomoney/yookassa-sdk-php)
[![Monthly Downloads](https://img.shields.io/packagist/dm/yoomoney/yookassa-sdk-php)](https://packagist.org/packages/yoomoney/yookassa-sdk-php)
[![License](https://img.shields.io/packagist/l/yoomoney/yookassa-sdk-php)](https://packagist.org/packages/yoomoney/yookassa-sdk-php)

[Russian](README.md) | English

This product is used for managing payments under [The YooKassa API](https://yookassa.ru/en/developers/api). For usage by those who implemented YooKassa using the API method.

## Requirements
PHP 8.0 (or later version) with the libcurl library

## Installation
### Under console using Composer

1. [Install Composer, a package manager](https://getcomposer.org/download/).
2. In the console, run the following command:
```bash
composer require yoomoney/yookassa-sdk-php
```

### Do the following for the composer.json file of your project:
1. Add a string `"yoomoney/yookassa-sdk-php": "^3.0"` to the list of dependencies of your project in the composer.json file:
```
...
   "require": {
        "php": ">=8.0",
        "yoomoney/yookassa-sdk-php": "^3.0"
...
```
2. Refresh the project's dependencies. In the console, navigate to the catalog with composer.json and run the following command:
```bash
composer update
```
3. Adjust your project's code to activate automated uploading of files for our product:
```php
require __DIR__ . '/vendor/autoload.php';
```

## Commencing work

1. Import required classes:
```php
use YooKassa\Client;
```
2. Create a sample of a client object, then set the store's identifier and secret key (you can get them under your YooKassa's Merchant Profile). [Issuing a secret key](https://yookassa.ru/docs/support/merchant/payments/implement/keys?lang=en)
```php
$client = new Client();
$client->setAuth('shopId', 'secretKey');
```
3. Call the required API method.

   [More details in our documentation for the YooKassa API](https://yookassa.ru/en/developers/api#create_payment)

   [More details in our documentation for the YooKassa SDK](docs/readme.md)

## Examples of using the API SDK

#### [What's new in SDK version 3.x](docs/examples/migration-3x.md)

#### [YooKassa SDK Settings](docs/examples/01-configuration.md)
* [Additional settings for Curl](docs/examples/01-configuration.md#Установка-дополнительных-настроек-для-Curl)
* [Authentication](docs/examples/01-configuration.md#Аутентификация)
* [Statistics about the environment used](docs/examples/01-configuration.md#Статистические-данные-об-используемом-окружении)
* [Getting information about the store](docs/examples/01-configuration.md#Получение-информации-о-магазине)
* [Working with Webhook](docs/examples/01-configuration.md#Работа-с-Webhook)
* [Notifications](docs/examples/01-configuration.md#Входящие-уведомления)

#### [Working with payments](docs/examples/02-payments.md)
* [Request to create a payment](docs/examples/02-payments.md#Запрос-на-создание-платежа)
* [Request to create a payment via the builder](docs/examples/02-payments.md#Запрос-на-создание-платежа-через-билдер)
* [Request for partial payment confirmation](docs/examples/02-payments.md#Запрос-на-частичное-подтверждение-платежа)
* [Request to cancel an incomplete payment](docs/examples/02-payments.md#Запрос-на-отмену-незавершенного-платежа)
* [Get payment information](docs/examples/02-payments.md#Получить-информацию-о-платеже)
* [Get a list of payments with filtering](docs/examples/02-payments.md#Получить-список-платежей-с-фильтрацией)

#### [Working with refunds](docs/examples/03-refunds.md)
* [Request to create a refund](docs/examples/03-refunds.md#Запрос-на-создание-возврата)
* [Request to create a refund via the builder](docs/examples/03-refunds.md#Запрос-на-создание-возврата-через-билдер)
* [Get refund information](docs/examples/03-refunds.md#Получить-информацию-о-возврате)
* [Get a list of returns with filtering](docs/examples/03-refunds.md#Получить-список-возвратов-с-фильтрацией)

#### [Working with receipts](docs/examples/04-receipts.md)
* [Request to create a receipt](docs/examples/04-receipts.md#Запрос-на-создание-чека)
* [Request to create a receipt via the builder](docs/examples/04-receipts.md#Запрос-на-создание-чека-через-билдер)
* [Get information about the receipt](docs/examples/04-receipts.md#Получить-информацию-о-чеке)
* [Get a list of receipts with filtering](docs/examples/04-receipts.md#Получить-список-чеков-с-фильтрацией)

#### [Working with safe deals](docs/examples/05-deals.md)
* [Request to create a deal](docs/examples/05-deals.md#Запрос-на-создание-сделки)
* [Request to create a deal via the builder](docs/examples/05-deals.md#Запрос-на-создание-сделки-через-билдер)
* [Request to create a payment with info about deal](docs/examples/05-deals.md#Запрос-на-создание-платежа-с-привязкой-к-сделке)
* [Get deal information](docs/examples/05-deals.md#Получить-информацию-о-сделке)
* [Get a list of deals with filtering](docs/examples/05-deals.md#Получить-список-сделок-с-фильтрацией)

#### [Working with payouts](docs/examples/06-payouts.md)
* [Request to create a payout](docs/examples/06-payouts.md#Запрос-на-выплату-продавцу)
  * [Payouts to bank card](docs/examples/06-payouts.md#Проведение-выплаты-на-банковскую-карту)
  * [Payouts to YooMoney wallets](docs/examples/06-payouts.md#Проведение-выплаты-в-кошелек-юmoney)
  * [Payouts via Fast Payment Service](docs/examples/06-payouts.md#Проведение-выплаты-через-сбп)
  * [Payouts to self-employed](docs/examples/06-payouts.md#Выплаты-самозанятым)
  * [Payouts by safe deal](docs/examples/06-payouts.md#Проведение-выплаты-по-безопасной-сделке)
* [Get payout information](docs/examples/06-payouts.md#Получить-информацию-о-выплате)

#### [Working with self-employed](docs/examples/07-self-employed.md)
* [Creation of self-employed](docs/examples/07-self-employed.md#Запрос-на-создание-самозанятого)
* [Get information about self-employed](docs/examples/07-self-employed.md#Получить-информацию-о-самозанятом)

#### [Working with personal data](docs/examples/08-personal-data.md)
* [Creation of personal data](docs/examples/08-personal-data.md#Создание-персональных-данных)
* [Get information about personal data](docs/examples/08-personal-data.md#Получить-информацию-о-персональных-данных)

#### [Working with the list of Fast Payment Service participants](docs/examples/09-sbp-banks.md)
* [Get a list of Fast Payment Service participants](docs/examples/09-sbp-banks.md#Получить-список-участников-СБП)

#### [Working with invoices](docs/examples/10-invoices.md)
* [Request to create an invoice](docs/examples/10-invoices.md#Запрос-на-создание-счета)
* [Request to create an invoice via the builder](docs/examples/10-invoices.md#Запрос-на-создание-счета-через-билдер)
* [Get information about the invoice](docs/examples/10-invoices.md#Получить-информацию-о-счете)

#### [Working with payment methods](docs/examples/11-payment-methods.md)
* [Request to create payment method](docs/examples/11-payment-methods.md#Запрос-на-создание-способа-оплаты)
* [Request to create payment method via the builder](docs/examples/11-payment-methods.md#Запрос-на-создание-способа-оплаты-через-билдер)
* [Get information about the payment method](docs/examples/11-payment-methods.md#Получить-информацию-о-способе-оплаты)
