# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Client
### Namespace: [\YooKassa](../namespaces/yookassa.md)
---
**Summary:**

Класс клиента API.


---
### Examples
Создание клиента

```php
require_once '../vendor/autoload.php';

$client = new \YooKassa\Client();
// по логин/паролю
$client->setAuth('xxxxxx', 'test_XXXXXXX');
// или по oauth токену
$client->setAuthToken('token_XXXXXXX');

```

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [ME_PATH](../classes/YooKassa-Client-BaseClient.md#constant_ME_PATH) |  | Точка входа для запроса к API по магазину |
| public | [PAYMENTS_PATH](../classes/YooKassa-Client-BaseClient.md#constant_PAYMENTS_PATH) |  | Точка входа для запросов к API по платежам |
| public | [REFUNDS_PATH](../classes/YooKassa-Client-BaseClient.md#constant_REFUNDS_PATH) |  | Точка входа для запросов к API по возвратам |
| public | [WEBHOOKS_PATH](../classes/YooKassa-Client-BaseClient.md#constant_WEBHOOKS_PATH) |  | Точка входа для запросов к API по вебхукам |
| public | [RECEIPTS_PATH](../classes/YooKassa-Client-BaseClient.md#constant_RECEIPTS_PATH) |  | Точка входа для запросов к API по чекам |
| public | [DEALS_PATH](../classes/YooKassa-Client-BaseClient.md#constant_DEALS_PATH) |  | Точка входа для запросов к API по сделкам |
| public | [PAYOUTS_PATH](../classes/YooKassa-Client-BaseClient.md#constant_PAYOUTS_PATH) |  | Точка входа для запросов к API по выплатам |
| public | [PERSONAL_DATA_PATH](../classes/YooKassa-Client-BaseClient.md#constant_PERSONAL_DATA_PATH) |  | Точка входа для запросов к API по персональным данным |
| public | [SBP_BANKS_PATH](../classes/YooKassa-Client-BaseClient.md#constant_SBP_BANKS_PATH) |  | Точка входа для запросов к API по участникам СБП |
| public | [SELF_EMPLOYED_PATH](../classes/YooKassa-Client-BaseClient.md#constant_SELF_EMPLOYED_PATH) |  | Точка входа для запросов к API по самозанятым |
| public | [INVOICES_PATH](../classes/YooKassa-Client-BaseClient.md#constant_INVOICES_PATH) |  | Точка входа для запросов к API по счетам |
| public | [PAYMENT_METHODS_PATH](../classes/YooKassa-Client-BaseClient.md#constant_PAYMENT_METHODS_PATH) |  | Точка входа для запросов к API по способам оплаты |
| public | [IDEMPOTENCE_KEY_HEADER](../classes/YooKassa-Client-BaseClient.md#constant_IDEMPOTENCE_KEY_HEADER) |  | Имя HTTP заголовка, используемого для передачи idempotence key |
| public | [DEFAULT_DELAY](../classes/YooKassa-Client-BaseClient.md#constant_DEFAULT_DELAY) |  | Значение по умолчанию времени ожидания между запросами при отправке повторного запроса в случае получения ответа с HTTP статусом 202. |
| public | [DEFAULT_TRIES_COUNT](../classes/YooKassa-Client-BaseClient.md#constant_DEFAULT_TRIES_COUNT) |  | Значение по умолчанию количества попыток получения информации от API если пришёл ответ с HTTP статусом 202 |
| public | [DEFAULT_ATTEMPTS_COUNT](../classes/YooKassa-Client-BaseClient.md#constant_DEFAULT_ATTEMPTS_COUNT) |  | Значение по умолчанию количества попыток получения информации от API если пришёл ответ с HTTP статусом 202 |
| public | [SDK_VERSION](../classes/YooKassa-Client.md#constant_SDK_VERSION) |  | Текущая версия библиотеки. |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$apiClient](../classes/YooKassa-Client-BaseClient.md#property_apiClient) |  | CURL клиент |
| protected | [$attempts](../classes/YooKassa-Client-BaseClient.md#property_attempts) |  | Количество повторных запросов при ответе API статусом 202. |
| protected | [$config](../classes/YooKassa-Client-BaseClient.md#property_config) |  | Настройки для CURL клиента. |
| protected | [$logger](../classes/YooKassa-Client-BaseClient.md#property_logger) |  | Объект для логирования работы SDK. |
| protected | [$login](../classes/YooKassa-Client-BaseClient.md#property_login) |  | shopId магазина. |
| protected | [$password](../classes/YooKassa-Client-BaseClient.md#property_password) |  | Секретный ключ магазина. |
| protected | [$timeout](../classes/YooKassa-Client-BaseClient.md#property_timeout) |  | Время через которое будут осуществляться повторные запросы. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Client-BaseClient.md#method___construct) |  | Constructor. |
| public | [addWebhook()](../classes/YooKassa-Client.md#method_addWebhook) |  | Создание Webhook. |
| public | [cancelPayment()](../classes/YooKassa-Client.md#method_cancelPayment) |  | Отменить незавершенную оплату заказа. |
| public | [capturePayment()](../classes/YooKassa-Client.md#method_capturePayment) |  | Подтверждение платежа. |
| public | [createDeal()](../classes/YooKassa-Client.md#method_createDeal) |  | Создание сделки. |
| public | [createInvoice()](../classes/YooKassa-Client.md#method_createInvoice) |  | Создание счета. |
| public | [createPayment()](../classes/YooKassa-Client.md#method_createPayment) |  | Создание платежа. |
| public | [createPaymentMethod()](../classes/YooKassa-Client.md#method_createPaymentMethod) |  | Создание способа оплаты. |
| public | [createPayout()](../classes/YooKassa-Client.md#method_createPayout) |  | Создание выплаты. |
| public | [createPersonalData()](../classes/YooKassa-Client.md#method_createPersonalData) |  | Создание персональных данных. |
| public | [createReceipt()](../classes/YooKassa-Client.md#method_createReceipt) |  | Отправка чека в облачную кассу. |
| public | [createRefund()](../classes/YooKassa-Client.md#method_createRefund) |  | Проведение возврата платежа. |
| public | [createSelfEmployed()](../classes/YooKassa-Client.md#method_createSelfEmployed) |  | Создание самозанятого. |
| public | [getApiClient()](../classes/YooKassa-Client-BaseClient.md#method_getApiClient) |  | Возвращает CURL клиента для работы с API. |
| public | [getConfig()](../classes/YooKassa-Client-BaseClient.md#method_getConfig) |  | Возвращает настройки клиента. |
| public | [getDealInfo()](../classes/YooKassa-Client.md#method_getDealInfo) |  | Получить информацию о сделке. |
| public | [getDeals()](../classes/YooKassa-Client.md#method_getDeals) |  | Получить список сделок магазина. |
| public | [getInvoiceInfo()](../classes/YooKassa-Client.md#method_getInvoiceInfo) |  | Получить информацию о счете |
| public | [getPaymentInfo()](../classes/YooKassa-Client.md#method_getPaymentInfo) |  | Получить информацию о платеже. |
| public | [getPaymentMethodInfo()](../classes/YooKassa-Client.md#method_getPaymentMethodInfo) |  | Получить информацию о способе оплаты |
| public | [getPayments()](../classes/YooKassa-Client.md#method_getPayments) |  | Получить список платежей магазина. |
| public | [getPayoutInfo()](../classes/YooKassa-Client.md#method_getPayoutInfo) |  | Получить информацию о выплате. |
| public | [getPersonalDataInfo()](../classes/YooKassa-Client.md#method_getPersonalDataInfo) |  | Получить информацию о персональных данных. |
| public | [getReceiptInfo()](../classes/YooKassa-Client.md#method_getReceiptInfo) |  | Получить информацию о чеке. |
| public | [getReceipts()](../classes/YooKassa-Client.md#method_getReceipts) |  | Получить список чеков магазина. |
| public | [getRefundInfo()](../classes/YooKassa-Client.md#method_getRefundInfo) |  | Получить информацию о возврате. |
| public | [getRefunds()](../classes/YooKassa-Client.md#method_getRefunds) |  | Получить список возвратов платежей. |
| public | [getSbpBanks()](../classes/YooKassa-Client.md#method_getSbpBanks) |  | Получить список участников СБП |
| public | [getSelfEmployedInfo()](../classes/YooKassa-Client.md#method_getSelfEmployedInfo) |  | Получить информацию о самозанятом |
| public | [getWebhooks()](../classes/YooKassa-Client.md#method_getWebhooks) |  | Список созданных Webhook. |
| public | [isNotificationIPTrusted()](../classes/YooKassa-Client-BaseClient.md#method_isNotificationIPTrusted) |  | Метод проверяет, находится ли IP адрес среди IP адресов Юkassa, с которых отправляются уведомления. |
| public | [me()](../classes/YooKassa-Client.md#method_me) |  | Информация о магазине. |
| public | [removeWebhook()](../classes/YooKassa-Client.md#method_removeWebhook) |  | Удаление Webhook. |
| public | [setApiClient()](../classes/YooKassa-Client-BaseClient.md#method_setApiClient) |  | Устанавливает CURL клиента для работы с API. |
| public | [setAuth()](../classes/YooKassa-Client-BaseClient.md#method_setAuth) |  | Устанавливает авторизацию по логин/паролю. |
| public | [setAuthToken()](../classes/YooKassa-Client-BaseClient.md#method_setAuthToken) |  | Устанавливает авторизацию по Oauth-токену. |
| public | [setConfig()](../classes/YooKassa-Client-BaseClient.md#method_setConfig) |  | Устанавливает настройки клиента. |
| public | [setLogger()](../classes/YooKassa-Client-BaseClient.md#method_setLogger) |  | Устанавливает логгер приложения. |
| public | [setMaxRequestAttempts()](../classes/YooKassa-Client-BaseClient.md#method_setMaxRequestAttempts) |  | Установка значения количества попыток повторных запросов при статусе 202. |
| public | [setRetryTimeout()](../classes/YooKassa-Client-BaseClient.md#method_setRetryTimeout) |  | Установка значения задержки между повторными запросами. |
| protected | [decodeData()](../classes/YooKassa-Client-BaseClient.md#method_decodeData) |  | Декодирует JSON строку в массив данных. |
| protected | [delay()](../classes/YooKassa-Client-BaseClient.md#method_delay) |  | Задержка между повторными запросами. |
| protected | [encodeData()](../classes/YooKassa-Client-BaseClient.md#method_encodeData) |  | Кодирует массив данных в JSON строку. |
| protected | [execute()](../classes/YooKassa-Client-BaseClient.md#method_execute) |  | Выполнение запроса и обработка 202 статуса. |
| protected | [handleError()](../classes/YooKassa-Client-BaseClient.md#method_handleError) |  | Выбрасывает исключение по коду ошибки. |

---
### Details
* File: [lib/Client.php](../../lib/Client.php)
* Package: Default
* Class Hierarchy: 
  * [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)
  * \YooKassa\Client

---
## Constants
<a name="constant_ME_PATH" class="anchor"></a>
###### ME_PATH
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Точка входа для запроса к API по магазину

```php
ME_PATH = '/me'
```


<a name="constant_PAYMENTS_PATH" class="anchor"></a>
###### PAYMENTS_PATH
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Точка входа для запросов к API по платежам

```php
PAYMENTS_PATH = '/payments'
```


<a name="constant_REFUNDS_PATH" class="anchor"></a>
###### REFUNDS_PATH
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Точка входа для запросов к API по возвратам

```php
REFUNDS_PATH = '/refunds'
```


<a name="constant_WEBHOOKS_PATH" class="anchor"></a>
###### WEBHOOKS_PATH
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Точка входа для запросов к API по вебхукам

```php
WEBHOOKS_PATH = '/webhooks'
```


<a name="constant_RECEIPTS_PATH" class="anchor"></a>
###### RECEIPTS_PATH
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Точка входа для запросов к API по чекам

```php
RECEIPTS_PATH = '/receipts'
```


<a name="constant_DEALS_PATH" class="anchor"></a>
###### DEALS_PATH
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Точка входа для запросов к API по сделкам

```php
DEALS_PATH = '/deals'
```


<a name="constant_PAYOUTS_PATH" class="anchor"></a>
###### PAYOUTS_PATH
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Точка входа для запросов к API по выплатам

```php
PAYOUTS_PATH = '/payouts'
```


<a name="constant_PERSONAL_DATA_PATH" class="anchor"></a>
###### PERSONAL_DATA_PATH
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Точка входа для запросов к API по персональным данным

```php
PERSONAL_DATA_PATH = '/personal_data'
```


<a name="constant_SBP_BANKS_PATH" class="anchor"></a>
###### SBP_BANKS_PATH
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Точка входа для запросов к API по участникам СБП

```php
SBP_BANKS_PATH = '/sbp_banks'
```


<a name="constant_SELF_EMPLOYED_PATH" class="anchor"></a>
###### SELF_EMPLOYED_PATH
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Точка входа для запросов к API по самозанятым

```php
SELF_EMPLOYED_PATH = '/self_employed'
```


<a name="constant_INVOICES_PATH" class="anchor"></a>
###### INVOICES_PATH
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Точка входа для запросов к API по счетам

```php
INVOICES_PATH = '/invoices'
```


<a name="constant_PAYMENT_METHODS_PATH" class="anchor"></a>
###### PAYMENT_METHODS_PATH
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Точка входа для запросов к API по способам оплаты

```php
PAYMENT_METHODS_PATH = '/payment_methods'
```


<a name="constant_IDEMPOTENCE_KEY_HEADER" class="anchor"></a>
###### IDEMPOTENCE_KEY_HEADER
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Имя HTTP заголовка, используемого для передачи idempotence key

```php
IDEMPOTENCE_KEY_HEADER = 'Idempotence-Key'
```


<a name="constant_DEFAULT_DELAY" class="anchor"></a>
###### DEFAULT_DELAY
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Значение по умолчанию времени ожидания между запросами при отправке повторного запроса в случае получения ответа с HTTP статусом 202.

```php
DEFAULT_DELAY = 1800
```


<a name="constant_DEFAULT_TRIES_COUNT" class="anchor"></a>
###### DEFAULT_TRIES_COUNT
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Значение по умолчанию количества попыток получения информации от API если пришёл ответ с HTTP статусом 202

```php
DEFAULT_TRIES_COUNT = 3
```


<a name="constant_DEFAULT_ATTEMPTS_COUNT" class="anchor"></a>
###### DEFAULT_ATTEMPTS_COUNT
Inherited from [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

Значение по умолчанию количества попыток получения информации от API если пришёл ответ с HTTP статусом 202

```php
DEFAULT_ATTEMPTS_COUNT = 3
```


<a name="constant_SDK_VERSION" class="anchor"></a>
###### SDK_VERSION
Текущая версия библиотеки.

```php
SDK_VERSION = '3.10.0'
```



---
## Properties
<a name="property_apiClient"></a>
#### protected $apiClient : ?\YooKassa\Client\ApiClientInterface
---
**Summary**

CURL клиент

**Type:** <a href="../?\YooKassa\Client\ApiClientInterface"><abbr title="?\YooKassa\Client\ApiClientInterface">ApiClientInterface</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)


<a name="property_attempts"></a>
#### protected $attempts : int
---
**Summary**

Количество повторных запросов при ответе API статусом 202.

***Description***

Значение по умолчанию 3

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)


<a name="property_config"></a>
#### protected $config : array
---
**Summary**

Настройки для CURL клиента.

**Type:** <a href="../array"><abbr title="array">array</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)


<a name="property_logger"></a>
#### protected $logger : ?\Psr\Log\LoggerInterface
---
**Summary**

Объект для логирования работы SDK.

**Type:** <a href="../?\Psr\Log\LoggerInterface"><abbr title="?\Psr\Log\LoggerInterface">LoggerInterface</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)


<a name="property_login"></a>
#### protected $login : ?int
---
**Summary**

shopId магазина.

**Type:** <a href="../?int"><abbr title="?int">?int</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)


<a name="property_password"></a>
#### protected $password : ?string
---
**Summary**

Секретный ключ магазина.

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)


<a name="property_timeout"></a>
#### protected $timeout : int
---
**Summary**

Время через которое будут осуществляться повторные запросы.

***Description***

Значение по умолчанию - 1800 миллисекунд.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>
Значение в миллисекундах
**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(?\YooKassa\Client\ApiClientInterface $apiClient = null, ?\YooKassa\Helpers\Config\ConfigurationLoaderInterface $configLoader = null) : mixed
```

**Summary**

Constructor.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">?\YooKassa\Client\ApiClientInterface</code> | apiClient  |  |
| <code lang="php">?\YooKassa\Helpers\Config\ConfigurationLoaderInterface</code> | configLoader  |  |

**Returns:** mixed - 


<a name="method_addWebhook" class="anchor"></a>
#### public addWebhook() : ?\YooKassa\Model\Webhook\Webhook

```php
public addWebhook(array|\YooKassa\Model\Webhook\Webhook $webhookData, null|string $idempotenceKey = null) : ?\YooKassa\Model\Webhook\Webhook
```

**Summary**

Создание Webhook.

**Description**

Запрос позволяет подписаться на уведомления о событии (например, на переход платежа в статус successed).

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Webhook\Webhook</code> | webhookData  | Запрос на создание вебхука |
| <code lang="php">null OR string</code> | idempotenceKey  | [Ключ идемпотентности](https://yookassa.ru/developers/using-api/basics?lang=php#idempotence) |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \Exception |  |

**Returns:** ?\YooKassa\Model\Webhook\Webhook - 
##### Examples:
Создание Webhook:

```php
// Работа с Webhook
// В данном примере мы устанавливаем вебхуки для succeeded и canceled уведомлений.
// А так же проверяем, не установлены ли уже вебхуки. И если установлены на неверный адрес, удаляем.
$client->setAuthToken('token_XXXXXXX');

try {
    $webHookUrl = 'https://merchant-site.ru/payment-notification';
    $needWebHookList = [
        \YooKassa\Model\Notification\NotificationEventType::PAYMENT_SUCCEEDED,
        \YooKassa\Model\Notification\NotificationEventType::PAYMENT_CANCELED,
    ];
    $currentWebHookList = $client->getWebhooks()->getItems();
    foreach ($needWebHookList as $event) {
        $hookIsSet = false;
        foreach ($currentWebHookList as $webHook) {
            if ($webHook->getEvent() !== $event) {
                continue;
            }
            if ($webHook->getUrl() !== $webHookUrl) {
                $client->removeWebhook($webHook->getId());
            } else {
                $hookIsSet = true;
            }

            break;
        }
        if (!$hookIsSet) {
            $client->addWebhook(['event' => $event, 'url' => $webHookUrl]);
        }
    }
    $response = 'SUCCESS';
} catch (\Exception $e) {
    $response = $e;
}

var_dump($response);

```


<a name="method_cancelPayment" class="anchor"></a>
#### public cancelPayment() : ?\YooKassa\Request\Payments\CancelResponse

```php
public cancelPayment(string $paymentId, null|string $idempotenceKey = null) : ?\YooKassa\Request\Payments\CancelResponse
```

**Summary**

Отменить незавершенную оплату заказа.

**Description**

Отменяет платеж, находящийся в статусе `waiting_for_capture`. Отмена платежа значит, что вы
не готовы выдать пользователю товар или оказать услугу. Как только вы отменяете платеж, мы начинаем
возвращать деньги на счет плательщика. Для платежей банковскими картами отмена происходит мгновенно.
Для остальных способов оплаты возврат может занимать до нескольких дней.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | paymentId  | Идентификатор платежа |
| <code lang="php">null OR string</code> | idempotenceKey  | [Ключ идемпотентности](https://yookassa.ru/developers/using-api/basics?lang=php#idempotence) |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов. |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \Exception |  |

**Returns:** ?\YooKassa\Request\Payments\CancelResponse - 
##### Examples:
Отменить незавершенную оплату заказа:

```php
$paymentId = '24e89cb0-000f-5000-9000-1de77fa0d6df';

try {
    $response = $client->cancelPayment($paymentId, uniqid('', true));
    echo $response->getStatus();
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_capturePayment" class="anchor"></a>
#### public capturePayment() : ?\YooKassa\Request\Payments\CreateCaptureResponse

```php
public capturePayment(array|\YooKassa\Request\Payments\CreateCaptureRequestInterface $captureData, string $paymentId, null|string $idempotenceKey = null) : ?\YooKassa\Request\Payments\CreateCaptureResponse
```

**Summary**

Подтверждение платежа.

**Description**

Подтверждает вашу готовность принять платеж. Платеж можно подтвердить, только если он находится
в статусе `waiting_for_capture`. Если платеж подтвержден успешно — значит, оплата прошла, и вы можете выдать
товар или оказать услугу пользователю. На следующий день после подтверждения платеж попадет в реестр,
и ЮKassa переведет деньги на ваш расчетный счет. Если вы не подтверждаете платеж до момента, указанного
в `expire_at`, по умолчанию он отменяется, а деньги возвращаются пользователю. При оплате банковской картой
у вас есть 7 дней на подтверждение платежа. Для остальных способов оплаты платеж необходимо подтвердить
в течение 6 часов.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Payments\CreateCaptureRequestInterface</code> | captureData  | Запрос на создание подтверждения платежа |
| <code lang="php">string</code> | paymentId  | Идентификатор платежа |
| <code lang="php">null OR string</code> | idempotenceKey  | [Ключ идемпотентности](https://yookassa.ru/developers/using-api/basics?lang=php#idempotence) |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \Exception |  |

**Returns:** ?\YooKassa\Request\Payments\CreateCaptureResponse - 
##### Examples:
Подтверждение платежа:

```php
$paymentId = '24e89cb0-000f-5000-9000-1de77fa0d6df';

try {
    $response = $client->capturePayment(
        [
            'amount' => [
                'value' => '1500.00',
                'currency' => 'RUB',
            ],
            'transfers' => [
                [
                    'account_id' => '123',
                    'amount' => [
                        'value' => '1000.00',
                        'currency' => 'RUB',
                    ],
                ],
                [
                    'account_id' => '456',
                    'amount' => [
                        'value' => '500.00',
                        'currency' => 'RUB',
                    ],
                ],
            ],
        ],
        $paymentId,
        uniqid('', true)
    );
    echo $response->getStatus();
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_createDeal" class="anchor"></a>
#### public createDeal() : ?\YooKassa\Request\Deals\CreateDealResponse

```php
public createDeal(array|\YooKassa\Request\Deals\CreateDealRequestInterface $dealData, null|string $idempotenceKey = null) : ?\YooKassa\Request\Deals\CreateDealResponse
```

**Summary**

Создание сделки.

**Description**

Запрос позволяет создать сделку, в рамках которой необходимо принять оплату от покупателя и перечислить ее продавцу.

Необходимо указать следующие параметры:
<ul>
<li>type — Тип сделки. Фиксированное значение: safe_deal — Безопасная сделка;</li>
<li>fee_moment — Момент перечисления вам вознаграждения платформы. Возможные значения: payment_succeeded — после успешной оплаты;</li>
<li>deal_closed — при закрытии сделки после успешной выплаты.</li>
</ul>

Дополнительные параметры:
<ul>
<li>metadata — Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа);</li>
<li>description — Описание сделки (не более 128 символов). Используется для фильтрации при получении списка сделок.</li>
</ul>

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Deals\CreateDealRequestInterface</code> | dealData  | Запрос на создание сделки |
| <code lang="php">null OR string</code> | idempotenceKey  | [Ключ идемпотентности](https://yookassa.ru/developers/using-api/basics?lang=php#idempotence) |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |

**Returns:** ?\YooKassa\Request\Deals\CreateDealResponse - 
##### Examples:
Запрос на создание сделки:

```php
try {
    $response = $client->createDeal(
        [
            'type' => \YooKassa\Model\Deal\DealType::SAFE_DEAL,
            'fee_moment' => \YooKassa\Model\Deal\FeeMoment::PAYMENT_SUCCEEDED,
            'metadata' => [
                'order_id' => '37',
            ],
            'description' => 'SAFE_DEAL 123554642-2432FF344R',
        ],
        uniqid('', true)
    );
    echo $response->getStatus();
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_createInvoice" class="anchor"></a>
#### public createInvoice() : \YooKassa\Model\Invoice\InvoiceInterface|null

```php
public createInvoice(array|\YooKassa\Request\Invoices\CreateInvoiceRequestInterface $invoice, string|null $idempotenceKey = null) : \YooKassa\Model\Invoice\InvoiceInterface|null
```

**Summary**

Создание счета.

**Description**

Используйте этот запрос, чтобы создать в ЮKassa [объект счета](https://yookassa.ru/developers/api?codeLang=bash#create_invoice).

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Invoices\CreateInvoiceRequestInterface</code> | invoice  |  |
| <code lang="php">string OR null</code> | idempotenceKey  |  |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiConnectionException |  |
| \YooKassa\Common\Exceptions\ApiException |  |
| \YooKassa\Common\Exceptions\AuthorizeException |  |
| \YooKassa\Common\Exceptions\BadApiRequestException |  |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException |  |
| \YooKassa\Common\Exceptions\ForbiddenException |  |
| \YooKassa\Common\Exceptions\InternalServerError |  |
| \YooKassa\Common\Exceptions\NotFoundException |  |
| \YooKassa\Common\Exceptions\ResponseProcessingException |  |
| \YooKassa\Common\Exceptions\TooManyRequestsException |  |
| \YooKassa\Common\Exceptions\UnauthorizedException |  |
| \JsonException |  |

**Returns:** \YooKassa\Model\Invoice\InvoiceInterface|null - 
##### Examples:
Запрос на создание счёта:

```php
    'payment_data'=> [
        'amount'=> [
            'value'=> '7.00',
            'currency'=> 'RUB',
        ],
        'capture'=> true,
        'description'=> 'Заказ №96',
        'metadata'=> [
            'order_id'=> '96',
        ],
        'receipt'=> [
            'customer' => [
                'email' => 'example@email.com'
            ],
            'items' => [
                [
                    'description'=> 'Товар арт. 12345',
                    'amount'=> [
                        'value'=> '7.00',
                        'currency'=> 'RUB',
                    ],
                    'quantity'=> 1.000,
                    'vat_code' => 1,
                    'payment_mode' => 'full_payment',
                    'payment_subject' => 'commodity',
                ],
            ],
            'tax_system_code' => 1,
        ],
    ],
    'cart'=> [
        [
            'description'=> 'Товар арт. 12345',
            'price'=> [
                'value'=> '9.00',
                'currency'=> 'RUB',
            ],
            'discount_price'=> [
                'value'=> '7.00',
                'currency'=> 'RUB',
            ],
            'quantity'=> 1.000,
        ],
    ],
    'delivery_method'=> [
        'type'=> 'self',
    ],
    'expires_at'=> '2024-10-20T10:58:18.139Z',
    'description'=> 'Счет на оплату заказа номер 96',
    'metadata'=> [
        'order_id'=> '96',
    ],
];
$idempotenceKey = uniqid('', true);
try {
    $response = $client->createInvoice(
        $invoiceData,
        $idempotenceKey
    );
    if ($response->getDeliveryMethod() && $response->getDeliveryMethod()->getType() === \YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodType::SELF) {
        $redirectUrl = $response->getDeliveryMethod()->getUrl();
        // Далее производим редирект на полученный URL
    }
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}



```


<a name="method_createPayment" class="anchor"></a>
#### public createPayment() : \YooKassa\Request\Payments\CreatePaymentResponse|null

```php
public createPayment(array|\YooKassa\Request\Payments\CreatePaymentRequestInterface $paymentData, null|string $idempotenceKey = null) : \YooKassa\Request\Payments\CreatePaymentResponse|null
```

**Summary**

Создание платежа.

**Description**

Чтобы принять оплату, необходимо создать объект платежа — `Payment`. Он содержит всю необходимую информацию
для проведения оплаты (сумму, валюту и статус). У платежа линейный жизненный цикл, он последовательно
переходит из статуса в статус.

Необходимо указать один из параметров:
<ul>
<li>payment_token — оплата по одноразовому PaymentToken, сформированному виджетом YooKassa JS;</li>
<li>payment_method_id — оплата по сохраненным платежным данным;</li>
<li>payment_method_data — оплата по новым платежным данным.</li>
</ul>

Если не указан ни один параметр и `confirmation.type = redirect`, то в качестве `confirmation_url`
возвращается ссылка, по которой пользователь сможет самостоятельно выбрать подходящий способ оплаты.
Дополнительные параметры:
<ul>
<li>confirmation — передается, если необходимо уточнить способ подтверждения платежа;</li>
<li>recipient — указывается при наличии нескольких товаров;</li>
<li>metadata — дополнительные данные (передаются магазином).</li>
</ul>

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Payments\CreatePaymentRequestInterface</code> | paymentData  | Запрос на создание платежа |
| <code lang="php">null OR string</code> | idempotenceKey  | [Ключ идемпотентности](https://yookassa.ru/developers/using-api/basics?lang=php#idempotence) |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\AuthorizeException |  |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ApiConnectionException |  |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |

**Returns:** \YooKassa\Request\Payments\CreatePaymentResponse|null - 
##### Examples:
Запрос на создание платежа:

```php
try {
    $idempotenceKey = uniqid('', true);
    $response = $client->createPayment(
        [
            'amount' => [
                'value' => '1.00',
                'currency' => 'RUB',
            ],
            'confirmation' => [
                'type' => 'redirect',
                'return_url' => 'https://merchant-site.ru/payment-notification',
            ],
            'description' => 'Оплата заказа на сумму 1 руб',
            'metadata' => [
                'orderNumber' => 1001,
            ],
            'capture' => true,
        ],
        $idempotenceKey
    );
    $confirmation = $response->getConfirmation();
    $redirectUrl = $confirmation->getConfirmationUrl();
    // Далее производим редирект на полученный URL
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_createPaymentMethod" class="anchor"></a>
#### public createPaymentMethod() : \YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface|null

```php
public createPaymentMethod(array|\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface $paymentMethod, string|null $idempotenceKey = null) : \YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface|null
```

**Summary**

Создание способа оплаты.

**Description**

Используйте этот запрос, чтобы создать в ЮKassa [объект способа оплаты](https://yookassa.ru/developers/api#create_payment_method).
В запросе необходимо передать код способа оплаты, который вы хотите сохранить, и при необходимости дополнительные параметры, связанные с той функциональностью, которую вы хотите использовать.

Идентификатор созданного способа оплаты вы можете использовать при проведении [автоплатежей](/developers/payment-acceptance/scenario-extensions/recurring-payments/create-recurring) или [выплат](/developers/payouts/scenario-extensions/multipurpose-token).

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface</code> | paymentMethod  |  |
| <code lang="php">string OR null</code> | idempotenceKey  |  |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiConnectionException |  |
| \YooKassa\Common\Exceptions\ApiException |  |
| \YooKassa\Common\Exceptions\AuthorizeException |  |
| \YooKassa\Common\Exceptions\BadApiRequestException |  |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException |  |
| \YooKassa\Common\Exceptions\ForbiddenException |  |
| \YooKassa\Common\Exceptions\InternalServerError |  |
| \YooKassa\Common\Exceptions\NotFoundException |  |
| \YooKassa\Common\Exceptions\ResponseProcessingException |  |
| \YooKassa\Common\Exceptions\TooManyRequestsException |  |
| \YooKassa\Common\Exceptions\UnauthorizedException |  |
| \JsonException |  |

**Returns:** \YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface|null - 
##### Examples:
Запрос на создание способа оплаты:

```php
// Запрос на создание метода оплаты
$paymentMethod = [
    'confirmation' => [
        'enforce' => true,
        'type' => 'redirect',
        'return_url' => 'https://www.example.com/return_url',
        'locale' => 'ru_RU'
    ],
    'type' => 'bank_card',
    'card' => [
        'number' => '4100000000000002',
        'expiry_year' => '2025',
        'expiry_month' => '12',
        'cardholder' => 'Inna Ivanovna',
        'csc' => '123'
    ],
    'holder' => [
        'gateway_id' => '123456'
    ],
    'client_ip' => '127.0.0.1',
];
$idempotenceKey = uniqid('', true);
try {
    $response = $client->createPaymentMethod(
        $paymentMethod,
        $idempotenceKey
    );
    if ($response->getStatus() && $response->getStatus() === \YooKassa\Model\SavePaymentMethod\SavePaymentMethodStatus::PENDING) {
        $redirectUrl = $response->getConfirmation()->getConfirmationUrl();
        // Далее производим редирект на полученный URL
    }
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);

```


<a name="method_createPayout" class="anchor"></a>
#### public createPayout() : ?\YooKassa\Request\Payouts\CreatePayoutResponse

```php
public createPayout(array|\YooKassa\Request\Payouts\CreatePayoutRequestInterface $payoutData, null|string $idempotenceKey = null) : ?\YooKassa\Request\Payouts\CreatePayoutResponse
```

**Summary**

Создание выплаты.

**Description**

Запрос позволяет перечислить продавцу оплату за выполненную услугу или проданный товар в рамках Безопасной сделки.
Выплату можно сделать на банковскую карту или на кошелек ЮMoney.

Обязательный параметр:
<ul>
<li>amount — сумма выплаты. Есть ограничения на минимальный и максимальный размер выплаты и сумму выплат за месяц.</li>
</ul>

Необходимо указать один из параметров:
<ul>
<li>payout_destination_data — данные платежного средства, на которое нужно сделать выплату;</li>
<li>payout_token — токенизированные данные для выплаты. Например, синоним банковской карты.</li>
</ul>

Дополнительные параметры:
<ul>
<li>description — описание транзакции (не более 128 символов);</li>
<li>deal — сделка, в рамках которой нужно провести выплату. Необходимо передавать, если вы проводите Безопасную сделку;</li>
<li>metadata — любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).</li>
</ul>

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Payouts\CreatePayoutRequestInterface</code> | payoutData  | Запрос на создание выплаты |
| <code lang="php">null OR string</code> | idempotenceKey  | [Ключ идемпотентности](https://yookassa.ru/developers/using-api/basics?lang=php#idempotence) |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности     * @throws NotFoundException Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \Exception |  |

**Returns:** ?\YooKassa\Request\Payouts\CreatePayoutResponse - 
##### Examples:
Запрос на создание выплаты:

```php
$request = [
    'amount' => [
        'value' => '80.00',
        'currency' => 'RUB',
    ],
    'payout_destination_data' => [
        'type' => \YooKassa\Model\Payment\PaymentMethodType::YOO_MONEY,
        'accountNumber' => '4100116075156746',
    ],
    'description' => 'Выплата по заказу №37',
    'metadata' => [
        'order_id' => '37',
    ],
    'deal' => [
        'id' => 'dl-2909e77d-0022-5000-8000-0c37205b3208',
    ],
];

try {
    $idempotenceKey = uniqid('', true);
    $response = $client->createPayout($request, $idempotenceKey);
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_createPersonalData" class="anchor"></a>
#### public createPersonalData() : null|\YooKassa\Request\PersonalData\PersonalDataResponse

```php
public createPersonalData(array|\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest $personalData, null|string $idempotenceKey = null) : null|\YooKassa\Request\PersonalData\PersonalDataResponse
```

**Summary**

Создание персональных данных.

**Description**

Используйте этот запрос, чтобы создать в ЮKassa [объект персональных данных](#personal_data_object).
В запросе необходимо передать фамилию, имя, отчество пользователя и указать, с какой целью эти данные будут использоваться.
Идентификатор созданного объекта персональных данных необходимо использовать в запросе на проведение выплаты через СБП с проверкой получателя.
[Подробнее о выплатах с проверкой получателя](/developers/payouts/scenario-extensions/recipient-check)

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest</code> | personalData  | Запрос на создание персональных данных |
| <code lang="php">null OR string</code> | idempotenceKey  | [Ключ идемпотентности](https://yookassa.ru/developers/using-api/basics?lang=php#idempotence) |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \Exception |  |

**Returns:** null|\YooKassa\Request\PersonalData\PersonalDataResponse - Объект персональных данных
##### Examples:
Запрос на создание персональных данных:

```php
$request = [
    'type' => 'sbp_payout_recipient',
    'last_name' => 'Иванов',
    'first_name' => 'Иван',
    'middle_name' => 'Иванович',
    'metadata' => ['recipient_id' => '37'],
];
$idempotenceKey = uniqid('', true);

try {
    $idempotenceKey = uniqid('', true);
    $response = $client->createPersonalData($request, $idempotenceKey);
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_createReceipt" class="anchor"></a>
#### public createReceipt() : ?\YooKassa\Request\Receipts\AbstractReceiptResponse

```php
public createReceipt(array|\YooKassa\Request\Receipts\CreatePostReceiptRequestInterface $receiptData, null|string $idempotenceKey = null) : ?\YooKassa\Request\Receipts\AbstractReceiptResponse
```

**Summary**

Отправка чека в облачную кассу.

**Description**

Создает объект чека — `Receipt`. Возвращает успешно созданный чек по уникальному идентификатору
платежа или возврата.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Receipts\CreatePostReceiptRequestInterface</code> | receiptData  | Запрос на создание чека |
| <code lang="php">null OR string</code> | idempotenceKey  | [Ключ идемпотентности](https://yookassa.ru/developers/using-api/basics?lang=php#idempotence) |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \YooKassa\Common\Exceptions\AuthorizeException | Ошибка авторизации. Не установлен заголовок |
| \Exception |  |

**Returns:** ?\YooKassa\Request\Receipts\AbstractReceiptResponse - 
##### Examples:
Запрос на создание чека:

```php
try {
    $response = $client->createReceipt(
        [
            'customer' => [
                'email' => 'johndoe@yoomoney.ru',
                'phone' => '79000000000',
            ],
            'type' => 'payment',
            'payment_id' => '24e89cb0-000f-5000-9000-1de77fa0d6df',
            'on_behalf_of' => '123',
            'send' => true,
            'items' => [
                [
                    'description' => 'Платок Gucci',
                    'quantity' => '1.00',
                    'amount' => [
                        'value' => '3000.00',
                        'currency' => 'RUB',
                    ],
                    'vat_code' => 2,
                    'payment_mode' => 'full_payment',
                    'payment_subject' => 'commodity',
                ],
            ],
            'settlements' => [
                [
                    'type' => 'prepayment',
                    'amount' => [
                        'value' => '3000.00',
                        'currency' => 'RUB',
                    ],
                ],
            ],
            'tax_system_code' => 1,
        ],
        uniqid('', true)
    );
    echo $response->getStatus();
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_createRefund" class="anchor"></a>
#### public createRefund() : \YooKassa\Request\Refunds\CreateRefundResponse|null

```php
public createRefund(array|\YooKassa\Request\Refunds\CreateRefundRequestInterface $refundData, null|string $idempotenceKey = null) : \YooKassa\Request\Refunds\CreateRefundResponse|null
```

**Summary**

Проведение возврата платежа.

**Description**

Создает объект возврата — `Refund`. Возвращает успешно завершенный платеж по уникальному идентификатору
этого платежа. Создание возврата возможно только для платежей в статусе `succeeded`. Комиссии за проведение
возврата нет. Комиссия, которую ЮKassa берёт за проведение исходного платежа, не возвращается.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Refunds\CreateRefundRequestInterface</code> | refundData  | Запрос на создание возврата |
| <code lang="php">null OR string</code> | idempotenceKey  | [Ключ идемпотентности](https://yookassa.ru/developers/using-api/basics?lang=php#idempotence) |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiConnectionException |  |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\AuthorizeException |  |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \Exception |  |

**Returns:** \YooKassa\Request\Refunds\CreateRefundResponse|null - 
##### Examples:
Запрос на создание возврата:

```php
try {
    $response = $client->createRefund(
        [
            'payment_id' => '24e89cb0-000f-5000-9000-1de77fa0d6df',
            'amount' => [
                'value' => '1000.00',
                'currency' => 'RUB',
            ],
            'sources' => [
                [
                    'account_id' => '456',
                    'amount' => [
                        'value' => '1000.00',
                        'currency' => 'RUB',
                    ],
                ],
            ],
        ],
        uniqid('', true)
    );
    echo $response->getStatus();
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_createSelfEmployed" class="anchor"></a>
#### public createSelfEmployed() : ?\YooKassa\Request\SelfEmployed\SelfEmployedResponse

```php
public createSelfEmployed(array|\YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface $selfEmployed, null|string $idempotenceKey = null) : ?\YooKassa\Request\SelfEmployed\SelfEmployedResponse
```

**Summary**

Создание самозанятого.

**Description**

Используйте этот запрос, чтобы создать в ЮKassa [объект самозанятого](https://yookassa.ru/developers/api?codeLang=bash#self_employed_object).

В запросе необходимо передать ИНН или телефон самозанятого для идентификации в сервисе Мой налог,
сценарий подтверждения пользователем заявки ЮMoney на получение прав для регистрации чеков и описание самозанятого.

Идентификатор созданного объекта самозанятого необходимо использовать в запросе на проведение выплаты.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface</code> | selfEmployed  | Запрос на создание самозанятого |
| <code lang="php">null OR string</code> | idempotenceKey  | [Ключ идемпотентности](https://yookassa.ru/developers/using-api/basics?lang=php#idempotence) |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |

**Returns:** ?\YooKassa\Request\SelfEmployed\SelfEmployedResponse - 
##### Examples:
Запрос на создание самозанятого:

```php
$request = [
    'itn' => '123456789012',
    'phone' => '79001002030',
    'confirmation' => ['type' => 'redirect'],
];
$idempotenceKey = uniqid('', true);

try {
    $idempotenceKey = uniqid('', true);
    $response = $client->createSelfEmployed($request, $idempotenceKey);
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_getApiClient" class="anchor"></a>
#### public getApiClient() : \YooKassa\Client\ApiClientInterface

```php
public getApiClient() : \YooKassa\Client\ApiClientInterface
```

**Summary**

Возвращает CURL клиента для работы с API.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

**Returns:** \YooKassa\Client\ApiClientInterface - 


<a name="method_getConfig" class="anchor"></a>
#### public getConfig() : array

```php
public getConfig() : array
```

**Summary**

Возвращает настройки клиента.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

**Returns:** array - 


<a name="method_getDealInfo" class="anchor"></a>
#### public getDealInfo() : ?\YooKassa\Model\Deal\DealInterface

```php
public getDealInfo(string $dealId) : ?\YooKassa\Model\Deal\DealInterface
```

**Summary**

Получить информацию о сделке.

**Description**

Запрос позволяет получить информацию о текущем состоянии сделки по её уникальному идентификатору.
Выдает объект чека {@link} в актуальном статусе.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | dealId  | Идентификатор сделки |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |

**Returns:** ?\YooKassa\Model\Deal\DealInterface - 
##### Examples:
Получить информацию о сделке:

```php
try {
    $response = $client->getDealInfo('dl-2909e77d-1022-5003-8004-0c37205b3208');
    echo $response->getStatus();
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_getDeals" class="anchor"></a>
#### public getDeals() : ?\YooKassa\Request\Deals\DealsResponse

```php
public getDeals(null|array|\YooKassa\Request\Deals\DealsRequestInterface $filter = null) : ?\YooKassa\Request\Deals\DealsResponse
```

**Summary**

Получить список сделок магазина.

**Description**

Запрос позволяет получить список сделок, отфильтрованный по заданным критериям.
В ответ на запрос вернется список сделок с учетом переданных параметров. В списке будет информация о сделках,
созданных за последние 3 года. Список будет отсортирован по времени создания сделок в порядке убывания.
Если результатов больше, чем задано в `limit`, список будет выводиться фрагментами.
В этом случае в ответе на запрос вернется фрагмент списка и параметр `next_cursor` с указателем на следующий фрагмент.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Deals\DealsRequestInterface</code> | filter  | Параметры фильтрации |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \Exception |  |

**Returns:** ?\YooKassa\Request\Deals\DealsResponse - 
##### Examples:
Получить список сделок с фильтрацией:

```php
$cursor = null;
$params = [
    'limit' => 30,
    'status' => \YooKassa\Model\Deal\DealStatus::OPENED,
    'full_text_search' => 'DEAL',
    'created_at_gte' => '2023-10-01T00:00:00.000Z',
    'created_at_lt' => '2023-11-01T23:59:59.999Z',
];

try {
    do {
        $params['cursor'] = $cursor;
        $deals = $client->getDeals($params);
        foreach ($deals->getItems() as $deal) {
            $res = [
                $deal->getCreatedAt()->format('Y-m-d H:i:s'),
                $deal->getBalance()->getValue() . ' ' . $deal->getBalance()->getCurrency(),
                $deal->getPayoutBalance()->getValue() . ' ' . $deal->getBalance()->getCurrency(),
                $deal->getStatus(),
                $deal->getId(),
            ];
            echo implode(' - ', $res) . "\n";
        }
    } while ($cursor = $deals->getNextCursor());
} catch (\Exception $e) {
    $response = $e;
    var_dump($response);
}

```


<a name="method_getInvoiceInfo" class="anchor"></a>
#### public getInvoiceInfo() : \YooKassa\Model\Invoice\InvoiceInterface|null

```php
public getInvoiceInfo(string $invoiceId) : \YooKassa\Model\Invoice\InvoiceInterface|null
```

**Summary**

Получить информацию о счете

**Description**

С помощью этого запроса вы можете получить информацию о текущем статусе счета по его уникальному идентификатору.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | invoiceId  | Идентификатор счета |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiConnectionException |  |
| \YooKassa\Common\Exceptions\ApiException |  |
| \YooKassa\Common\Exceptions\AuthorizeException |  |
| \YooKassa\Common\Exceptions\BadApiRequestException |  |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException |  |
| \YooKassa\Common\Exceptions\ForbiddenException |  |
| \YooKassa\Common\Exceptions\InternalServerError |  |
| \JsonException |  |
| \YooKassa\Common\Exceptions\NotFoundException |  |
| \YooKassa\Common\Exceptions\ResponseProcessingException |  |
| \YooKassa\Common\Exceptions\TooManyRequestsException |  |
| \YooKassa\Common\Exceptions\UnauthorizedException |  |

**Returns:** \YooKassa\Model\Invoice\InvoiceInterface|null - 
##### Examples:
Получить информацию о счете:

```php

try {
    $response = $client->getInvoiceInfo($invoiceId);
    if ($response->getPaymentDetails()) {
        print_r($response->getPaymentDetails()->toArray());
        $paymentResponse = $client->getPaymentInfo($response->getPaymentDetails()->getId());
        print_r($paymentResponse->toArray());
    }
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}

// Запрос на создание метода оплаты

```


<a name="method_getPaymentInfo" class="anchor"></a>
#### public getPaymentInfo() : null|\YooKassa\Model\Payment\PaymentInterface

```php
public getPaymentInfo(string $paymentId) : null|\YooKassa\Model\Payment\PaymentInterface
```

**Summary**

Получить информацию о платеже.

**Description**

Запрос позволяет получить информацию о текущем состоянии платежа по его уникальному идентификатору.
Выдает объект платежа {@link} в актуальном статусе.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | paymentId  | Идентификатор платежа |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API. |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов. |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \Exception |  |

**Returns:** null|\YooKassa\Model\Payment\PaymentInterface - Объект платежа
##### Examples:
Получить информацию о платеже:

```php
try {
    $response = $client->getPaymentInfo('24e89cb0-000f-5000-9000-1de77fa0d6df');
    echo $response->getStatus();
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_getPaymentMethodInfo" class="anchor"></a>
#### public getPaymentMethodInfo() : \YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface|null

```php
public getPaymentMethodInfo(string $paymentMethodId) : \YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface|null
```

**Summary**

Получить информацию о способе оплаты

**Description**

Используйте этот запрос, чтобы получить информацию о текущем состоянии способа оплаты по его уникальному идентификатору.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | paymentMethodId  | Идентификатор способа оплаты |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiConnectionException |  |
| \YooKassa\Common\Exceptions\ApiException |  |
| \YooKassa\Common\Exceptions\AuthorizeException |  |
| \YooKassa\Common\Exceptions\BadApiRequestException |  |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException |  |
| \YooKassa\Common\Exceptions\ForbiddenException |  |
| \YooKassa\Common\Exceptions\InternalServerError |  |
| \JsonException |  |
| \YooKassa\Common\Exceptions\NotFoundException |  |
| \YooKassa\Common\Exceptions\ResponseProcessingException |  |
| \YooKassa\Common\Exceptions\TooManyRequestsException |  |
| \YooKassa\Common\Exceptions\UnauthorizedException |  |

**Returns:** \YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface|null - 
##### Examples:
Получить информацию о способе оплаты:

```php
// Получить информацию о способе платежа
$paymentMethodId = '285c0ab7-0003-5000-9000-0e1166498fda';
try {
    $response = $client->getPaymentMethodInfo($paymentMethodId);
    if ($response->getCard()) {
        print_r($response->getCard()->toArray());
    }
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);

```


<a name="method_getPayments" class="anchor"></a>
#### public getPayments() : ?\YooKassa\Request\Payments\PaymentsResponse

```php
public getPayments(array|\YooKassa\Request\Payments\PaymentsRequestInterface|null $filter = null) : ?\YooKassa\Request\Payments\PaymentsResponse
```

**Summary**

Получить список платежей магазина.

**Description**

Запрос позволяет получить список платежей, отфильтрованный по заданным критериям.
В ответ на запрос вернется список платежей с учетом переданных параметров. В списке будет информация о платежах,
созданных за последние 3 года. Список будет отсортирован по времени создания платежей в порядке убывания.
Если результатов больше, чем задано в `limit`, список будет выводиться фрагментами. В этом случае в ответе
на запрос вернется фрагмент списка и параметр `next_cursor` с указателем на следующий фрагмент.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Payments\PaymentsRequestInterface OR null</code> | filter  | Параметры фильтрации |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \Exception |  |

**Returns:** ?\YooKassa\Request\Payments\PaymentsResponse - 
##### Examples:
Получить список платежей магазина с фильтрацией:

```php
$cursor = null;
$params = [
    'limit' => 30,
    'status' => \YooKassa\Model\Payment\PaymentStatus::CANCELED,
    'payment_method' => \YooKassa\Model\Payment\PaymentMethodType::BANK_CARD,
    'created_at_gte' => '2023-01-01T00:00:00.000Z',
    'created_at_lt' => '2023-03-30T23:59:59.999Z',
];

try {
    do {
        $params['cursor'] = $cursor;
        $payments = $client->getPayments($params);
        foreach ($payments->getItems() as $payment) {
            echo $payment->getCreatedAt()->format('Y-m-d H:i:s') . ' - ' .
                 $payment->getStatus() . ' - ' .
                 $payment->getId() . "\n";
        }
    } while ($cursor = $payments->getNextCursor());
} catch (\Exception $e) {
    $response = $e;
}

var_dump($response);

```


<a name="method_getPayoutInfo" class="anchor"></a>
#### public getPayoutInfo() : null|\YooKassa\Model\Payout\PayoutInterface

```php
public getPayoutInfo(string $payoutId) : null|\YooKassa\Model\Payout\PayoutInterface
```

**Summary**

Получить информацию о выплате.

**Description**

Запрос позволяет получить информацию о текущем состоянии выплаты по ее уникальному идентификатору.
Выдает объект выплаты {@link} в актуальном статусе.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | payoutId  | Идентификатор выплаты |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |

**Returns:** null|\YooKassa\Model\Payout\PayoutInterface - Объект выплаты
##### Examples:
Получить информацию о выплате:

```php
$payoutId = 'po-285c0ab7-0003-5000-9000-0e1166498fda';

try {
    $response = $client->getPayoutInfo($payoutId);
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_getPersonalDataInfo" class="anchor"></a>
#### public getPersonalDataInfo() : null|\YooKassa\Model\PersonalData\PersonalDataInterface

```php
public getPersonalDataInfo(string $personalDataId) : null|\YooKassa\Model\PersonalData\PersonalDataInterface
```

**Summary**

Получить информацию о персональных данных.

**Description**

Запрос позволяет получить информацию о текущем состоянии персональных данных по их уникальному идентификатору.
Выдает объект платежа {@link} в актуальном статусе.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | personalDataId  | Идентификатор персональных данных |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |

**Returns:** null|\YooKassa\Model\PersonalData\PersonalDataInterface - Объект персональных данных
##### Examples:
Получить информацию о персональных данных:

```php
$payoutId = 'pd-285c0ab7-0003-5000-9000-0e1166498fda';

try {
    $response = $client->getPersonalDataInfo($payoutId);
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_getReceiptInfo" class="anchor"></a>
#### public getReceiptInfo() : ?\YooKassa\Request\Receipts\ReceiptResponseInterface

```php
public getReceiptInfo(string $receiptId) : ?\YooKassa\Request\Receipts\ReceiptResponseInterface
```

**Summary**

Получить информацию о чеке.

**Description**

Запрос позволяет получить информацию о текущем состоянии чека по его уникальному идентификатору.
Выдает объект чека {@link} в актуальном статусе.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | receiptId  | Идентификатор чека |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |

**Returns:** ?\YooKassa\Request\Receipts\ReceiptResponseInterface - 
##### Examples:
Получить информацию о чеке:

```php
try {
    $response = $client->getPaymentInfo('24e89cb0-000f-5000-9000-1de77fa0d6df');
    echo $response->getStatus();
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_getReceipts" class="anchor"></a>
#### public getReceipts() : \YooKassa\Request\Receipts\ReceiptsResponse

```php
public getReceipts(null|array|\YooKassa\Request\Receipts\ReceiptsRequestInterface $filter = null) : \YooKassa\Request\Receipts\ReceiptsResponse
```

**Summary**

Получить список чеков магазина.

**Description**

Запрос позволяет получить список чеков, отфильтрованный по заданным критериям.
Можно запросить чеки по конкретному платежу, чеки по конкретному возврату или все чеки магазина.
В ответ на запрос вернется список чеков с учетом переданных параметров. В списке будет информация о чеках,
созданных за последние 3 года. Список будет отсортирован по времени создания чеков в порядке убывания.
Если результатов больше, чем задано в `limit`, список будет выводиться фрагментами.
В этом случае в ответе на запрос вернется фрагмент списка и параметр `next_cursor` с указателем на следующий фрагмент.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Receipts\ReceiptsRequestInterface</code> | filter  | Параметры фильтрации |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \Exception |  |

**Returns:** \YooKassa\Request\Receipts\ReceiptsResponse - 
##### Examples:
Получить список чеков магазина с фильтрацией:

```php
$cursor = null;
$params = [
    'limit' => 30,
    'status' => \YooKassa\Model\Payment\PaymentStatus::CANCELED,
    'payment_method' => \YooKassa\Model\Payment\PaymentMethodType::BANK_CARD,
    'created_at_gte' => '2023-01-01T00:00:00.000Z',
    'created_at_lt' => '2023-03-30T23:59:59.999Z',
];

try {
    do {
        $params['cursor'] = $cursor;
        $payments = $client->getPayments($params);
        foreach ($payments->getItems() as $payment) {
            echo $payment->getCreatedAt()->format('Y-m-d H:i:s') . ' - ' .
                 $payment->getStatus() . ' - ' .
                 $payment->getId() . "\n";
        }
    } while ($cursor = $payments->getNextCursor());
} catch (\Exception $e) {
    $response = $e;
}

var_dump($response);

```


<a name="method_getRefundInfo" class="anchor"></a>
#### public getRefundInfo() : ?\YooKassa\Request\Refunds\RefundResponse

```php
public getRefundInfo(string $refundId) : ?\YooKassa\Request\Refunds\RefundResponse
```

**Summary**

Получить информацию о возврате.

**Description**

Запрос позволяет получить информацию о текущем состоянии возврата по его уникальному идентификатору.
В ответ на запрос придет объект возврата {@link} в актуальном статусе.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | refundId  | Идентификатор возврата |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \Exception |  |

**Returns:** ?\YooKassa\Request\Refunds\RefundResponse - 
##### Examples:
Получить информацию о возврате:

```php
try {
    $response = $client->getReceiptInfo('ra-27ed1660-0001-0050-7a5e-10f80e0f0f29');
    echo $response->getStatus();
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_getRefunds" class="anchor"></a>
#### public getRefunds() : ?\YooKassa\Request\Refunds\RefundsResponse

```php
public getRefunds(null|array|\YooKassa\Request\Refunds\RefundsRequestInterface $filter = null) : ?\YooKassa\Request\Refunds\RefundsResponse
```

**Summary**

Получить список возвратов платежей.

**Description**

Запрос позволяет получить список возвратов, отфильтрованный по заданным критериям.
В ответ на запрос вернется список возвратов с учетом переданных параметров. В списке будет информация о возвратах,
созданных за последние 3 года. Список будет отсортирован по времени создания возвратов в порядке убывания.
Если результатов больше, чем задано в `limit`, список будет выводиться фрагментами. В этом случае в ответе
на запрос вернется фрагмент списка и параметр `next_cursor` с указателем на следующий фрагмент.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Refunds\RefundsRequestInterface</code> | filter  | Параметры фильтрации |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности. |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов. |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \Exception |  |

**Returns:** ?\YooKassa\Request\Refunds\RefundsResponse - 
##### Examples:
Получить список возвратов платежей магазина с фильтрацией:

```php
$cursor = null;
$params = [
    'limit' => 30,
    'status' => \YooKassa\Model\Refund\RefundStatus::SUCCEEDED,
    'payment_id' => '1da5c87d-0984-50e8-a7f3-8de646dd9ec9',
    'created_at_gte' => '2023-01-01T00:00:00.000Z',
    'created_at_lt' => '2023-03-30T23:59:59.999Z',
];

try {
    do {
        $params['cursor'] = $cursor;
        $refunds = $client->getRefunds($params);
        foreach ($refunds->getItems() as $refund) {
            echo $refund->getCreatedAt()->format('Y-m-d H:i:s') . ' - ' .
                 $refund->getStatus() . ' - ' .
                 $refund->getId() . "\n";
        }
    } while ($cursor = $refunds->getNextCursor());
} catch (\Exception $e) {
    $response = $e;
}

var_dump($response);

```


<a name="method_getSbpBanks" class="anchor"></a>
#### public getSbpBanks() : ?\YooKassa\Request\Payouts\SbpBanksResponse

```php
public getSbpBanks() : ?\YooKassa\Request\Payouts\SbpBanksResponse
```

**Summary**

Получить список участников СБП

**Description**

С помощью этого запроса вы можете получить актуальный список всех участников СБП.
Список нужно вывести получателю выплаты, идентификатор выбранного участника СБП необходимо использовать
в запросе на создание выплаты.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |

**Returns:** ?\YooKassa\Request\Payouts\SbpBanksResponse - 
##### Examples:
Получить список участников СБП:

```php
try {
    $response = $client->getSbpBanks();
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_getSelfEmployedInfo" class="anchor"></a>
#### public getSelfEmployedInfo() : ?\YooKassa\Model\SelfEmployed\SelfEmployedInterface

```php
public getSelfEmployedInfo(string $selfEmployedId) : ?\YooKassa\Model\SelfEmployed\SelfEmployedInterface
```

**Summary**

Получить информацию о самозанятом

**Description**

С помощью этого запроса вы можете получить информацию о текущем статусе самозанятого по его уникальному идентификатору.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | selfEmployedId  | Идентификатор самозанятого |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |

**Returns:** ?\YooKassa\Model\SelfEmployed\SelfEmployedInterface - 
##### Examples:
Получить информацию о самозанятом:

```php
$payoutId = 'se-285c0ab7-0003-5000-9000-0e1166498fda';

try {
    $response = $client->getSelfEmployedInfo($payoutId);
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_getWebhooks" class="anchor"></a>
#### public getWebhooks() : ?\YooKassa\Request\Webhook\WebhookListResponse

```php
public getWebhooks() : ?\YooKassa\Request\Webhook\WebhookListResponse
```

**Summary**

Список созданных Webhook.

**Description**

Запрос позволяет узнать, какие webhook есть для переданного OAuth-токена.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API. |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \YooKassa\Common\Exceptions\AuthorizeException | Ошибка авторизации. Не установлен заголовок |

**Returns:** ?\YooKassa\Request\Webhook\WebhookListResponse - 
##### Examples:
Список созданных Webhook:

```php
// Работа с Webhook
// В данном примере мы устанавливаем вебхуки для succeeded и canceled уведомлений.
// А так же проверяем, не установлены ли уже вебхуки. И если установлены на неверный адрес, удаляем.
$client->setAuthToken('token_XXXXXXX');

try {
    $webHookUrl = 'https://merchant-site.ru/payment-notification';
    $needWebHookList = [
        \YooKassa\Model\Notification\NotificationEventType::PAYMENT_SUCCEEDED,
        \YooKassa\Model\Notification\NotificationEventType::PAYMENT_CANCELED,
    ];
    $currentWebHookList = $client->getWebhooks()->getItems();
    foreach ($needWebHookList as $event) {
        $hookIsSet = false;
        foreach ($currentWebHookList as $webHook) {
            if ($webHook->getEvent() !== $event) {
                continue;
            }
            if ($webHook->getUrl() !== $webHookUrl) {
                $client->removeWebhook($webHook->getId());
            } else {
                $hookIsSet = true;
            }

            break;
        }
        if (!$hookIsSet) {
            $client->addWebhook(['event' => $event, 'url' => $webHookUrl]);
        }
    }
    $response = 'SUCCESS';
} catch (\Exception $e) {
    $response = $e;
}

var_dump($response);

```


<a name="method_isNotificationIPTrusted" class="anchor"></a>
#### public isNotificationIPTrusted() : bool

```php
public isNotificationIPTrusted(string $ip) : bool
```

**Summary**

Метод проверяет, находится ли IP адрес среди IP адресов Юkassa, с которых отправляются уведомления.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | ip  | IPv4 или IPv6 адрес webhook уведомления |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \Exception | Выбрасывается, если будет передан IP адрес неверного формата |

**Returns:** bool - 


<a name="method_me" class="anchor"></a>
#### public me() : null|array

```php
public me(null|array|int|string $filter = null) : null|array
```

**Summary**

Информация о магазине.

**Description**

Запрос позволяет получить информацию о магазине для переданного OAuth-токена.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR int OR string</code> | filter  | Параметры поиска. В настоящее время доступен только `on_behalf_of` |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов. |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \YooKassa\Common\Exceptions\AuthorizeException | Ошибка авторизации. Не установлен заголовок |

**Returns:** null|array - Массив с информацией о магазине
##### Examples:
Информация о магазине:

```php
try {
    $response = $client->me();
    var_dump($response->toArray());
} catch (\Exception $e) {
    var_dump($e);
}


```


<a name="method_removeWebhook" class="anchor"></a>
#### public removeWebhook() : ?\YooKassa\Model\Webhook\Webhook

```php
public removeWebhook(string $webhookId, null|string $idempotenceKey = null) : ?\YooKassa\Model\Webhook\Webhook
```

**Summary**

Удаление Webhook.

**Description**

Запрос позволяет отписаться от уведомлений о событии для переданного OAuth-токена.
Чтобы удалить webhook, вам нужно передать в запросе его идентификатор.

**Details:**
* Inherited From: [\YooKassa\Client](../classes/YooKassa-Client.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | webhookId  | Идентификатор Webhook |
| <code lang="php">null OR string</code> | idempotenceKey  | [Ключ идемпотентности](https://yookassa.ru/developers/using-api/basics?lang=php#idempotence) |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | Неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API |
| \YooKassa\Common\Exceptions\ForbiddenException | Секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности |
| \YooKassa\Common\Exceptions\NotFoundException | Ресурс не найден |
| \YooKassa\Common\Exceptions\ResponseProcessingException | Запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов |
| \YooKassa\Common\Exceptions\UnauthorizedException | Неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException | Требуемое PHP расширение не установлено |
| \Exception |  |

**Returns:** ?\YooKassa\Model\Webhook\Webhook - 
##### Examples:
Удаление Webhook:

```php
// Работа с Webhook
// В данном примере мы устанавливаем вебхуки для succeeded и canceled уведомлений.
// А так же проверяем, не установлены ли уже вебхуки. И если установлены на неверный адрес, удаляем.
$client->setAuthToken('token_XXXXXXX');

try {
    $webHookUrl = 'https://merchant-site.ru/payment-notification';
    $needWebHookList = [
        \YooKassa\Model\Notification\NotificationEventType::PAYMENT_SUCCEEDED,
        \YooKassa\Model\Notification\NotificationEventType::PAYMENT_CANCELED,
    ];
    $currentWebHookList = $client->getWebhooks()->getItems();
    foreach ($needWebHookList as $event) {
        $hookIsSet = false;
        foreach ($currentWebHookList as $webHook) {
            if ($webHook->getEvent() !== $event) {
                continue;
            }
            if ($webHook->getUrl() !== $webHookUrl) {
                $client->removeWebhook($webHook->getId());
            } else {
                $hookIsSet = true;
            }

            break;
        }
        if (!$hookIsSet) {
            $client->addWebhook(['event' => $event, 'url' => $webHookUrl]);
        }
    }
    $response = 'SUCCESS';
} catch (\Exception $e) {
    $response = $e;
}

var_dump($response);

```


<a name="method_setApiClient" class="anchor"></a>
#### public setApiClient() : $this

```php
public setApiClient(\YooKassa\Client\ApiClientInterface $apiClient) : $this
```

**Summary**

Устанавливает CURL клиента для работы с API.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Client\ApiClientInterface</code> | apiClient  |  |

**Returns:** $this - 


<a name="method_setAuth" class="anchor"></a>
#### public setAuth() : $this

```php
public setAuth(string|int $login, string $password) : $this
```

**Summary**

Устанавливает авторизацию по логин/паролю.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR int</code> | login  |  |
| <code lang="php">string</code> | password  |  |

**Returns:** $this - 
##### Examples:
Пример авторизации:

```php
$client->setAuth('xxxxxx', 'test_XXXXXXX');

```


<a name="method_setAuthToken" class="anchor"></a>
#### public setAuthToken() : $this

```php
public setAuthToken(string $token) : $this
```

**Summary**

Устанавливает авторизацию по Oauth-токену.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | token  |  |

**Returns:** $this - 
##### Examples:
Пример авторизации:

```php
$client->setAuthToken('token_XXXXXXX');

```


<a name="method_setConfig" class="anchor"></a>
#### public setConfig() : void

```php
public setConfig(array $config) : void
```

**Summary**

Устанавливает настройки клиента.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | config  |  |

**Returns:** void - 


<a name="method_setLogger" class="anchor"></a>
#### public setLogger() : self

```php
public setLogger(null|callable|\Psr\Log\LoggerInterface|object $value) : self
```

**Summary**

Устанавливает логгер приложения.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR callable OR \Psr\Log\LoggerInterface OR object</code> | value  | Инстанс логгера |

**Returns:** self - 


<a name="method_setMaxRequestAttempts" class="anchor"></a>
#### public setMaxRequestAttempts() : $this

```php
public setMaxRequestAttempts(int $attempts = self::DEFAULT_ATTEMPTS_COUNT) : $this
```

**Summary**

Установка значения количества попыток повторных запросов при статусе 202.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int</code> | attempts  |  |

**Returns:** $this - 


<a name="method_setRetryTimeout" class="anchor"></a>
#### public setRetryTimeout() : $this

```php
public setRetryTimeout(int $timeout = self::DEFAULT_DELAY) : $this
```

**Summary**

Установка значения задержки между повторными запросами.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int</code> | timeout  |  |

**Returns:** $this - 


<a name="method_decodeData" class="anchor"></a>
#### protected decodeData() : array

```php
protected decodeData(\YooKassa\Common\ResponseObject $response) : array
```

**Summary**

Декодирует JSON строку в массив данных.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ResponseObject</code> | response  | Объект ответа на запрос к API |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \JsonException |  |

**Returns:** array - Массив данных


<a name="method_delay" class="anchor"></a>
#### protected delay() : void

```php
protected delay(\YooKassa\Common\ResponseObject $response) : void
```

**Summary**

Задержка между повторными запросами.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ResponseObject</code> | response  | Объект ответа на запрос к API |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \JsonException |  |

**Returns:** void - 


<a name="method_encodeData" class="anchor"></a>
#### protected encodeData() : string

```php
protected encodeData(array $serializedData) : string
```

**Summary**

Кодирует массив данных в JSON строку.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | serializedData  | Массив данных для кодировки |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \JsonException | Выбрасывается, если не удалось конвертировать данные в строку JSON |

**Returns:** string - Строка JSON


<a name="method_execute" class="anchor"></a>
#### protected execute() : mixed|\YooKassa\Common\ResponseObject

```php
protected execute(string $path, string $method, array $queryParams, null|string $httpBody = null, array $headers = []) : mixed|\YooKassa\Common\ResponseObject
```

**Summary**

Выполнение запроса и обработка 202 статуса.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | path  | URL запроса |
| <code lang="php">string</code> | method  | HTTP метод |
| <code lang="php">array</code> | queryParams  | Массив GET параметров запроса |
| <code lang="php">null OR string</code> | httpBody  | Тело запроса |
| <code lang="php">array</code> | headers  | Массив заголовков запроса |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException |  |
| \YooKassa\Common\Exceptions\AuthorizeException |  |
| \YooKassa\Common\Exceptions\ApiConnectionException |  |
| \YooKassa\Common\Exceptions\ExtensionNotFoundException |  |

**Returns:** mixed|\YooKassa\Common\ResponseObject - 


<a name="method_handleError" class="anchor"></a>
#### protected handleError() : void

```php
protected handleError(\YooKassa\Common\ResponseObject $response) : void
```

**Summary**

Выбрасывает исключение по коду ошибки.

**Details:**
* Inherited From: [\YooKassa\Client\BaseClient](../classes/YooKassa-Client-BaseClient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ResponseObject</code> | response  |  |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\ApiException | неожиданный код ошибки |
| \YooKassa\Common\Exceptions\BadApiRequestException | Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API. |
| \YooKassa\Common\Exceptions\ForbiddenException | секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции |
| \YooKassa\Common\Exceptions\InternalServerError | Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности. |
| \YooKassa\Common\Exceptions\NotFoundException | ресурс не найден |
| \YooKassa\Common\Exceptions\GoneException | ресурс удален |
| \YooKassa\Common\Exceptions\ResponseProcessingException | запрос был принят на обработку, но она не завершена |
| \YooKassa\Common\Exceptions\TooManyRequestsException | Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов. |
| \YooKassa\Common\Exceptions\UnauthorizedException | неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации |
| \YooKassa\Common\Exceptions\AuthorizeException | Ошибка авторизации. Не установлен заголовок. |

**Returns:** void - 



---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney