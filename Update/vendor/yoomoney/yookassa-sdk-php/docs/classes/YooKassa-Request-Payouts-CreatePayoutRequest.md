# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payouts\CreatePayoutRequest
### Namespace: [\YooKassa\Request\Payouts](../namespaces/yookassa-request-payouts.md)
---
**Summary:**

Класс, представляющий модель CreatePayoutRequest.

**Description:**

Класс объекта запроса к API на проведение новой выплаты.

---
### Examples
Пример использования билдера

```php
try {
    $payoutBuilder = \YooKassa\Request\Payouts\CreatePayoutRequest::builder();
    $payoutBuilder
        ->setAmount(new \YooKassa\Model\MonetaryAmount(80))
        ->setPayoutDestinationData(
            new \YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataYooMoney(
                [
                    'type' => \YooKassa\Model\Payment\PaymentMethodType::YOO_MONEY,
                    'account_number' => '4100116075156746'
                ]
            )
        )
        ->setDeal(new \YooKassa\Model\Deal\PayoutDealInfo(['id' => 'dl-2909e77d-0022-5000-8000-0c37205b3208']))
        ->setDescription('Выплата по заказу №37')
    ;

    // Создаем объект запроса
    $request = $payoutBuilder->build();

    $idempotenceKey = uniqid('', true);
    $response = $client->createPayout($request, $idempotenceKey);
} catch (Exception $e) {
    $response = $e;
}

var_dump($response);

```

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [MAX_LENGTH_DESCRIPTION](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#constant_MAX_LENGTH_DESCRIPTION) |  |  |
| public | [MAX_PERSONAL_DATA](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#constant_MAX_PERSONAL_DATA) |  |  |
| public | [MIN_PERSONAL_DATA](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#constant_MIN_PERSONAL_DATA) |  |  |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$amount](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_amount) |  | Сумма создаваемой выплаты |
| public | [$deal](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_deal) |  | Сделка, в рамках которой нужно провести выплату. Необходимо передавать, если вы проводите Безопасную сделку |
| public | [$description](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_description) |  | Описание транзакции (не более 128 символов). Например: «Выплата по договору N» |
| public | [$metadata](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_metadata) |  | Метаданные привязанные к выплате |
| public | [$payment_method_id](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_payment_method_id) |  | Идентификатор сохраненного способа оплаты, данные которого нужно использовать для проведения выплаты |
| public | [$paymentMethodId](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_paymentMethodId) |  | Идентификатор сохраненного способа оплаты, данные которого нужно использовать для проведения выплаты |
| public | [$payout_destination_data](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_payout_destination_data) |  | Данные платежного средства, на которое нужно сделать выплату. Обязательный параметр, если не передан payout_token. |
| public | [$payout_token](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_payout_token) |  | Токенизированные данные для выплаты. Например, синоним банковской карты. Обязательный параметр, если не передан payout_destination_data |
| public | [$payoutDestinationData](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_payoutDestinationData) |  | Данные платежного средства, на которое нужно сделать выплату. Обязательный параметр, если не передан payout_token. |
| public | [$payoutToken](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_payoutToken) |  | Токенизированные данные для выплаты. Например, синоним банковской карты. Обязательный параметр, если не передан payout_destination_data |
| public | [$personal_data](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_personal_data) |  | Персональные данные получателя выплаты. Только для обычных выплат. Необходимо передавать в этих сценариях: - [выплаты с проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check) (только для выплат через СБП); - [выплаты с передачей данных получателя для выписок из реестра](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-data-send). В массиве можно одновременно передать несколько идентификаторов, но только для разных типов данных. |
| public | [$personalData](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_personalData) |  | Персональные данные получателя выплаты. Только для обычных выплат. Необходимо передавать в этих сценариях: - [выплаты с проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check) (только для выплат через СБП); - [выплаты с передачей данных получателя для выписок из реестра](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-data-send). В массиве можно одновременно передать несколько идентификаторов, но только для разных типов данных. |
| public | [$receipt_data](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_receipt_data) |  | Данные для формирования чека в сервисе Мой налог. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат. |
| public | [$receiptData](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_receiptData) |  | Данные для формирования чека в сервисе Мой налог. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат. |
| public | [$self_employed](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_self_employed) |  | Данные самозанятого, который получит выплату. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат. |
| public | [$selfEmployed](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#property_selfEmployed) |  | Данные самозанятого, который получит выплату. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [builder()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_builder) |  | Возвращает билдер объектов запросов создания выплаты. |
| public | [clearValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_clearValidationError) |  | Очищает статус валидации текущего запроса. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAmount()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_getAmount) |  | Возвращает сумму выплаты. |
| public | [getDeal()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_getDeal) |  | Возвращает сделку, в рамках которой нужно провести выплату. |
| public | [getDescription()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_getDescription) |  | Возвращает описание транзакции. |
| public | [getLastValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_getLastValidationError) |  | Возвращает последнюю ошибку валидации. |
| public | [getMetadata()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_getMetadata) |  | Возвращает данные оплаты установленные мерчантом. |
| public | [getPaymentMethodId()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_getPaymentMethodId) |  | Возвращает идентификатор сохраненного способа оплаты. |
| public | [getPayoutDestinationData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_getPayoutDestinationData) |  | Возвращает данные для создания метода оплаты. |
| public | [getPayoutToken()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_getPayoutToken) |  | Возвращает токенизированные данные для выплаты. |
| public | [getPersonalData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_getPersonalData) |  | Возвращает персональные данные получателя выплаты. |
| public | [getReceiptData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_getReceiptData) |  | Возвращает данные для формирования чека в сервисе Мой налог. |
| public | [getSelfEmployed()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_getSelfEmployed) |  | Возвращает данные самозанятого, который получит выплату. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [hasAmount()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_hasAmount) |  | Проверяет, была ли установлена сумма выплаты. |
| public | [hasDeal()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_hasDeal) |  | Проверяет наличие сделки в создаваемой выплате. |
| public | [hasDescription()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_hasDescription) |  | Проверяет наличие описания транзакции в создаваемом платеже. |
| public | [hasMetadata()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_hasMetadata) |  | Проверяет, были ли установлены метаданные выплаты. |
| public | [hasPaymentMethodId()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_hasPaymentMethodId) |  | Проверяет наличие идентификатора сохраненного способа оплаты. |
| public | [hasPayoutDestinationData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_hasPayoutDestinationData) |  | Проверяет установлен ли объект с методом оплаты. |
| public | [hasPayoutToken()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_hasPayoutToken) |  | Проверяет наличие токенизированных данных для выплаты. |
| public | [hasPersonalData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_hasPersonalData) |  | Проверяет наличие персональных данных в создаваемой выплате. |
| public | [hasReceiptData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_hasReceiptData) |  | Проверяет наличие данных для формирования чека в сервисе Мой налог. |
| public | [hasSelfEmployed()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_hasSelfEmployed) |  | Проверяет наличие данных самозанятого в создаваемой выплате. |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAmount()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_setAmount) |  | Устанавливает сумму выплаты. |
| public | [setDeal()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_setDeal) |  | Устанавливает сделку, в рамках которой нужно провести выплату. |
| public | [setDescription()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_setDescription) |  | Устанавливает описание транзакции. |
| public | [setMetadata()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_setMetadata) |  | Устанавливает метаданные, привязанные к выплате. |
| public | [setPaymentMethodId()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_setPaymentMethodId) |  | Устанавливает идентификатор сохраненного способа оплаты. |
| public | [setPayoutDestinationData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_setPayoutDestinationData) |  | Устанавливает объект с информацией для создания метода оплаты. |
| public | [setPayoutToken()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_setPayoutToken) |  | Устанавливает токенизированные данные для выплаты. |
| public | [setPersonalData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_setPersonalData) |  | Устанавливает персональные данные получателя выплаты. |
| public | [setReceiptData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_setReceiptData) |  | Устанавливает данные для формирования чека в сервисе Мой налог. |
| public | [setSelfEmployed()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_setSelfEmployed) |  | Устанавливает данные самозанятого, который получит выплату. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| public | [validate()](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md#method_validate) |  | Проверяет на валидность текущий объект |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [setValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_setValidationError) |  | Устанавливает ошибку валидации. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payouts/CreatePayoutRequest.php](../../lib/Request/Payouts/CreatePayoutRequest.php)
* Package: YooKassa\Request
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)
  * \YooKassa\Request\Payouts\CreatePayoutRequest
* Implements:
  * [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_MAX_LENGTH_DESCRIPTION" class="anchor"></a>
###### MAX_LENGTH_DESCRIPTION
```php
MAX_LENGTH_DESCRIPTION = 128 : int
```


<a name="constant_MAX_PERSONAL_DATA" class="anchor"></a>
###### MAX_PERSONAL_DATA
```php
MAX_PERSONAL_DATA = 2 : int
```


<a name="constant_MIN_PERSONAL_DATA" class="anchor"></a>
###### MIN_PERSONAL_DATA
```php
MIN_PERSONAL_DATA = 1 : int
```



---
## Properties
<a name="property_amount"></a>
#### public $amount : \YooKassa\Model\AmountInterface
---
***Description***

Сумма создаваемой выплаты

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_deal"></a>
#### public $deal : \YooKassa\Model\Deal\PayoutDealInfo
---
***Description***

Сделка, в рамках которой нужно провести выплату. Необходимо передавать, если вы проводите Безопасную сделку

**Type:** <a href="../classes/YooKassa-Model-Deal-PayoutDealInfo.html"><abbr title="\YooKassa\Model\Deal\PayoutDealInfo">PayoutDealInfo</abbr></a>

**Details:**


<a name="property_description"></a>
#### public $description : string
---
***Description***

Описание транзакции (не более 128 символов). Например: «Выплата по договору N»

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_metadata"></a>
#### public $metadata : \YooKassa\Model\Metadata
---
***Description***

Метаданные привязанные к выплате

**Type:** <a href="../classes/YooKassa-Model-Metadata.html"><abbr title="\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**


<a name="property_payment_method_id"></a>
#### public $payment_method_id : string
---
***Description***

Идентификатор сохраненного способа оплаты, данные которого нужно использовать для проведения выплаты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_paymentMethodId"></a>
#### public $paymentMethodId : string
---
***Description***

Идентификатор сохраненного способа оплаты, данные которого нужно использовать для проведения выплаты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_payout_destination_data"></a>
#### public $payout_destination_data : \YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData
---
***Description***

Данные платежного средства, на которое нужно сделать выплату. Обязательный параметр, если не передан payout_token.

**Type:** <a href="../classes/YooKassa-Request-Payouts-PayoutDestinationData-AbstractPayoutDestinationData.html"><abbr title="\YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData">AbstractPayoutDestinationData</abbr></a>

**Details:**


<a name="property_payout_token"></a>
#### public $payout_token : string
---
***Description***

Токенизированные данные для выплаты. Например, синоним банковской карты. Обязательный параметр, если не передан payout_destination_data

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_payoutDestinationData"></a>
#### public $payoutDestinationData : \YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData
---
***Description***

Данные платежного средства, на которое нужно сделать выплату. Обязательный параметр, если не передан payout_token.

**Type:** <a href="../classes/YooKassa-Request-Payouts-PayoutDestinationData-AbstractPayoutDestinationData.html"><abbr title="\YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData">AbstractPayoutDestinationData</abbr></a>

**Details:**


<a name="property_payoutToken"></a>
#### public $payoutToken : string
---
***Description***

Токенизированные данные для выплаты. Например, синоним банковской карты. Обязательный параметр, если не передан payout_destination_data

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_personal_data"></a>
#### public $personal_data : \YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payouts\PayoutPersonalData[]
---
***Description***

Персональные данные получателя выплаты. Только для обычных выплат. Необходимо передавать в этих сценариях: - [выплаты с проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check) (только для выплат через СБП); - [выплаты с передачей данных получателя для выписок из реестра](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-data-send). В массиве можно одновременно передать несколько идентификаторов, но только для разных типов данных.

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payouts\PayoutPersonalData[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payouts\PayoutPersonalData[]">PayoutPersonalData[]</abbr></a>

**Details:**


<a name="property_personalData"></a>
#### public $personalData : \YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payouts\PayoutPersonalData[]
---
***Description***

Персональные данные получателя выплаты. Только для обычных выплат. Необходимо передавать в этих сценариях: - [выплаты с проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check) (только для выплат через СБП); - [выплаты с передачей данных получателя для выписок из реестра](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-data-send). В массиве можно одновременно передать несколько идентификаторов, но только для разных типов данных.

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payouts\PayoutPersonalData[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Request\Payouts\PayoutPersonalData[]">PayoutPersonalData[]</abbr></a>

**Details:**


<a name="property_receipt_data"></a>
#### public $receipt_data : \YooKassa\Request\Payouts\IncomeReceiptData
---
***Description***

Данные для формирования чека в сервисе Мой налог. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат.

**Type:** <a href="../classes/YooKassa-Request-Payouts-IncomeReceiptData.html"><abbr title="\YooKassa\Request\Payouts\IncomeReceiptData">IncomeReceiptData</abbr></a>

**Details:**


<a name="property_receiptData"></a>
#### public $receiptData : \YooKassa\Request\Payouts\IncomeReceiptData
---
***Description***

Данные для формирования чека в сервисе Мой налог. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат.

**Type:** <a href="../classes/YooKassa-Request-Payouts-IncomeReceiptData.html"><abbr title="\YooKassa\Request\Payouts\IncomeReceiptData">IncomeReceiptData</abbr></a>

**Details:**


<a name="property_self_employed"></a>
#### public $self_employed : \YooKassa\Request\Payouts\PayoutSelfEmployedInfo
---
***Description***

Данные самозанятого, который получит выплату. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат.

**Type:** <a href="../classes/YooKassa-Request-Payouts-PayoutSelfEmployedInfo.html"><abbr title="\YooKassa\Request\Payouts\PayoutSelfEmployedInfo">PayoutSelfEmployedInfo</abbr></a>

**Details:**


<a name="property_selfEmployed"></a>
#### public $selfEmployed : \YooKassa\Request\Payouts\PayoutSelfEmployedInfo
---
***Description***

Данные самозанятого, который получит выплату. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат.

**Type:** <a href="../classes/YooKassa-Request-Payouts-PayoutSelfEmployedInfo.html"><abbr title="\YooKassa\Request\Payouts\PayoutSelfEmployedInfo">PayoutSelfEmployedInfo</abbr></a>

**Details:**



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

AbstractObject constructor.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  |  |

**Returns:** mixed - 


<a name="method___get" class="anchor"></a>
#### public __get() : mixed

```php
public __get(string $propertyName) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method___isset" class="anchor"></a>
#### public __isset() : bool

```php
public __isset(string $propertyName) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method___set" class="anchor"></a>
#### public __set() : void

```php
public __set(string $propertyName, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method___unset" class="anchor"></a>
#### public __unset() : void

```php
public __unset(string $propertyName) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_builder" class="anchor"></a>
#### public builder() : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder

```php
Static public builder() : \YooKassa\Request\Payouts\CreatePayoutRequestBuilder
```

**Summary**

Возвращает билдер объектов запросов создания выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** \YooKassa\Request\Payouts\CreatePayoutRequestBuilder - Инстанс билдера объектов запросов


<a name="method_clearValidationError" class="anchor"></a>
#### public clearValidationError() : void

```php
public clearValidationError() : void
```

**Summary**

Очищает статус валидации текущего запроса.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

**Returns:** void - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array|\Traversable $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \Traversable</code> | sourceArray  | Ассоциативный массив с настройками |

**Returns:** void - 


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма выплаты


<a name="method_getDeal" class="anchor"></a>
#### public getDeal() : \YooKassa\Model\Deal\PayoutDealInfo|null

```php
public getDeal() : \YooKassa\Model\Deal\PayoutDealInfo|null
```

**Summary**

Возвращает сделку, в рамках которой нужно провести выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** \YooKassa\Model\Deal\PayoutDealInfo|null - Сделка, в рамках которой нужно провести выплату


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает описание транзакции.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** string|null - Описание транзакции


<a name="method_getLastValidationError" class="anchor"></a>
#### public getLastValidationError() : string|null

```php
public getLastValidationError() : string|null
```

**Summary**

Возвращает последнюю ошибку валидации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

**Returns:** string|null - Последняя произошедшая ошибка валидации


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает данные оплаты установленные мерчантом.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** \YooKassa\Model\Metadata|null - Метаданные, привязанные к выплате


<a name="method_getPaymentMethodId" class="anchor"></a>
#### public getPaymentMethodId() : null|string

```php
public getPaymentMethodId() : null|string
```

**Summary**

Возвращает идентификатор сохраненного способа оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** null|string - Идентификатор сохраненного способа оплаты


<a name="method_getPayoutDestinationData" class="anchor"></a>
#### public getPayoutDestinationData() : \YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData|null

```php
public getPayoutDestinationData() : \YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData|null
```

**Summary**

Возвращает данные для создания метода оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** \YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData|null - Данные используемые для создания метода оплаты


<a name="method_getPayoutToken" class="anchor"></a>
#### public getPayoutToken() : string|null

```php
public getPayoutToken() : string|null
```

**Summary**

Возвращает токенизированные данные для выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** string|null - Токенизированные данные для выплаты


<a name="method_getPersonalData" class="anchor"></a>
#### public getPersonalData() : \YooKassa\Request\Payouts\PayoutPersonalData[]|\YooKassa\Common\ListObjectInterface

```php
public getPersonalData() : \YooKassa\Request\Payouts\PayoutPersonalData[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает персональные данные получателя выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** \YooKassa\Request\Payouts\PayoutPersonalData[]|\YooKassa\Common\ListObjectInterface - Персональные данные получателя выплаты


<a name="method_getReceiptData" class="anchor"></a>
#### public getReceiptData() : null|\YooKassa\Request\Payouts\IncomeReceiptData

```php
public getReceiptData() : null|\YooKassa\Request\Payouts\IncomeReceiptData
```

**Summary**

Возвращает данные для формирования чека в сервисе Мой налог.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** null|\YooKassa\Request\Payouts\IncomeReceiptData - Данные для формирования чека в сервисе Мой налог


<a name="method_getSelfEmployed" class="anchor"></a>
#### public getSelfEmployed() : null|\YooKassa\Request\Payouts\PayoutSelfEmployedInfo

```php
public getSelfEmployed() : null|\YooKassa\Request\Payouts\PayoutSelfEmployedInfo
```

**Summary**

Возвращает данные самозанятого, который получит выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** null|\YooKassa\Request\Payouts\PayoutSelfEmployedInfo - Данные самозанятого, который получит выплату


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_hasAmount" class="anchor"></a>
#### public hasAmount() : bool

```php
public hasAmount() : bool
```

**Summary**

Проверяет, была ли установлена сумма выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** bool - True если сумма выплаты была установлена, false если нет


<a name="method_hasDeal" class="anchor"></a>
#### public hasDeal() : bool

```php
public hasDeal() : bool
```

**Summary**

Проверяет наличие сделки в создаваемой выплате.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** bool - True если сделка есть, false если нет


<a name="method_hasDescription" class="anchor"></a>
#### public hasDescription() : bool

```php
public hasDescription() : bool
```

**Summary**

Проверяет наличие описания транзакции в создаваемом платеже.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** bool - True если описание транзакции есть, false если нет


<a name="method_hasMetadata" class="anchor"></a>
#### public hasMetadata() : bool

```php
public hasMetadata() : bool
```

**Summary**

Проверяет, были ли установлены метаданные выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** bool - True если метаданные были установлены, false если нет


<a name="method_hasPaymentMethodId" class="anchor"></a>
#### public hasPaymentMethodId() : bool

```php
public hasPaymentMethodId() : bool
```

**Summary**

Проверяет наличие идентификатора сохраненного способа оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** bool - True если идентификатора установлен, false если нет


<a name="method_hasPayoutDestinationData" class="anchor"></a>
#### public hasPayoutDestinationData() : bool

```php
public hasPayoutDestinationData() : bool
```

**Summary**

Проверяет установлен ли объект с методом оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** bool - True если объект метода оплаты установлен, false если нет


<a name="method_hasPayoutToken" class="anchor"></a>
#### public hasPayoutToken() : bool

```php
public hasPayoutToken() : bool
```

**Summary**

Проверяет наличие токенизированных данных для выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** bool - True если токен установлен, false если нет


<a name="method_hasPersonalData" class="anchor"></a>
#### public hasPersonalData() : bool

```php
public hasPersonalData() : bool
```

**Summary**

Проверяет наличие персональных данных в создаваемой выплате.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** bool - True если персональные данные есть, false если нет


<a name="method_hasReceiptData" class="anchor"></a>
#### public hasReceiptData() : bool

```php
public hasReceiptData() : bool
```

**Summary**

Проверяет наличие данных для формирования чека в сервисе Мой налог.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** bool - True если данные для формирования чека есть, false если нет


<a name="method_hasSelfEmployed" class="anchor"></a>
#### public hasSelfEmployed() : bool

```php
public hasSelfEmployed() : bool
```

**Summary**

Проверяет наличие данных самозанятого в создаваемой выплате.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** bool - True если данные самозанятого есть, false если нет


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_offsetExists" class="anchor"></a>
#### public offsetExists() : bool

```php
public offsetExists(string $offset) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method_offsetGet" class="anchor"></a>
#### public offsetGet() : mixed

```php
public offsetGet(string $offset) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method_offsetSet" class="anchor"></a>
#### public offsetSet() : void

```php
public offsetSet(string $offset, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method_offsetUnset" class="anchor"></a>
#### public offsetUnset() : void

```php
public offsetUnset(string $offset) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : self

```php
public setAmount(\YooKassa\Model\AmountInterface|array|null $amount) : self
```

**Summary**

Устанавливает сумму выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | amount  | Сумма выплаты |

**Returns:** self - 


<a name="method_setDeal" class="anchor"></a>
#### public setDeal() : self

```php
public setDeal(\YooKassa\Model\Deal\PayoutDealInfo|array|null $deal) : self
```

**Summary**

Устанавливает сделку, в рамках которой нужно провести выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Deal\PayoutDealInfo OR array OR null</code> | deal  | Сделка, в рамках которой нужно провести выплату |

**Returns:** self - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description = null) : self
```

**Summary**

Устанавливает описание транзакции.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Описание транзакции |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : $this

```php
public setMetadata(\YooKassa\Model\Metadata|array|null $metadata = null) : $this
```

**Summary**

Устанавливает метаданные, привязанные к выплате.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Metadata OR array OR null</code> | metadata  | Метаданные выплаты, устанавливаемые мерчантом |

**Returns:** $this - 


<a name="method_setPaymentMethodId" class="anchor"></a>
#### public setPaymentMethodId() : self

```php
public setPaymentMethodId(null|string $payment_method_id = null) : self
```

**Summary**

Устанавливает идентификатор сохраненного способа оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | payment_method_id  | Идентификатор сохраненного способа оплаты |

**Returns:** self - 


<a name="method_setPayoutDestinationData" class="anchor"></a>
#### public setPayoutDestinationData() : self

```php
public setPayoutDestinationData(\YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData|array|null $payout_destination_data) : self
```

**Summary**

Устанавливает объект с информацией для создания метода оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData OR array OR null</code> | payout_destination_data  | Объект создания метода оплаты или null |

**Returns:** self - 


<a name="method_setPayoutToken" class="anchor"></a>
#### public setPayoutToken() : self

```php
public setPayoutToken(string|null $payout_token) : self
```

**Summary**

Устанавливает токенизированные данные для выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | payout_token  | Токенизированные данные для выплаты |

**Returns:** self - 


<a name="method_setPersonalData" class="anchor"></a>
#### public setPersonalData() : self

```php
public setPersonalData(\YooKassa\Common\ListObjectInterface|array|null $personal_data = null) : self
```

**Summary**

Устанавливает персональные данные получателя выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR array OR null</code> | personal_data  | Персональные данные получателя выплаты |

**Returns:** self - 


<a name="method_setReceiptData" class="anchor"></a>
#### public setReceiptData() : $this

```php
public setReceiptData(\YooKassa\Request\Payouts\IncomeReceiptData|array|null $receipt_data = null) : $this
```

**Summary**

Устанавливает данные для формирования чека в сервисе Мой налог.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payouts\IncomeReceiptData OR array OR null</code> | receipt_data  | Данные для формирования чека в сервисе Мой налог |

**Returns:** $this - 


<a name="method_setSelfEmployed" class="anchor"></a>
#### public setSelfEmployed() : self

```php
public setSelfEmployed(\YooKassa\Request\Payouts\PayoutSelfEmployedInfo|array|null $self_employed = null) : self
```

**Summary**

Устанавливает данные самозанятого, который получит выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payouts\PayoutSelfEmployedInfo OR array OR null</code> | self_employed  | Данные самозанятого, который получит выплату |

**Returns:** self - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации
Является алиасом метода AbstractObject::jsonSerialize().

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_validate" class="anchor"></a>
#### public validate() : bool

```php
public validate() : bool
```

**Summary**

Проверяет на валидность текущий объект

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequest](../classes/YooKassa-Request-Payouts-CreatePayoutRequest.md)

**Returns:** bool - True если объект запроса валиден, false если нет


<a name="method_getUnknownProperties" class="anchor"></a>
#### protected getUnknownProperties() : array

```php
protected getUnknownProperties() : array
```

**Summary**

Возвращает массив свойств которые не существуют, но были заданы у объекта.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив с не существующими у текущего объекта свойствами


<a name="method_setValidationError" class="anchor"></a>
#### protected setValidationError() : void

```php
protected setValidationError(string $value) : void
```

**Summary**

Устанавливает ошибку валидации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | Ошибка, произошедшая при валидации объекта |

**Returns:** void - 


<a name="method_validatePropertyValue" class="anchor"></a>
#### protected validatePropertyValue() : mixed

```php
protected validatePropertyValue(string $propertyName, mixed $propertyValue) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  |  |
| <code lang="php">mixed</code> | propertyValue  |  |

**Returns:** mixed - 



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