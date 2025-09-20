# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Refunds\CreateRefundResponse
### Namespace: [\YooKassa\Request\Refunds](../namespaces/yookassa-request-refunds.md)
---
**Summary:**

Класс, представляющий модель CreateRefundResponse.

**Description:**

Класс объекта ответа от API при создании нового возврата.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [MAX_LENGTH_DESCRIPTION](../classes/YooKassa-Model-Refund-Refund.md#constant_MAX_LENGTH_DESCRIPTION) |  | Максимальная длина строки описания возврата |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$amount](../classes/YooKassa-Model-Refund-Refund.md#property_amount) |  | Сумма возврата |
| public | [$cancellation_details](../classes/YooKassa-Model-Refund-Refund.md#property_cancellation_details) |  | Комментарий к статусу `canceled` |
| public | [$cancellationDetails](../classes/YooKassa-Model-Refund-Refund.md#property_cancellationDetails) |  | Комментарий к статусу `canceled` |
| public | [$created_at](../classes/YooKassa-Model-Refund-Refund.md#property_created_at) |  | Время создания возврата |
| public | [$createdAt](../classes/YooKassa-Model-Refund-Refund.md#property_createdAt) |  | Время создания возврата |
| public | [$deal](../classes/YooKassa-Model-Refund-Refund.md#property_deal) |  | Данные о сделке, в составе которой проходит возврат |
| public | [$description](../classes/YooKassa-Model-Refund-Refund.md#property_description) |  | Комментарий, основание для возврата средств покупателю |
| public | [$id](../classes/YooKassa-Model-Refund-Refund.md#property_id) |  | Идентификатор возврата платежа |
| public | [$payment_id](../classes/YooKassa-Model-Refund-Refund.md#property_payment_id) |  | Идентификатор платежа |
| public | [$paymentId](../classes/YooKassa-Model-Refund-Refund.md#property_paymentId) |  | Идентификатор платежа |
| public | [$receipt_registration](../classes/YooKassa-Model-Refund-Refund.md#property_receipt_registration) |  | Статус регистрации чека |
| public | [$receiptRegistration](../classes/YooKassa-Model-Refund-Refund.md#property_receiptRegistration) |  | Статус регистрации чека |
| public | [$refund_authorization_details](../classes/YooKassa-Model-Refund-Refund.md#property_refund_authorization_details) |  | Данные об авторизации возврата. Присутствуют только для возвратов платежей, совершенных этими способами оплаты: банковская карта, Mir Pay |
| public | [$refund_method](../classes/YooKassa-Model-Refund-Refund.md#property_refund_method) |  | Детали возврата. Зависят от способа оплаты, который использовался при проведении платежа |
| public | [$refundAuthorizationDetails](../classes/YooKassa-Model-Refund-Refund.md#property_refundAuthorizationDetails) |  | Данные об авторизации возврата. Присутствуют только для возвратов платежей, совершенных этими способами оплаты: банковская карта, Mir Pay |
| public | [$refundMethod](../classes/YooKassa-Model-Refund-Refund.md#property_refundMethod) |  | Детали возврата. Зависят от способа оплаты, который использовался при проведении платежа |
| public | [$sources](../classes/YooKassa-Model-Refund-Refund.md#property_sources) |  | Данные о том, с какого магазина и какую сумму нужно удержать для проведения возврата |
| public | [$status](../classes/YooKassa-Model-Refund-Refund.md#property_status) |  | Статус возврата |
| protected | [$_amount](../classes/YooKassa-Model-Refund-Refund.md#property__amount) |  |  |
| protected | [$_cancellation_details](../classes/YooKassa-Model-Refund-Refund.md#property__cancellation_details) |  |  |
| protected | [$_created_at](../classes/YooKassa-Model-Refund-Refund.md#property__created_at) |  |  |
| protected | [$_deal](../classes/YooKassa-Model-Refund-Refund.md#property__deal) |  |  |
| protected | [$_description](../classes/YooKassa-Model-Refund-Refund.md#property__description) |  |  |
| protected | [$_id](../classes/YooKassa-Model-Refund-Refund.md#property__id) |  |  |
| protected | [$_payment_id](../classes/YooKassa-Model-Refund-Refund.md#property__payment_id) |  |  |
| protected | [$_receipt_registration](../classes/YooKassa-Model-Refund-Refund.md#property__receipt_registration) |  |  |
| protected | [$_refund_authorization_details](../classes/YooKassa-Model-Refund-Refund.md#property__refund_authorization_details) |  |  |
| protected | [$_refund_method](../classes/YooKassa-Model-Refund-Refund.md#property__refund_method) |  |  |
| protected | [$_sources](../classes/YooKassa-Model-Refund-Refund.md#property__sources) |  |  |
| protected | [$_status](../classes/YooKassa-Model-Refund-Refund.md#property__status) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAmount()](../classes/YooKassa-Model-Refund-Refund.md#method_getAmount) |  | Возвращает сумму возврата. |
| public | [getCancellationDetails()](../classes/YooKassa-Model-Refund-Refund.md#method_getCancellationDetails) |  | Возвращает cancellation_details. |
| public | [getCreatedAt()](../classes/YooKassa-Model-Refund-Refund.md#method_getCreatedAt) |  | Возвращает дату создания возврата. |
| public | [getDeal()](../classes/YooKassa-Model-Refund-Refund.md#method_getDeal) |  | Возвращает данные о сделке, в составе которой проходит возврат |
| public | [getDescription()](../classes/YooKassa-Model-Refund-Refund.md#method_getDescription) |  | Возвращает комментарий к возврату. |
| public | [getId()](../classes/YooKassa-Model-Refund-Refund.md#method_getId) |  | Возвращает идентификатор возврата платежа. |
| public | [getPaymentId()](../classes/YooKassa-Model-Refund-Refund.md#method_getPaymentId) |  | Возвращает идентификатор платежа. |
| public | [getReceiptRegistration()](../classes/YooKassa-Model-Refund-Refund.md#method_getReceiptRegistration) |  | Возвращает статус регистрации чека. |
| public | [getRefundAuthorizationDetails()](../classes/YooKassa-Model-Refund-Refund.md#method_getRefundAuthorizationDetails) |  | Возвращает refund_authorization_details. |
| public | [getRefundMethod()](../classes/YooKassa-Model-Refund-Refund.md#method_getRefundMethod) |  | Возвращает метод возврата. |
| public | [getSources()](../classes/YooKassa-Model-Refund-Refund.md#method_getSources) |  | Возвращает информацию о распределении денег — сколько и в какой магазин нужно перевести. |
| public | [getStatus()](../classes/YooKassa-Model-Refund-Refund.md#method_getStatus) |  | Возвращает статус текущего возврата. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAmount()](../classes/YooKassa-Model-Refund-Refund.md#method_setAmount) |  | Устанавливает сумму возврата. |
| public | [setCancellationDetails()](../classes/YooKassa-Model-Refund-Refund.md#method_setCancellationDetails) |  | Устанавливает cancellation_details. |
| public | [setCreatedAt()](../classes/YooKassa-Model-Refund-Refund.md#method_setCreatedAt) |  | Устанавливает время создания возврата. |
| public | [setDeal()](../classes/YooKassa-Model-Refund-Refund.md#method_setDeal) |  | Устанавливает данные о сделке, в составе которой проходит возврат. |
| public | [setDescription()](../classes/YooKassa-Model-Refund-Refund.md#method_setDescription) |  | Устанавливает комментарий к возврату. |
| public | [setId()](../classes/YooKassa-Model-Refund-Refund.md#method_setId) |  | Устанавливает идентификатор возврата. |
| public | [setPaymentId()](../classes/YooKassa-Model-Refund-Refund.md#method_setPaymentId) |  | Устанавливает идентификатор платежа. |
| public | [setReceiptRegistration()](../classes/YooKassa-Model-Refund-Refund.md#method_setReceiptRegistration) |  | Устанавливает статус регистрации чека. |
| public | [setRefundAuthorizationDetails()](../classes/YooKassa-Model-Refund-Refund.md#method_setRefundAuthorizationDetails) |  | Устанавливает refund_authorization_details. |
| public | [setRefundMethod()](../classes/YooKassa-Model-Refund-Refund.md#method_setRefundMethod) |  | Устанавливает метод возврата. |
| public | [setSources()](../classes/YooKassa-Model-Refund-Refund.md#method_setSources) |  | Устанавливает sources (массив распределения денег между магазинами). |
| public | [setStatus()](../classes/YooKassa-Model-Refund-Refund.md#method_setStatus) |  | Устанавливает статус возврата платежа. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Refunds/CreateRefundResponse.php](../../lib/Request/Refunds/CreateRefundResponse.php)
* Package: YooKassa\Request
* Class Hierarchy:   
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)
  * [\YooKassa\Request\Refunds\AbstractRefundResponse](../classes/YooKassa-Request-Refunds-AbstractRefundResponse.md)
  * \YooKassa\Request\Refunds\CreateRefundResponse

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
Inherited from [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

Максимальная длина строки описания возврата

```php
MAX_LENGTH_DESCRIPTION = 250
```



---
## Properties
<a name="property_amount"></a>
#### public $amount : \YooKassa\Model\AmountInterface
---
***Description***

Сумма возврата

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_cancellation_details"></a>
#### public $cancellation_details : \YooKassa\Model\Refund\RefundCancellationDetails
---
***Description***

Комментарий к статусу `canceled`

**Type:** <a href="../classes/YooKassa-Model-Refund-RefundCancellationDetails.html"><abbr title="\YooKassa\Model\Refund\RefundCancellationDetails">RefundCancellationDetails</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_cancellationDetails"></a>
#### public $cancellationDetails : \YooKassa\Model\Refund\RefundCancellationDetails
---
***Description***

Комментарий к статусу `canceled`

**Type:** <a href="../classes/YooKassa-Model-Refund-RefundCancellationDetails.html"><abbr title="\YooKassa\Model\Refund\RefundCancellationDetails">RefundCancellationDetails</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_created_at"></a>
#### public $created_at : \DateTime
---
***Description***

Время создания возврата

**Type:** \DateTime

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_createdAt"></a>
#### public $createdAt : \DateTime
---
***Description***

Время создания возврата

**Type:** \DateTime

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_deal"></a>
#### public $deal : \YooKassa\Model\Deal\RefundDealInfo
---
***Description***

Данные о сделке, в составе которой проходит возврат

**Type:** <a href="../classes/YooKassa-Model-Deal-RefundDealInfo.html"><abbr title="\YooKassa\Model\Deal\RefundDealInfo">RefundDealInfo</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_description"></a>
#### public $description : string
---
***Description***

Комментарий, основание для возврата средств покупателю

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_id"></a>
#### public $id : string
---
***Description***

Идентификатор возврата платежа

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_payment_id"></a>
#### public $payment_id : string
---
***Description***

Идентификатор платежа

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_paymentId"></a>
#### public $paymentId : string
---
***Description***

Идентификатор платежа

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_receipt_registration"></a>
#### public $receipt_registration : string
---
***Description***

Статус регистрации чека

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_receiptRegistration"></a>
#### public $receiptRegistration : string
---
***Description***

Статус регистрации чека

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_refund_authorization_details"></a>
#### public $refund_authorization_details : \YooKassa\Model\Refund\RefundAuthorizationDetails|null
---
***Description***

Данные об авторизации возврата. Присутствуют только для возвратов платежей, совершенных этими способами оплаты: банковская карта, Mir Pay

**Type:** <a href="../\YooKassa\Model\Refund\RefundAuthorizationDetails|null"><abbr title="\YooKassa\Model\Refund\RefundAuthorizationDetails|null">RefundAuthorizationDetails|null</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_refund_method"></a>
#### public $refund_method : \YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod|null
---
***Description***

Детали возврата. Зависят от способа оплаты, который использовался при проведении платежа

**Type:** <a href="../\YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod|null"><abbr title="\YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod|null">AbstractRefundMethod|null</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_refundAuthorizationDetails"></a>
#### public $refundAuthorizationDetails : \YooKassa\Model\Refund\RefundAuthorizationDetails|null
---
***Description***

Данные об авторизации возврата. Присутствуют только для возвратов платежей, совершенных этими способами оплаты: банковская карта, Mir Pay

**Type:** <a href="../\YooKassa\Model\Refund\RefundAuthorizationDetails|null"><abbr title="\YooKassa\Model\Refund\RefundAuthorizationDetails|null">RefundAuthorizationDetails|null</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_refundMethod"></a>
#### public $refundMethod : \YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod|null
---
***Description***

Детали возврата. Зависят от способа оплаты, который использовался при проведении платежа

**Type:** <a href="../\YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod|null"><abbr title="\YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod|null">AbstractRefundMethod|null</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_sources"></a>
#### public $sources : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Refund\SourceInterface[]
---
***Description***

Данные о том, с какого магазина и какую сумму нужно удержать для проведения возврата

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Refund\SourceInterface[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Refund\SourceInterface[]">SourceInterface[]</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property_status"></a>
#### public $status : string
---
***Description***

Статус возврата

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property__amount"></a>
#### protected $_amount : ?\YooKassa\Model\AmountInterface
---
**Type:** <a href="../?\YooKassa\Model\AmountInterface"><abbr title="?\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>
Сумма возврата
**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property__cancellation_details"></a>
#### protected $_cancellation_details : ?\YooKassa\Model\CancellationDetailsInterface
---
**Type:** <a href="../?\YooKassa\Model\CancellationDetailsInterface"><abbr title="?\YooKassa\Model\CancellationDetailsInterface">CancellationDetailsInterface</abbr></a>
Комментарий к статусу `canceled`
**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property__created_at"></a>
#### protected $_created_at : ?\DateTime
---
**Type:** <a href="../?\DateTime"><abbr title="?\DateTime">DateTime</abbr></a>
Время создания возврата
**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property__deal"></a>
#### protected $_deal : ?\YooKassa\Model\Deal\RefundDealInfo
---
**Type:** <a href="../?\YooKassa\Model\Deal\RefundDealInfo"><abbr title="?\YooKassa\Model\Deal\RefundDealInfo">RefundDealInfo</abbr></a>
Данные о сделке, в составе которой проходит возврат
**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property__description"></a>
#### protected $_description : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Комментарий, основание для возврата средств покупателю
**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property__id"></a>
#### protected $_id : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Идентификатор возврата платежа
**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property__payment_id"></a>
#### protected $_payment_id : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Идентификатор платежа
**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property__receipt_registration"></a>
#### protected $_receipt_registration : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Статус регистрации чека
**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property__refund_authorization_details"></a>
#### protected $_refund_authorization_details : ?\YooKassa\Model\Refund\RefundAuthorizationDetails
---
**Type:** <a href="../?\YooKassa\Model\Refund\RefundAuthorizationDetails"><abbr title="?\YooKassa\Model\Refund\RefundAuthorizationDetails">RefundAuthorizationDetails</abbr></a>
Данные об авторизации возврата
**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property__refund_method"></a>
#### protected $_refund_method : ?\YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod
---
**Type:** <a href="../?\YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod"><abbr title="?\YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod">AbstractRefundMethod</abbr></a>
Детали возврата. Зависят от способа оплаты, который использовался при проведении платежа.
**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property__sources"></a>
#### protected $_sources : ?\YooKassa\Common\ListObject
---
**Type:** <a href="../?\YooKassa\Common\ListObject"><abbr title="?\YooKassa\Common\ListObject">ListObject</abbr></a>
Данные о распределении денег — сколько и в какой магазин нужно перевести
**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)


<a name="property__status"></a>
#### protected $_status : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Статус возврата
**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)



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

Возвращает сумму возврата.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма возврата


<a name="method_getCancellationDetails" class="anchor"></a>
#### public getCancellationDetails() : \YooKassa\Model\CancellationDetailsInterface|null

```php
public getCancellationDetails() : \YooKassa\Model\CancellationDetailsInterface|null
```

**Summary**

Возвращает cancellation_details.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

**Returns:** \YooKassa\Model\CancellationDetailsInterface|null - 


<a name="method_getCreatedAt" class="anchor"></a>
#### public getCreatedAt() : \DateTime|null

```php
public getCreatedAt() : \DateTime|null
```

**Summary**

Возвращает дату создания возврата.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

**Returns:** \DateTime|null - Время создания возврата


<a name="method_getDeal" class="anchor"></a>
#### public getDeal() : null|\YooKassa\Model\Deal\RefundDealInfo

```php
public getDeal() : null|\YooKassa\Model\Deal\RefundDealInfo
```

**Summary**

Возвращает данные о сделке, в составе которой проходит возврат

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

**Returns:** null|\YooKassa\Model\Deal\RefundDealInfo - Данные о сделке, в составе которой проходит возврат


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает комментарий к возврату.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

**Returns:** string|null - Комментарий, основание для возврата средств покупателю


<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает идентификатор возврата платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

**Returns:** string|null - Идентификатор возврата


<a name="method_getPaymentId" class="anchor"></a>
#### public getPaymentId() : string|null

```php
public getPaymentId() : string|null
```

**Summary**

Возвращает идентификатор платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

**Returns:** string|null - Идентификатор платежа


<a name="method_getReceiptRegistration" class="anchor"></a>
#### public getReceiptRegistration() : string|null

```php
public getReceiptRegistration() : string|null
```

**Summary**

Возвращает статус регистрации чека.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

**Returns:** string|null - Статус регистрации чека


<a name="method_getRefundAuthorizationDetails" class="anchor"></a>
#### public getRefundAuthorizationDetails() : \YooKassa\Model\Refund\RefundAuthorizationDetails|null

```php
public getRefundAuthorizationDetails() : \YooKassa\Model\Refund\RefundAuthorizationDetails|null
```

**Summary**

Возвращает refund_authorization_details.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

**Returns:** \YooKassa\Model\Refund\RefundAuthorizationDetails|null - 


<a name="method_getRefundMethod" class="anchor"></a>
#### public getRefundMethod() : \YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod|null

```php
public getRefundMethod() : \YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod|null
```

**Summary**

Возвращает метод возврата.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

**Returns:** \YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod|null - 


<a name="method_getSources" class="anchor"></a>
#### public getSources() : \YooKassa\Model\Refund\SourceInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getSources() : \YooKassa\Model\Refund\SourceInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает информацию о распределении денег — сколько и в какой магазин нужно перевести.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

**Returns:** \YooKassa\Model\Refund\SourceInterface[]|\YooKassa\Common\ListObjectInterface - 


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает статус текущего возврата.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

**Returns:** string|null - Статус возврата


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


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
public setAmount(\YooKassa\Model\AmountInterface|array|null $amount = null) : self
```

**Summary**

Устанавливает сумму возврата.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | amount  | Сумма возврата |

**Returns:** self - 


<a name="method_setCancellationDetails" class="anchor"></a>
#### public setCancellationDetails() : self

```php
public setCancellationDetails(\YooKassa\Model\CancellationDetailsInterface|array|null $cancellation_details = null) : self
```

**Summary**

Устанавливает cancellation_details.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\CancellationDetailsInterface OR array OR null</code> | cancellation_details  |  |

**Returns:** self - 


<a name="method_setCreatedAt" class="anchor"></a>
#### public setCreatedAt() : self

```php
public setCreatedAt(\DateTime|string|null $created_at = null) : self
```

**Summary**

Устанавливает время создания возврата.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | created_at  | Время создания возврата |

**Returns:** self - 


<a name="method_setDeal" class="anchor"></a>
#### public setDeal() : self

```php
public setDeal(\YooKassa\Model\Deal\RefundDealInfo|array|null $deal = null) : self
```

**Summary**

Устанавливает данные о сделке, в составе которой проходит возврат.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Deal\RefundDealInfo OR array OR null</code> | deal  | Данные о сделке, в составе которой проходит возврат |

**Returns:** self - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description = null) : self
```

**Summary**

Устанавливает комментарий к возврату.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Комментарий, основание для возврата средств покупателю |

**Returns:** self - 


<a name="method_setId" class="anchor"></a>
#### public setId() : self

```php
public setId(string|null $id = null) : self
```

**Summary**

Устанавливает идентификатор возврата.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | id  | Идентификатор возврата |

**Returns:** self - 


<a name="method_setPaymentId" class="anchor"></a>
#### public setPaymentId() : self

```php
public setPaymentId(string|null $payment_id = null) : self
```

**Summary**

Устанавливает идентификатор платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | payment_id  | Идентификатор платежа |

**Returns:** self - 


<a name="method_setReceiptRegistration" class="anchor"></a>
#### public setReceiptRegistration() : self

```php
public setReceiptRegistration(string|null $receipt_registration = null) : self
```

**Summary**

Устанавливает статус регистрации чека.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | receipt_registration  | Статус регистрации чека |

**Returns:** self - 


<a name="method_setRefundAuthorizationDetails" class="anchor"></a>
#### public setRefundAuthorizationDetails() : self

```php
public setRefundAuthorizationDetails(\YooKassa\Model\Refund\RefundAuthorizationDetails|array|null $refund_authorization_details = null) : self
```

**Summary**

Устанавливает refund_authorization_details.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Refund\RefundAuthorizationDetails OR array OR null</code> | refund_authorization_details  | Данные об авторизации возврата |

**Returns:** self - 


<a name="method_setRefundMethod" class="anchor"></a>
#### public setRefundMethod() : self

```php
public setRefundMethod(\YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod|array|null $refund_method = null) : self
```

**Summary**

Устанавливает метод возврата.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod OR array OR null</code> | refund_method  |  |

**Returns:** self - 


<a name="method_setSources" class="anchor"></a>
#### public setSources() : self

```php
public setSources(\YooKassa\Common\ListObjectInterface|array|null $sources = null) : self
```

**Summary**

Устанавливает sources (массив распределения денег между магазинами).

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR array OR null</code> | sources  |  |

**Returns:** self - 


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : self

```php
public setStatus(string|null $status = null) : self
```

**Summary**

Устанавливает статус возврата платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Refund](../classes/YooKassa-Model-Refund-Refund.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | status  | Статус возврата платежа |

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