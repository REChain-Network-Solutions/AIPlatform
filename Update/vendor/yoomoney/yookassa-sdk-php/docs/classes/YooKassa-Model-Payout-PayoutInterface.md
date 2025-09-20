# [YooKassa API SDK](../home.md)

# Interface: PayoutInterface
### Namespace: [\YooKassa\Model\Payout](../namespaces/yookassa-model-payout.md)
---
**Summary:**

Interface DealInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAmount()](../classes/YooKassa-Model-Payout-PayoutInterface.md#method_getAmount) |  | Возвращает баланс сделки. |
| public | [getCancellationDetails()](../classes/YooKassa-Model-Payout-PayoutInterface.md#method_getCancellationDetails) |  | Возвращает комментарий к статусу canceled: кто отменил выплату и по какой причине. |
| public | [getCreatedAt()](../classes/YooKassa-Model-Payout-PayoutInterface.md#method_getCreatedAt) |  | Возвращает время создания выплаты. |
| public | [getDeal()](../classes/YooKassa-Model-Payout-PayoutInterface.md#method_getDeal) |  | Возвращает сделку, в рамках которой нужно провести выплату. |
| public | [getDescription()](../classes/YooKassa-Model-Payout-PayoutInterface.md#method_getDescription) |  | Возвращает описание транзакции (не более 128 символов). |
| public | [getId()](../classes/YooKassa-Model-Payout-PayoutInterface.md#method_getId) |  | Возвращает Id сделки. |
| public | [getMetadata()](../classes/YooKassa-Model-Payout-PayoutInterface.md#method_getMetadata) |  | Возвращает дополнительные данные сделки. |
| public | [getPayoutDestination()](../classes/YooKassa-Model-Payout-PayoutInterface.md#method_getPayoutDestination) |  | Возвращает платежное средство, на которое ЮKassa зачисляет выплату. |
| public | [getStatus()](../classes/YooKassa-Model-Payout-PayoutInterface.md#method_getStatus) |  | Возвращает статус сделки. |
| public | [getSucceededAt()](../classes/YooKassa-Model-Payout-PayoutInterface.md#method_getSucceededAt) |  | Возвращает время успешного проведения выплаты. |
| public | [getTest()](../classes/YooKassa-Model-Payout-PayoutInterface.md#method_getTest) |  | Возвращает признак тестовой операции. |

---
### Details
* File: [lib/Model/Payout/PayoutInterface.php](../../lib/Model/Payout/PayoutInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |
| property |  | Идентификатор выплаты |
| property |  | Сумма выплаты |
| property |  | Текущее состояние выплаты |
| property |  | Способ проведения выплаты |
| property |  | Способ проведения выплаты |
| property |  | Описание транзакции |
| property |  | Время создания заказа |
| property |  | Время создания заказа |
| property |  | Время успешного проведения выплаты |
| property |  | Время успешного проведения выплаты |
| property |  | Сделка, в рамках которой нужно провести выплату |
| property |  | Комментарий к отмене выплаты |
| property |  | Комментарий к отмене выплаты |
| property |  | Метаданные платежа указанные мерчантом |
| property |  | Признак тестовой операции |

---
## Methods
<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает Id сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutInterface](../classes/YooKassa-Model-Payout-PayoutInterface.md)

**Returns:** string|null - Id сделки


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает баланс сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutInterface](../classes/YooKassa-Model-Payout-PayoutInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Баланс сделки


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает статус сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutInterface](../classes/YooKassa-Model-Payout-PayoutInterface.md)

**Returns:** string|null - Статус сделки


<a name="method_getPayoutDestination" class="anchor"></a>
#### public getPayoutDestination() : \YooKassa\Model\Payout\AbstractPayoutDestination|null

```php
public getPayoutDestination() : \YooKassa\Model\Payout\AbstractPayoutDestination|null
```

**Summary**

Возвращает платежное средство, на которое ЮKassa зачисляет выплату.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutInterface](../classes/YooKassa-Model-Payout-PayoutInterface.md)

**Returns:** \YooKassa\Model\Payout\AbstractPayoutDestination|null - Платежное средство, на которое ЮKassa зачисляет выплату.


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает описание транзакции (не более 128 символов).

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutInterface](../classes/YooKassa-Model-Payout-PayoutInterface.md)

**Returns:** string|null - Описание транзакции


<a name="method_getCreatedAt" class="anchor"></a>
#### public getCreatedAt() : \DateTime|null

```php
public getCreatedAt() : \DateTime|null
```

**Summary**

Возвращает время создания выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutInterface](../classes/YooKassa-Model-Payout-PayoutInterface.md)

**Returns:** \DateTime|null - Время создания сделки


<a name="method_getSucceededAt" class="anchor"></a>
#### public getSucceededAt() : \DateTime|null

```php
public getSucceededAt() : \DateTime|null
```

**Summary**

Возвращает время успешного проведения выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutInterface](../classes/YooKassa-Model-Payout-PayoutInterface.md)

**Returns:** \DateTime|null - Время создания выплаты


<a name="method_getDeal" class="anchor"></a>
#### public getDeal() : \YooKassa\Model\Deal\PayoutDealInfo|null

```php
public getDeal() : \YooKassa\Model\Deal\PayoutDealInfo|null
```

**Summary**

Возвращает сделку, в рамках которой нужно провести выплату.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutInterface](../classes/YooKassa-Model-Payout-PayoutInterface.md)

**Returns:** \YooKassa\Model\Deal\PayoutDealInfo|null - Сделка, в рамках которой нужно провести выплату


<a name="method_getCancellationDetails" class="anchor"></a>
#### public getCancellationDetails() : null|\YooKassa\Model\Payout\PayoutCancellationDetails

```php
public getCancellationDetails() : null|\YooKassa\Model\Payout\PayoutCancellationDetails
```

**Summary**

Возвращает комментарий к статусу canceled: кто отменил выплату и по какой причине.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutInterface](../classes/YooKassa-Model-Payout-PayoutInterface.md)

**Returns:** null|\YooKassa\Model\Payout\PayoutCancellationDetails - Комментарий к статусу canceled: кто отменил выплату и по какой причине


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает дополнительные данные сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutInterface](../classes/YooKassa-Model-Payout-PayoutInterface.md)

**Returns:** \YooKassa\Model\Metadata|null - Дополнительные данные сделки


<a name="method_getTest" class="anchor"></a>
#### public getTest() : bool|null

```php
public getTest() : bool|null
```

**Summary**

Возвращает признак тестовой операции.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutInterface](../classes/YooKassa-Model-Payout-PayoutInterface.md)

**Returns:** bool|null - Признак тестовой операции




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