# [YooKassa API SDK](../home.md)

# Interface: RefundInterface
### Namespace: [\YooKassa\Model\Refund](../namespaces/yookassa-model-refund.md)
---
**Summary:**

Interface RefundInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAmount()](../classes/YooKassa-Model-Refund-RefundInterface.md#method_getAmount) |  | Возвращает сумму возврата. |
| public | [getCreatedAt()](../classes/YooKassa-Model-Refund-RefundInterface.md#method_getCreatedAt) |  | Возвращает дату создания возврата. |
| public | [getDeal()](../classes/YooKassa-Model-Refund-RefundInterface.md#method_getDeal) |  | Возвращает сделку, в рамках которой нужно провести возврат. |
| public | [getDescription()](../classes/YooKassa-Model-Refund-RefundInterface.md#method_getDescription) |  | Возвращает комментарий к возврату. |
| public | [getId()](../classes/YooKassa-Model-Refund-RefundInterface.md#method_getId) |  | Возвращает идентификатор возврата платежа. |
| public | [getPaymentId()](../classes/YooKassa-Model-Refund-RefundInterface.md#method_getPaymentId) |  | Возвращает идентификатор платежа. |
| public | [getReceiptRegistration()](../classes/YooKassa-Model-Refund-RefundInterface.md#method_getReceiptRegistration) |  | Возвращает статус регистрации чека. |
| public | [getSources()](../classes/YooKassa-Model-Refund-RefundInterface.md#method_getSources) |  | Возвращает информацию о распределении денег — сколько и в какой магазин нужно перевести. |
| public | [getStatus()](../classes/YooKassa-Model-Refund-RefundInterface.md#method_getStatus) |  | Возвращает статус текущего возврата. |

---
### Details
* File: [lib/Model/Refund/RefundInterface.php](../../lib/Model/Refund/RefundInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Идентификатор возврата платежа |
| property |  | Идентификатор платежа |
| property |  | Идентификатор платежа |
| property |  | Статус возврата |
| property |  | Комментарий к статусу `canceled` |
| property |  | Комментарий к статусу `canceled` |
| property |  | Время создания возврата |
| property |  | Время создания возврата |
| property |  | Сумма возврата |
| property |  | Статус регистрации чека |
| property |  | Статус регистрации чека |
| property |  | Комментарий, основание для возврата средств покупателю |
| property |  | Данные о том, с какого магазина и какую сумму нужно удержать для проведения возврата |
| property |  | Данные о сделке, в составе которой проходит возврат |

---
## Methods
<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает идентификатор возврата платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundInterface](../classes/YooKassa-Model-Refund-RefundInterface.md)

**Returns:** string|null - Идентификатор возврата


<a name="method_getPaymentId" class="anchor"></a>
#### public getPaymentId() : string|null

```php
public getPaymentId() : string|null
```

**Summary**

Возвращает идентификатор платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundInterface](../classes/YooKassa-Model-Refund-RefundInterface.md)

**Returns:** string|null - Идентификатор платежа


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает статус текущего возврата.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundInterface](../classes/YooKassa-Model-Refund-RefundInterface.md)

**Returns:** string|null - Статус возврата


<a name="method_getCreatedAt" class="anchor"></a>
#### public getCreatedAt() : \DateTime|null

```php
public getCreatedAt() : \DateTime|null
```

**Summary**

Возвращает дату создания возврата.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundInterface](../classes/YooKassa-Model-Refund-RefundInterface.md)

**Returns:** \DateTime|null - Время создания возврата


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму возврата.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundInterface](../classes/YooKassa-Model-Refund-RefundInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма возврата


<a name="method_getReceiptRegistration" class="anchor"></a>
#### public getReceiptRegistration() : string|null

```php
public getReceiptRegistration() : string|null
```

**Summary**

Возвращает статус регистрации чека.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundInterface](../classes/YooKassa-Model-Refund-RefundInterface.md)

**Returns:** string|null - Статус регистрации чека


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает комментарий к возврату.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundInterface](../classes/YooKassa-Model-Refund-RefundInterface.md)

**Returns:** string|null - Комментарий, основание для возврата средств покупателю


<a name="method_getSources" class="anchor"></a>
#### public getSources() : \YooKassa\Model\Refund\SourceInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getSources() : \YooKassa\Model\Refund\SourceInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает информацию о распределении денег — сколько и в какой магазин нужно перевести.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundInterface](../classes/YooKassa-Model-Refund-RefundInterface.md)

**Returns:** \YooKassa\Model\Refund\SourceInterface[]|\YooKassa\Common\ListObjectInterface - 


<a name="method_getDeal" class="anchor"></a>
#### public getDeal() : null|\YooKassa\Model\Deal\RefundDealInfo

```php
public getDeal() : null|\YooKassa\Model\Deal\RefundDealInfo
```

**Summary**

Возвращает сделку, в рамках которой нужно провести возврат.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundInterface](../classes/YooKassa-Model-Refund-RefundInterface.md)

**Returns:** null|\YooKassa\Model\Deal\RefundDealInfo - Сделка, в рамках которой нужно провести возврат




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