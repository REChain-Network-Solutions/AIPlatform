# [YooKassa API SDK](../home.md)

# Interface: CreateRefundRequestInterface
### Namespace: [\YooKassa\Request\Refunds](../namespaces/yookassa-request-refunds.md)
---
**Summary:**

Interface CreateRefundRequestInterface

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAmount()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_getAmount) |  | Возвращает сумму возвращаемых средств. |
| public | [getDeal()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_getDeal) |  | Возвращает информацию о сделке. |
| public | [getDescription()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_getDescription) |  | Возвращает комментарий к возврату или null, если комментарий не задан |
| public | [getPaymentId()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_getPaymentId) |  | Возвращает айди платежа для которого создаётся возврат средств. |
| public | [getReceipt()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_getReceipt) |  | Возвращает инстанс чека или null, если чек не задан. |
| public | [getSources()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_getSources) |  | Возвращает информацию о распределении денег — сколько и в какой магазин нужно перевести. |
| public | [hasDeal()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_hasDeal) |  | Проверяет наличие информации о сделке. |
| public | [hasDescription()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_hasDescription) |  | Проверяет задан ли комментарий к создаваемому возврату. |
| public | [hasPaymentId()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_hasPaymentId) |  | Проверяет, был ли установлена идентификатор платежа. |
| public | [hasReceipt()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_hasReceipt) |  | Проверяет задан ли чек. |
| public | [hasSources()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_hasSources) |  | Проверяет наличие информации о распределении денег. |
| public | [setDeal()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_setDeal) |  | Устанавливает информацию о сделке. |
| public | [setDescription()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_setDescription) |  | Устанавливает комментарий к возврату. |
| public | [setReceipt()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_setReceipt) |  | Устанавливает чек. |
| public | [setSources()](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md#method_setSources) |  | Устанавливает информацию о распределении денег — сколько и в какой магазин нужно перевести. |

---
### Details
* File: [lib/Request/Refunds/CreateRefundRequestInterface.php](../../lib/Request/Refunds/CreateRefundRequestInterface.php)
* Package: \YooKassa\Request
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Айди платежа для которого создаётся возврат |
| property |  | Сумма возврата |
| property |  | Комментарий к операции возврата, основание для возврата средств покупателю. |
| property |  | Инстанс чека или null |
| property |  | Информация о распределении денег — сколько и в какой магазин нужно перевести |
| property |  | Информация о сделке |

---
## Methods
<a name="method_getPaymentId" class="anchor"></a>
#### public getPaymentId() : string|null

```php
public getPaymentId() : string|null
```

**Summary**

Возвращает айди платежа для которого создаётся возврат средств.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

**Returns:** string|null - Айди платежа для которого создаётся возврат


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму возвращаемых средств.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма возврата


<a name="method_hasPaymentId" class="anchor"></a>
#### public hasPaymentId() : bool

```php
public hasPaymentId() : bool
```

**Summary**

Проверяет, был ли установлена идентификатор платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

**Returns:** bool - True если идентификатор платежа был установлен, false если нет


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : mixed

```php
public setDescription(string|null $description) : mixed
```

**Summary**

Устанавливает комментарий к возврату.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Комментарий к операции возврата, основание для возврата средств покупателю |

**Returns:** mixed - 


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает комментарий к возврату или null, если комментарий не задан

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

**Returns:** string|null - Комментарий к операции возврата, основание для возврата средств покупателю.


<a name="method_hasDescription" class="anchor"></a>
#### public hasDescription() : bool

```php
public hasDescription() : bool
```

**Summary**

Проверяет задан ли комментарий к создаваемому возврату.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

**Returns:** bool - True если комментарий установлен, false если нет


<a name="method_setReceipt" class="anchor"></a>
#### public setReceipt() : mixed

```php
public setReceipt(null|\YooKassa\Model\Receipt\ReceiptInterface $receipt) : mixed
```

**Summary**

Устанавливает чек.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \YooKassa\Model\Receipt\ReceiptInterface</code> | receipt  | Инстанс чека или null для удаления информации о чеке |

**Returns:** mixed - 


<a name="method_getReceipt" class="anchor"></a>
#### public getReceipt() : null|\YooKassa\Model\Receipt\ReceiptInterface

```php
public getReceipt() : null|\YooKassa\Model\Receipt\ReceiptInterface
```

**Summary**

Возвращает инстанс чека или null, если чек не задан.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

**Returns:** null|\YooKassa\Model\Receipt\ReceiptInterface - Инстанс чека или null


<a name="method_hasReceipt" class="anchor"></a>
#### public hasReceipt() : bool

```php
public hasReceipt() : bool
```

**Summary**

Проверяет задан ли чек.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

**Returns:** bool - True если чек есть, false если нет


<a name="method_setSources" class="anchor"></a>
#### public setSources() : mixed

```php
public setSources(\YooKassa\Model\Refund\SourceInterface[]|array|null $sources) : mixed
```

**Summary**

Устанавливает информацию о распределении денег — сколько и в какой магазин нужно перевести.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Refund\SourceInterface[] OR array OR null</code> | sources  | Информация о распределении денег |

**Returns:** mixed - 


<a name="method_getSources" class="anchor"></a>
#### public getSources() : \YooKassa\Model\Refund\SourceInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getSources() : \YooKassa\Model\Refund\SourceInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает информацию о распределении денег — сколько и в какой магазин нужно перевести.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

**Returns:** \YooKassa\Model\Refund\SourceInterface[]|\YooKassa\Common\ListObjectInterface - Информация о распределении денег


<a name="method_hasSources" class="anchor"></a>
#### public hasSources() : bool

```php
public hasSources() : bool
```

**Summary**

Проверяет наличие информации о распределении денег.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

**Returns:** bool - 


<a name="method_setDeal" class="anchor"></a>
#### public setDeal() : mixed

```php
public setDeal(\YooKassa\Model\Deal\RefundDealData|null $deal) : mixed
```

**Summary**

Устанавливает информацию о сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Deal\RefundDealData OR null</code> | deal  | Информация о сделке |

**Returns:** mixed - 


<a name="method_getDeal" class="anchor"></a>
#### public getDeal() : \YooKassa\Model\Deal\RefundDealData|null

```php
public getDeal() : \YooKassa\Model\Deal\RefundDealData|null
```

**Summary**

Возвращает информацию о сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

**Returns:** \YooKassa\Model\Deal\RefundDealData|null - Информация о сделке


<a name="method_hasDeal" class="anchor"></a>
#### public hasDeal() : bool

```php
public hasDeal() : bool
```

**Summary**

Проверяет наличие информации о сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\CreateRefundRequestInterface](../classes/YooKassa-Request-Refunds-CreateRefundRequestInterface.md)

**Returns:** bool - 




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