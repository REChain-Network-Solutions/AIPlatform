# [YooKassa API SDK](../home.md)

# Interface: CreateCaptureRequestInterface
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Interface CreateCaptureRequestInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAmount()](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md#method_getAmount) |  | Возвращает подтверждаемую сумму оплаты. |
| public | [getDeal()](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md#method_getDeal) |  | Возвращает данные о сделке. |
| public | [getReceipt()](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md#method_getReceipt) |  | Возвращает чек, если он есть. |
| public | [getTransfers()](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md#method_getTransfers) |  | Возвращает данные о распределении денег. |
| public | [hasAmount()](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md#method_hasAmount) |  | Проверяет, была ли установлена сумма оплаты. |
| public | [hasDeal()](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md#method_hasDeal) |  | Проверяет наличие данных о сделке. |
| public | [hasReceipt()](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md#method_hasReceipt) |  | Проверяет наличие чека в создаваемом платеже. |
| public | [hasTransfers()](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md#method_hasTransfers) |  | Проверяет наличие данных о распределении денег. |
| public | [setAmount()](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md#method_setAmount) |  | Устанавливает сумму оплаты. |
| public | [setDeal()](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md#method_setDeal) |  | Устанавливает данные о сделке. |
| public | [setReceipt()](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md#method_setReceipt) |  | Устанавливает чек. |
| public | [setTransfers()](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md#method_setTransfers) |  | Устанавливает transfers (массив распределения денег между магазинами). |

---
### Details
* File: [lib/Request/Payments/CreateCaptureRequestInterface.php](../../lib/Request/Payments/CreateCaptureRequestInterface.php)
* Package: \YooKassa\Request
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Подтверждаемая сумма оплаты |
| property |  | Данные фискального чека 54-ФЗ |
| property |  | Данные о сделке, в составе которой проходит платеж |

---
## Methods
<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает подтверждаемую сумму оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestInterface](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Подтверждаемая сумма оплаты


<a name="method_hasAmount" class="anchor"></a>
#### public hasAmount() : bool

```php
public hasAmount() : bool
```

**Summary**

Проверяет, была ли установлена сумма оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestInterface](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md)

**Returns:** bool - True если сумма оплаты была установлена, false если нет


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : mixed

```php
public setAmount(\YooKassa\Model\AmountInterface|array|string $amount) : mixed
```

**Summary**

Устанавливает сумму оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestInterface](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR string</code> | amount  | Сумма оплаты |

**Returns:** mixed - 


<a name="method_getReceipt" class="anchor"></a>
#### public getReceipt() : null|\YooKassa\Model\Receipt\ReceiptInterface

```php
public getReceipt() : null|\YooKassa\Model\Receipt\ReceiptInterface
```

**Summary**

Возвращает чек, если он есть.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestInterface](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md)

**Returns:** null|\YooKassa\Model\Receipt\ReceiptInterface - Данные фискального чека 54-ФЗ или null, если чека нет


<a name="method_hasReceipt" class="anchor"></a>
#### public hasReceipt() : bool

```php
public hasReceipt() : bool
```

**Summary**

Проверяет наличие чека в создаваемом платеже.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestInterface](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md)

**Returns:** bool - True если чек есть, false если нет


<a name="method_setReceipt" class="anchor"></a>
#### public setReceipt() : \YooKassa\Common\AbstractRequestInterface

```php
public setReceipt(null|\YooKassa\Model\Receipt\ReceiptInterface $value) : \YooKassa\Common\AbstractRequestInterface
```

**Summary**

Устанавливает чек.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestInterface](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \YooKassa\Model\Receipt\ReceiptInterface</code> | value  | Инстанс чека или null для удаления информации о чеке |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если передан не инстанс класса чека и не null |

**Returns:** \YooKassa\Common\AbstractRequestInterface - 


<a name="method_hasTransfers" class="anchor"></a>
#### public hasTransfers() : bool

```php
public hasTransfers() : bool
```

**Summary**

Проверяет наличие данных о распределении денег.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestInterface](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md)

**Returns:** bool - 


<a name="method_getTransfers" class="anchor"></a>
#### public getTransfers() : \YooKassa\Request\Payments\TransferDataInterface[]|\YooKassa\Common\ListObjectInterface|null

```php
public getTransfers() : \YooKassa\Request\Payments\TransferDataInterface[]|\YooKassa\Common\ListObjectInterface|null
```

**Summary**

Возвращает данные о распределении денег.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestInterface](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md)

**Returns:** \YooKassa\Request\Payments\TransferDataInterface[]|\YooKassa\Common\ListObjectInterface|null - 


<a name="method_setTransfers" class="anchor"></a>
#### public setTransfers() : self

```php
public setTransfers(null|array|\YooKassa\Request\Payments\TransferDataInterface[] $transfers = null) : self
```

**Summary**

Устанавливает transfers (массив распределения денег между магазинами).

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestInterface](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Payments\TransferDataInterface[]</code> | transfers  |  |

**Returns:** self - 


<a name="method_hasDeal" class="anchor"></a>
#### public hasDeal() : bool

```php
public hasDeal() : bool
```

**Summary**

Проверяет наличие данных о сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestInterface](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md)

**Returns:** bool - 


<a name="method_getDeal" class="anchor"></a>
#### public getDeal() : ?\YooKassa\Model\Deal\CaptureDealData

```php
public getDeal() : ?\YooKassa\Model\Deal\CaptureDealData
```

**Summary**

Возвращает данные о сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestInterface](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md)

**Returns:** ?\YooKassa\Model\Deal\CaptureDealData - 


<a name="method_setDeal" class="anchor"></a>
#### public setDeal() : \YooKassa\Common\AbstractRequestInterface

```php
public setDeal(null|array|\YooKassa\Model\Deal\CaptureDealData $deal) : \YooKassa\Common\AbstractRequestInterface
```

**Summary**

Устанавливает данные о сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestInterface](../classes/YooKassa-Request-Payments-CreateCaptureRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Deal\CaptureDealData</code> | deal  |  |

**Returns:** \YooKassa\Common\AbstractRequestInterface - 




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