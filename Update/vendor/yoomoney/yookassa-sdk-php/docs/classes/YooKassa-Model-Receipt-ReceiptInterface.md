# [YooKassa API SDK](../home.md)

# Interface: ReceiptInterface
### Namespace: [\YooKassa\Model\Receipt](../namespaces/yookassa-model-receipt.md)
---
**Summary:**

Interface ReceiptInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getCustomer()](../classes/YooKassa-Model-Receipt-ReceiptInterface.md#method_getCustomer) |  | Возвращает информацию о плательщике. |
| public | [getItems()](../classes/YooKassa-Model-Receipt-ReceiptInterface.md#method_getItems) |  | Возвращает список позиций в текущем чеке. |
| public | [getObjectId()](../classes/YooKassa-Model-Receipt-ReceiptInterface.md#method_getObjectId) |  | Возвращает Id объекта чека. |
| public | [getSettlements()](../classes/YooKassa-Model-Receipt-ReceiptInterface.md#method_getSettlements) |  | Возвращает массив оплат, обеспечивающих выдачу товара. |
| public | [getTaxSystemCode()](../classes/YooKassa-Model-Receipt-ReceiptInterface.md#method_getTaxSystemCode) |  | Возвращает код системы налогообложения. |
| public | [normalize()](../classes/YooKassa-Model-Receipt-ReceiptInterface.md#method_normalize) |  | Подгоняет стоимость товаров в чеке к общей цене заказа. |
| public | [notEmpty()](../classes/YooKassa-Model-Receipt-ReceiptInterface.md#method_notEmpty) |  | Проверяет есть ли в чеке хотя бы одна позиция. |

---
### Details
* File: [lib/Model/Receipt/ReceiptInterface.php](../../lib/Model/Receipt/ReceiptInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |
| property |  | Информация о плательщике |
| property |  | Список товаров в заказе |
| property |  | Код системы налогообложения. Число 1-6. |
| property |  | Код системы налогообложения. Число 1-6. |

---
## Methods
<a name="method_getObjectId" class="anchor"></a>
#### public getObjectId() : null|string

```php
public getObjectId() : null|string
```

**Summary**

Возвращает Id объекта чека.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptInterface](../classes/YooKassa-Model-Receipt-ReceiptInterface.md)

**Returns:** null|string - Id объекта чека


<a name="method_getCustomer" class="anchor"></a>
#### public getCustomer() : \YooKassa\Model\Receipt\ReceiptCustomerInterface

```php
public getCustomer() : \YooKassa\Model\Receipt\ReceiptCustomerInterface
```

**Summary**

Возвращает информацию о плательщике.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptInterface](../classes/YooKassa-Model-Receipt-ReceiptInterface.md)

**Returns:** \YooKassa\Model\Receipt\ReceiptCustomerInterface - Информация о плательщике


<a name="method_getItems" class="anchor"></a>
#### public getItems() : \YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getItems() : \YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает список позиций в текущем чеке.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptInterface](../classes/YooKassa-Model-Receipt-ReceiptInterface.md)

**Returns:** \YooKassa\Model\Receipt\ReceiptItemInterface[]|\YooKassa\Common\ListObjectInterface - Список товаров в заказе


<a name="method_getSettlements" class="anchor"></a>
#### public getSettlements() : \YooKassa\Model\Receipt\SettlementInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getSettlements() : \YooKassa\Model\Receipt\SettlementInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает массив оплат, обеспечивающих выдачу товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptInterface](../classes/YooKassa-Model-Receipt-ReceiptInterface.md)

**Returns:** \YooKassa\Model\Receipt\SettlementInterface[]|\YooKassa\Common\ListObjectInterface - Массив оплат, обеспечивающих выдачу товара


<a name="method_getTaxSystemCode" class="anchor"></a>
#### public getTaxSystemCode() : int|null

```php
public getTaxSystemCode() : int|null
```

**Summary**

Возвращает код системы налогообложения.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptInterface](../classes/YooKassa-Model-Receipt-ReceiptInterface.md)

**Returns:** int|null - Код системы налогообложения. Число 1-6.


<a name="method_notEmpty" class="anchor"></a>
#### public notEmpty() : bool

```php
public notEmpty() : bool
```

**Summary**

Проверяет есть ли в чеке хотя бы одна позиция.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptInterface](../classes/YooKassa-Model-Receipt-ReceiptInterface.md)

**Returns:** bool - True если чек не пуст, false если в чеке нет ни одной позиции


<a name="method_normalize" class="anchor"></a>
#### public normalize() : mixed

```php
public normalize(\YooKassa\Model\AmountInterface $orderAmount, bool $withShipping = false) : mixed
```

**Summary**

Подгоняет стоимость товаров в чеке к общей цене заказа.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptInterface](../classes/YooKassa-Model-Receipt-ReceiptInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface</code> | orderAmount  | Общая стоимость заказа |
| <code lang="php">bool</code> | withShipping  | Поменять ли заодно и цену доставки |

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