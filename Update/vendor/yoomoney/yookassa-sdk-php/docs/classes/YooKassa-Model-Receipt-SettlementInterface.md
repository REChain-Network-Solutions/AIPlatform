# [YooKassa API SDK](../home.md)

# Interface: SettlementInterface
### Namespace: [\YooKassa\Model\Receipt](../namespaces/yookassa-model-receipt.md)
---
**Summary:**

Interface PostReceiptResponseSettlementInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAmount()](../classes/YooKassa-Model-Receipt-SettlementInterface.md#method_getAmount) |  | Возвращает размер оплаты. |
| public | [getType()](../classes/YooKassa-Model-Receipt-SettlementInterface.md#method_getType) |  | Возвращает вид оплаты в чеке (cashless | prepayment | postpayment | consideration). |

---
### Details
* File: [lib/Model/Receipt/SettlementInterface.php](../../lib/Model/Receipt/SettlementInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |
| property |  | Вид оплаты в чеке |
| property |  | Размер оплаты |

---
## Methods
<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает вид оплаты в чеке (cashless | prepayment | postpayment | consideration).

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\SettlementInterface](../classes/YooKassa-Model-Receipt-SettlementInterface.md)

**Returns:** string|null - Вид оплаты в чеке


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает размер оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\SettlementInterface](../classes/YooKassa-Model-Receipt-SettlementInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Размер оплаты




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