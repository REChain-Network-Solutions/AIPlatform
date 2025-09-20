# [YooKassa API SDK](../home.md)

# Interface: VatDataInterface
### Namespace: [\YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank](../namespaces/yookassa-model-payment-paymentmethod-b2b-sberbank.md)
---
**Summary:**

Interface VatDataInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAmount()](../classes/YooKassa-Model-Payment-PaymentMethod-B2b-Sberbank-VatDataInterface.md#method_getAmount) |  | Возвращает сумму НДС |
| public | [getRate()](../classes/YooKassa-Model-Payment-PaymentMethod-B2b-Sberbank-VatDataInterface.md#method_getRate) |  | Возвращает данные об НДС |
| public | [getType()](../classes/YooKassa-Model-Payment-PaymentMethod-B2b-Sberbank-VatDataInterface.md#method_getType) |  | Возвращает способ расчёта НДС |

---
### Details
* File: [lib/Model/Payment/PaymentMethod/B2b/Sberbank/VatDataInterface.php](../../lib/Model/Payment/PaymentMethod/B2b/Sberbank/VatDataInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Способ расчёта НДС |
| property |  | Данные об НДС в случае, если сумма НДС включена в сумму платежа |
| property |  | Сумма НДС |

---
## Methods
<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает способ расчёта НДС

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataInterface](../classes/YooKassa-Model-Payment-PaymentMethod-B2b-Sberbank-VatDataInterface.md)

**Returns:** string|null - Способ расчёта НДС


<a name="method_getRate" class="anchor"></a>
#### public getRate() : string|null

```php
public getRate() : string|null
```

**Summary**

Возвращает данные об НДС

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataInterface](../classes/YooKassa-Model-Payment-PaymentMethod-B2b-Sberbank-VatDataInterface.md)

**Returns:** string|null - Данные об НДС


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму НДС

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataInterface](../classes/YooKassa-Model-Payment-PaymentMethod-B2b-Sberbank-VatDataInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма НДС




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