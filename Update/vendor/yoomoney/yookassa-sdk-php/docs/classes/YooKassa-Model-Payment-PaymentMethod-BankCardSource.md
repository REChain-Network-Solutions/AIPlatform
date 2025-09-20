# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payment\PaymentMethod\BankCardSource
### Namespace: [\YooKassa\Model\Payment\PaymentMethod](../namespaces/yookassa-model-payment-paymentmethod.md)
---
**Summary:**

Класс, представляющий модель BankCardSource.

**Description:**

Источник данных банковской карты.

Возможные значения:
- apple_pay - Источник данных ApplePay
- google_pay - Источник данных GooglePay
- mir_pay - Источник данных MirPay

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [APPLE_PAY](../classes/YooKassa-Model-Payment-PaymentMethod-BankCardSource.md#constant_APPLE_PAY) |  | Источник данных ApplePay |
| public | [GOOGLE_PAY](../classes/YooKassa-Model-Payment-PaymentMethod-BankCardSource.md#constant_GOOGLE_PAY) |  | Источник данных GooglePay |
| public | [MIR_PAY](../classes/YooKassa-Model-Payment-PaymentMethod-BankCardSource.md#constant_MIR_PAY) |  | Источник данных MirPay |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-Payment-PaymentMethod-BankCardSource.md#property_validValues) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/Payment/PaymentMethod/BankCardSource.php](../../lib/Model/Payment/PaymentMethod/BankCardSource.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\Payment\PaymentMethod\BankCardSource

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
<a name="constant_APPLE_PAY" class="anchor"></a>
###### APPLE_PAY
Источник данных ApplePay

```php
APPLE_PAY = 'apple_pay'
```


<a name="constant_GOOGLE_PAY" class="anchor"></a>
###### GOOGLE_PAY
Источник данных GooglePay

```php
GOOGLE_PAY = 'google_pay'
```


<a name="constant_MIR_PAY" class="anchor"></a>
###### MIR_PAY
Источник данных MirPay

```php
MIR_PAY = 'mir_pay'
```



---
## Properties
<a name="property_validValues"></a>
#### protected $validValues : array
---
**Type:** <a href="../array"><abbr title="array">array</abbr></a>
Массив принимаемых enum&#039;ом значений
**Details:**



---
## Methods
<a name="method_getEnabledValues" class="anchor"></a>
#### public getEnabledValues() : string[]

```php
Static public getEnabledValues() : string[]
```

**Summary**

Возвращает значения в enum'е значения которых разрешены.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

**Returns:** string[] - Массив разрешённых значений


<a name="method_getValidValues" class="anchor"></a>
#### public getValidValues() : array

```php
Static public getValidValues() : array
```

**Summary**

Возвращает все значения в enum'e.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

**Returns:** array - Массив значений в перечислении


<a name="method_valueExists" class="anchor"></a>
#### public valueExists() : bool

```php
Static public valueExists(mixed $value) : bool
```

**Summary**

Проверяет наличие значения в enum'e.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | value  | Проверяемое значение |

**Returns:** bool - True если значение имеется, false если нет



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