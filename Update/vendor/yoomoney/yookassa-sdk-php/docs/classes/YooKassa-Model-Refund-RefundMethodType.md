# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Refund\RefundMethodType
### Namespace: [\YooKassa\Model\Refund](../namespaces/yookassa-model-refund.md)
---
**Summary:**

Класс, представляющий модель PaymentMethodType.

**Description:**

Тип источника средств для проведения платежа.

Возможные значения:
- `sbp` - Возврат через сервис быстрых платежей
- `electronic_certificate` - Возврат платежа по электронному сертификату

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [SBP](../classes/YooKassa-Model-Refund-RefundMethodType.md#constant_SBP) |  | Возврат через сервис быстрых платежей |
| public | [ELECTRONIC_CERTIFICATE](../classes/YooKassa-Model-Refund-RefundMethodType.md#constant_ELECTRONIC_CERTIFICATE) |  | Прием платежей по электронному сертификату, привязанному к карте «Мир» |
| public | [UNKNOWN](../classes/YooKassa-Model-Refund-RefundMethodType.md#constant_UNKNOWN) | *deprecated* | Для неизвестных методов возврата |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-Refund-RefundMethodType.md#property_validValues) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/Refund/RefundMethodType.php](../../lib/Model/Refund/RefundMethodType.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\Refund\RefundMethodType

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
<a name="constant_SBP" class="anchor"></a>
###### SBP
Возврат через сервис быстрых платежей

```php
SBP = 'sbp'
```


<a name="constant_ELECTRONIC_CERTIFICATE" class="anchor"></a>
###### ELECTRONIC_CERTIFICATE
Прием платежей по электронному сертификату, привязанному к карте «Мир»

```php
ELECTRONIC_CERTIFICATE = 'electronic_certificate'
```


<a name="constant_UNKNOWN" class="anchor"></a>
###### ~~UNKNOWN~~
Для неизвестных методов возврата

```php
UNKNOWN = 'unknown'
```

**deprecated**
Не используется для реальных возвратов


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