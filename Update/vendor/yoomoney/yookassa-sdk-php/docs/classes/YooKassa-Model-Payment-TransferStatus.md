# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payment\TransferStatus
### Namespace: [\YooKassa\Model\Payment](../namespaces/yookassa-model-payment.md)
---
**Summary:**

Класс, представляющий модель TransferStatus.

**Description:**

Статус распределения денег между магазинами.

Возможные значения:
- `pending` - Ожидает оплаты покупателем
- `waiting_for_capture` - Успешно оплачен покупателем, ожидает подтверждения магазином (capture или aviso)
- `succeeded` - Успешно оплачен и получен магазином
- `canceled` - Неуспех оплаты или отменен магазином (cancel)

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [PENDING](../classes/YooKassa-Model-Payment-TransferStatus.md#constant_PENDING) |  | Ожидает оплаты покупателем |
| public | [WAITING_FOR_CAPTURE](../classes/YooKassa-Model-Payment-TransferStatus.md#constant_WAITING_FOR_CAPTURE) |  | Успешно оплачен покупателем, ожидает подтверждения магазином (capture или aviso) |
| public | [SUCCEEDED](../classes/YooKassa-Model-Payment-TransferStatus.md#constant_SUCCEEDED) |  | Успешно оплачен и получен магазином |
| public | [CANCELED](../classes/YooKassa-Model-Payment-TransferStatus.md#constant_CANCELED) |  | Неуспех оплаты или отменен магазином (cancel) |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-Payment-TransferStatus.md#property_validValues) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/Payment/TransferStatus.php](../../lib/Model/Payment/TransferStatus.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\Payment\TransferStatus

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_PENDING" class="anchor"></a>
###### PENDING
Ожидает оплаты покупателем

```php
PENDING = 'pending'
```


<a name="constant_WAITING_FOR_CAPTURE" class="anchor"></a>
###### WAITING_FOR_CAPTURE
Успешно оплачен покупателем, ожидает подтверждения магазином (capture или aviso)

```php
WAITING_FOR_CAPTURE = 'waiting_for_capture'
```


<a name="constant_SUCCEEDED" class="anchor"></a>
###### SUCCEEDED
Успешно оплачен и получен магазином

```php
SUCCEEDED = 'succeeded'
```


<a name="constant_CANCELED" class="anchor"></a>
###### CANCELED
Неуспех оплаты или отменен магазином (cancel)

```php
CANCELED = 'canceled'
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