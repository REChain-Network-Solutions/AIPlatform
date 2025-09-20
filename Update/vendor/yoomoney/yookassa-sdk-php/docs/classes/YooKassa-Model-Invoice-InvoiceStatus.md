# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Invoice\InvoiceStatus
### Namespace: [\YooKassa\Model\Invoice](../namespaces/yookassa-model-invoice.md)
---
**Summary:**

Класс, представляющий модель InvoiceStatus.

**Description:**

Статус счета.

Возможные значения:
- `pending` — счет создан и ожидает успешной оплаты;
- `succeeded` — счет успешно оплачен, есть связанный платеж в статусе `succeeded` (финальный и неизменяемый статус для платежей в одну стадию);
- `canceled` — вы отменили счет, успешный платеж по нему не поступил или был отменен (при оплате в две стадии) либо истек срок действия счета (финальный и неизменяемый статус).

Подробнее про [жизненный цикл счета](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/invoices/basics#invoice-status)

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [PENDING](../classes/YooKassa-Model-Invoice-InvoiceStatus.md#constant_PENDING) |  | Счет создан и ожидает успешной оплаты |
| public | [SUCCEEDED](../classes/YooKassa-Model-Invoice-InvoiceStatus.md#constant_SUCCEEDED) |  | Счет успешно оплачен, есть связанный платеж в статусе ~`succeeded` |
| public | [CANCELED](../classes/YooKassa-Model-Invoice-InvoiceStatus.md#constant_CANCELED) |  | Вы отменили счет, успешный платеж по нему не поступил или был отменен |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-Invoice-InvoiceStatus.md#property_validValues) |  | Возвращает список доступных значений |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/Invoice/InvoiceStatus.php](../../lib/Model/Invoice/InvoiceStatus.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\Invoice\InvoiceStatus

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
<a name="constant_PENDING" class="anchor"></a>
###### PENDING
Счет создан и ожидает успешной оплаты

```php
PENDING = 'pending'
```


<a name="constant_SUCCEEDED" class="anchor"></a>
###### SUCCEEDED
Счет успешно оплачен, есть связанный платеж в статусе ~`succeeded`

```php
SUCCEEDED = 'succeeded'
```


<a name="constant_CANCELED" class="anchor"></a>
###### CANCELED
Вы отменили счет, успешный платеж по нему не поступил или был отменен

```php
CANCELED = 'canceled'
```



---
## Properties
<a name="property_validValues"></a>
#### protected $validValues : array
---
**Summary**

Возвращает список доступных значений

**Type:** <a href="../array"><abbr title="array">array</abbr></a>
Массив принимаемых enum&#039;ом значений
**Details:**


##### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| return |  |  |


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