# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\SavePaymentMethod\Confirmation\ConfirmationType
### Namespace: [\YooKassa\Model\SavePaymentMethod\Confirmation](../namespaces/yookassa-model-savepaymentmethod-confirmation.md)
---
**Summary:**

Класс, представляющий модель ConfirmationType.

**Description:**

Код сценария подтверждения пользователем привязки платежного средства к вашему магазину в ЮKassa. Фиксированное значение:
~`redirect` — Перенаправление пользователя на сайт ЮKassa для подтверждения привязки или страницу банка-эмитента для аутентификации по 3-D Secure.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [REDIRECT](../classes/YooKassa-Model-SavePaymentMethod-Confirmation-ConfirmationType.md#constant_REDIRECT) |  | Перенаправление пользователя на сайт ЮKassa для подтверждения привязки или страницу банка-эмитента для аутентификации по 3-D Secure. |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-SavePaymentMethod-Confirmation-ConfirmationType.md#property_validValues) |  | Возвращает список доступных значений |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/SavePaymentMethod/Confirmation/ConfirmationType.php](../../lib/Model/SavePaymentMethod/Confirmation/ConfirmationType.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\SavePaymentMethod\Confirmation\ConfirmationType

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
<a name="constant_REDIRECT" class="anchor"></a>
###### REDIRECT
Перенаправление пользователя на сайт ЮKassa для подтверждения привязки или страницу банка-эмитента для аутентификации по 3-D Secure.

```php
REDIRECT = 'redirect'
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