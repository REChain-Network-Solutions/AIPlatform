# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\PersonalData\PersonalDataStatus
### Namespace: [\YooKassa\Model\PersonalData](../namespaces/yookassa-model-personaldata.md)
---
**Summary:**

Класс, представляющий модель PersonalDataStatus.

**Description:**

Статус персональных данных.
Возможные значения:
- `waiting_for_operation` — данные сохранены, но не использованы при проведении выплаты;
- `active` — данные сохранены и использованы при проведении выплаты; данные можно использовать повторно до срока, указанного в параметре `expires_at`;
- `canceled` — хранение данных отменено, данные удалены, инициатор и причина отмены указаны в объекте `cancellation_details` (финальный и неизменяемый статус).

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [WAITING_FOR_OPERATION](../classes/YooKassa-Model-PersonalData-PersonalDataStatus.md#constant_WAITING_FOR_OPERATION) |  | Данные сохранены, но не использованы при проведении выплаты |
| public | [ACTIVE](../classes/YooKassa-Model-PersonalData-PersonalDataStatus.md#constant_ACTIVE) |  | Данные сохранены и использованы при проведении выплаты |
| public | [CANCELED](../classes/YooKassa-Model-PersonalData-PersonalDataStatus.md#constant_CANCELED) |  | Хранение данных отменено |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-PersonalData-PersonalDataStatus.md#property_validValues) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/PersonalData/PersonalDataStatus.php](../../lib/Model/PersonalData/PersonalDataStatus.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\PersonalData\PersonalDataStatus

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
<a name="constant_WAITING_FOR_OPERATION" class="anchor"></a>
###### WAITING_FOR_OPERATION
Данные сохранены, но не использованы при проведении выплаты

```php
WAITING_FOR_OPERATION = 'waiting_for_operation'
```


<a name="constant_ACTIVE" class="anchor"></a>
###### ACTIVE
Данные сохранены и использованы при проведении выплаты

```php
ACTIVE = 'active'
```


<a name="constant_CANCELED" class="anchor"></a>
###### CANCELED
Хранение данных отменено

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