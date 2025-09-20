# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\SelfEmployed\SelfEmployedStatus
### Namespace: [\YooKassa\Model\SelfEmployed](../namespaces/yookassa-model-selfemployed.md)
---
**Summary:**

Класс, представляющий модель SelfEmployedStatus.

**Description:**

Статус подключения самозанятого и выдачи ЮMoney прав на регистрацию чеков.

Возможные значения:
- `pending` — ЮMoney запросили права на регистрацию чеков, но самозанятый еще не ответил на заявку;
- `confirmed` — самозанятый выдал права ЮMoney; вы можете делать выплаты;
- `canceled` — самозанятый отклонил заявку или отозвал ранее выданные права;
- `unregistered` — самозанятый с таким ИНН не зарегистрирован в сервисе Мой налог, или пользователь потерял статус самозанятого.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [PENDING](../classes/YooKassa-Model-SelfEmployed-SelfEmployedStatus.md#constant_PENDING) |  | ЮMoney запросили права на регистрацию чеков, но самозанятый еще не ответил на заявку |
| public | [CONFIRMED](../classes/YooKassa-Model-SelfEmployed-SelfEmployedStatus.md#constant_CONFIRMED) |  | Самозанятый выдал права ЮMoney, вы можете делать выплаты |
| public | [CANCELED](../classes/YooKassa-Model-SelfEmployed-SelfEmployedStatus.md#constant_CANCELED) |  | Самозанятый отклонил заявку или отозвал ранее выданные права |
| public | [UNREGISTERED](../classes/YooKassa-Model-SelfEmployed-SelfEmployedStatus.md#constant_UNREGISTERED) |  | Самозанятый с таким ИНН не зарегистрирован в сервисе Мой налог, или пользователь потерял статус самозанятого |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-SelfEmployed-SelfEmployedStatus.md#property_validValues) |  | Возвращает список доступных значений |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/SelfEmployed/SelfEmployedStatus.php](../../lib/Model/SelfEmployed/SelfEmployedStatus.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\SelfEmployed\SelfEmployedStatus

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
ЮMoney запросили права на регистрацию чеков, но самозанятый еще не ответил на заявку

```php
PENDING = 'pending'
```


<a name="constant_CONFIRMED" class="anchor"></a>
###### CONFIRMED
Самозанятый выдал права ЮMoney, вы можете делать выплаты

```php
CONFIRMED = 'confirmed'
```


<a name="constant_CANCELED" class="anchor"></a>
###### CANCELED
Самозанятый отклонил заявку или отозвал ранее выданные права

```php
CANCELED = 'canceled'
```


<a name="constant_UNREGISTERED" class="anchor"></a>
###### UNREGISTERED
Самозанятый с таким ИНН не зарегистрирован в сервисе Мой налог, или пользователь потерял статус самозанятого

```php
UNREGISTERED = 'unregistered'
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