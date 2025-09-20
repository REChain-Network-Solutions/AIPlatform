# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Receipt\AgentType
### Namespace: [\YooKassa\Model\Receipt](../namespaces/yookassa-model-receipt.md)
---
**Summary:**

AgentType - Тип посредника.

**Description:**

Тип посредника передается в запросе на создание чека в массиве `items`, в параметре `agent_type`,
если вы отправляете данные для формирования чека по сценарию Сначала платеж, потом чек.
Параметр `agent_type` нужно передавать, начиная с ФФД 1.1. Убедитесь, что ваша онлайн-касса обновлена до этой версии.

Возможные значения:
- `banking_payment_agent` - Безналичный расчет
- `banking_payment_subagent` - Предоплата (аванс)
- `payment_agent` - Постоплата (кредит)
- `payment_subagent` - Встречное предоставление
- `attorney` - Встречное предоставление
- `commissioner` - Встречное предоставление
- `agent` - Встречное предоставление

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [BANKING_PAYMENT_AGENT](../classes/YooKassa-Model-Receipt-AgentType.md#constant_BANKING_PAYMENT_AGENT) |  | Банковский платежный агент |
| public | [BANKING_PAYMENT_SUBAGENT](../classes/YooKassa-Model-Receipt-AgentType.md#constant_BANKING_PAYMENT_SUBAGENT) |  | Банковский платежный субагент |
| public | [PAYMENT_AGENT](../classes/YooKassa-Model-Receipt-AgentType.md#constant_PAYMENT_AGENT) |  | Платежный агент |
| public | [PAYMENT_SUBAGENT](../classes/YooKassa-Model-Receipt-AgentType.md#constant_PAYMENT_SUBAGENT) |  | Платежный субагент |
| public | [ATTORNEY](../classes/YooKassa-Model-Receipt-AgentType.md#constant_ATTORNEY) |  | Поверенный |
| public | [COMMISSIONER](../classes/YooKassa-Model-Receipt-AgentType.md#constant_COMMISSIONER) |  | Комиссионер |
| public | [AGENT](../classes/YooKassa-Model-Receipt-AgentType.md#constant_AGENT) |  | Агент |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-Receipt-AgentType.md#property_validValues) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/Receipt/AgentType.php](../../lib/Model/Receipt/AgentType.php)
* Package: Default
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\Receipt\AgentType

---
## Constants
<a name="constant_BANKING_PAYMENT_AGENT" class="anchor"></a>
###### BANKING_PAYMENT_AGENT
Банковский платежный агент

```php
BANKING_PAYMENT_AGENT = 'banking_payment_agent'
```


<a name="constant_BANKING_PAYMENT_SUBAGENT" class="anchor"></a>
###### BANKING_PAYMENT_SUBAGENT
Банковский платежный субагент

```php
BANKING_PAYMENT_SUBAGENT = 'banking_payment_subagent'
```


<a name="constant_PAYMENT_AGENT" class="anchor"></a>
###### PAYMENT_AGENT
Платежный агент

```php
PAYMENT_AGENT = 'payment_agent'
```


<a name="constant_PAYMENT_SUBAGENT" class="anchor"></a>
###### PAYMENT_SUBAGENT
Платежный субагент

```php
PAYMENT_SUBAGENT = 'payment_subagent'
```


<a name="constant_ATTORNEY" class="anchor"></a>
###### ATTORNEY
Поверенный

```php
ATTORNEY = 'attorney'
```


<a name="constant_COMMISSIONER" class="anchor"></a>
###### COMMISSIONER
Комиссионер

```php
COMMISSIONER = 'commissioner'
```


<a name="constant_AGENT" class="anchor"></a>
###### AGENT
Агент

```php
AGENT = 'agent'
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