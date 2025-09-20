# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Settings\FiscalizationProvider
### Namespace: [\YooKassa\Model\Settings](../namespaces/yookassa-model-settings.md)
---
**Summary:**

Класс, представляющий модель FiscalizationProvider.

**Description:**

Решение ЮKassa, которое магазин использует для отправки чеков.
Возможные значения:
- [Чеки для самозанятых](https://yookassa.ru/developers/payment-acceptance/receipts/self-employed/basics) — `fns`
- [54-ФЗ: Чеки от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — `avanpost`
- [54-ФЗ: сторонняя онлайн-касса](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) (наименование онлайн-кассы) — ~`a_qsi` (aQsi online), ~`atol` (АТОЛ Онлайн), ~`business_ru` (Бизнес.ру), ~`digital_kassa` (digitalkassa), ~`evotor` (Эвотор), ~`first_ofd` (Первый ОФД), ~`kit_invest` (Кит Инвест), ~`komtet` (КОМТЕТ Касса), ~`life_pay` (LIFE PAY), ~`mertrade` (Mertrade), ~`modul_kassa` (МодульКасса), ~`rocket` (RocketR), ~`shtrih_m` (Orange Data).

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [ATOL](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_ATOL) |  | АТОЛ Онлайн |
| public | [BUSINESS_RU](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_BUSINESS_RU) |  | Бизнес.ру |
| public | [SHTRIH_M](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_SHTRIH_M) |  | Orange Data |
| public | [MODUL_KASSA](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_MODUL_KASSA) |  | МодульКасса |
| public | [EVOTOR](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_EVOTOR) |  | Эвотор |
| public | [KIT_INVEST](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_KIT_INVEST) |  | Кит Инвест |
| public | [A_QSI](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_A_QSI) |  | aQsi online |
| public | [FNS](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_FNS) |  | Чеки для самозанятых |
| public | [AVANPOST](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_AVANPOST) |  | 54-ФЗ: Чеки от ЮKassa |
| public | [MERTRADE](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_MERTRADE) |  | Mertrade |
| public | [FIRST_OFD](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_FIRST_OFD) |  | Первый ОФД |
| public | [LIFE_PAY](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_LIFE_PAY) |  | LIFE PAY |
| public | [ROCKET](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_ROCKET) |  | RocketR |
| public | [DIGITAL_KASSA](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_DIGITAL_KASSA) |  | digitalkassa |
| public | [KOMTET](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#constant_KOMTET) |  | КОМТЕТ Касса |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-Settings-FiscalizationProvider.md#property_validValues) |  | Возвращает список доступных значений |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/Settings/FiscalizationProvider.php](../../lib/Model/Settings/FiscalizationProvider.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\Settings\FiscalizationProvider

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
<a name="constant_ATOL" class="anchor"></a>
###### ATOL
АТОЛ Онлайн

```php
ATOL = 'atol'
```


<a name="constant_BUSINESS_RU" class="anchor"></a>
###### BUSINESS_RU
Бизнес.ру

```php
BUSINESS_RU = 'business_ru'
```


<a name="constant_SHTRIH_M" class="anchor"></a>
###### SHTRIH_M
Orange Data

```php
SHTRIH_M = 'shtrih_m'
```


<a name="constant_MODUL_KASSA" class="anchor"></a>
###### MODUL_KASSA
МодульКасса

```php
MODUL_KASSA = 'modul_kassa'
```


<a name="constant_EVOTOR" class="anchor"></a>
###### EVOTOR
Эвотор

```php
EVOTOR = 'evotor'
```


<a name="constant_KIT_INVEST" class="anchor"></a>
###### KIT_INVEST
Кит Инвест

```php
KIT_INVEST = 'kit_invest'
```


<a name="constant_A_QSI" class="anchor"></a>
###### A_QSI
aQsi online

```php
A_QSI = 'a_qsi'
```


<a name="constant_FNS" class="anchor"></a>
###### FNS
Чеки для самозанятых

```php
FNS = 'fns'
```


<a name="constant_AVANPOST" class="anchor"></a>
###### AVANPOST
54-ФЗ: Чеки от ЮKassa

```php
AVANPOST = 'avanpost'
```


<a name="constant_MERTRADE" class="anchor"></a>
###### MERTRADE
Mertrade

```php
MERTRADE = 'mertrade'
```


<a name="constant_FIRST_OFD" class="anchor"></a>
###### FIRST_OFD
Первый ОФД

```php
FIRST_OFD = 'first_ofd'
```


<a name="constant_LIFE_PAY" class="anchor"></a>
###### LIFE_PAY
LIFE PAY

```php
LIFE_PAY = 'life_pay'
```


<a name="constant_ROCKET" class="anchor"></a>
###### ROCKET
RocketR

```php
ROCKET = 'rocket'
```


<a name="constant_DIGITAL_KASSA" class="anchor"></a>
###### DIGITAL_KASSA
digitalkassa

```php
DIGITAL_KASSA = 'digital_kassa'
```


<a name="constant_KOMTET" class="anchor"></a>
###### KOMTET
КОМТЕТ Касса

```php
KOMTET = 'komtet'
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