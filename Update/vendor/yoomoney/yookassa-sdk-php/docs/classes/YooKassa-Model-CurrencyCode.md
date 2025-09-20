# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\CurrencyCode
### Namespace: [\YooKassa\Model](../namespaces/yookassa-model.md)
---
**Summary:**

Класс, представляющий модель CurrencyCode.

**Description:**

Трехбуквенный код валюты в формате [ISO-4217](https://www.iso.org/iso-4217-currency-codes.md).

Пример: ~`RUB`. Должен соответствовать валюте субаккаунта (`recipient.gateway_id`),
если вы разделяете потоки платежей, и валюте аккаунта (shopId в [личном кабинете](https://yookassa.ru/my)), если не разделяете.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [RUB](../classes/YooKassa-Model-CurrencyCode.md#constant_RUB) |  | Российский рубль |
| public | [USD](../classes/YooKassa-Model-CurrencyCode.md#constant_USD) |  | Доллар США |
| public | [EUR](../classes/YooKassa-Model-CurrencyCode.md#constant_EUR) |  | Евро |
| public | [BYN](../classes/YooKassa-Model-CurrencyCode.md#constant_BYN) |  | Белорусский рубль |
| public | [CNY](../classes/YooKassa-Model-CurrencyCode.md#constant_CNY) |  | Китайская йена |
| public | [KZT](../classes/YooKassa-Model-CurrencyCode.md#constant_KZT) |  | Казахский тенге |
| public | [UAH](../classes/YooKassa-Model-CurrencyCode.md#constant_UAH) |  | Украинская гривна |
| public | [UZS](../classes/YooKassa-Model-CurrencyCode.md#constant_UZS) |  | Узбекский сум |
| public | [_TRY](../classes/YooKassa-Model-CurrencyCode.md#constant__TRY) |  | Турецкая лира |
| public | [INR](../classes/YooKassa-Model-CurrencyCode.md#constant_INR) |  | Индийская рупия |
| public | [MDL](../classes/YooKassa-Model-CurrencyCode.md#constant_MDL) |  | Молдавский лей |
| public | [AZN](../classes/YooKassa-Model-CurrencyCode.md#constant_AZN) |  | Азербайджанский манат |
| public | [AMD](../classes/YooKassa-Model-CurrencyCode.md#constant_AMD) |  | Армянский драм |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-CurrencyCode.md#property_validValues) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/CurrencyCode.php](../../lib/Model/CurrencyCode.php)
* Package: Default
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\CurrencyCode

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_RUB" class="anchor"></a>
###### RUB
Российский рубль

```php
RUB = 'RUB'
```


<a name="constant_USD" class="anchor"></a>
###### USD
Доллар США

```php
USD = 'USD'
```


<a name="constant_EUR" class="anchor"></a>
###### EUR
Евро

```php
EUR = 'EUR'
```


<a name="constant_BYN" class="anchor"></a>
###### BYN
Белорусский рубль

```php
BYN = 'BYN'
```


<a name="constant_CNY" class="anchor"></a>
###### CNY
Китайская йена

```php
CNY = 'CNY'
```


<a name="constant_KZT" class="anchor"></a>
###### KZT
Казахский тенге

```php
KZT = 'KZT'
```


<a name="constant_UAH" class="anchor"></a>
###### UAH
Украинская гривна

```php
UAH = 'UAH'
```


<a name="constant_UZS" class="anchor"></a>
###### UZS
Узбекский сум

```php
UZS = 'UZS'
```


<a name="constant__TRY" class="anchor"></a>
###### _TRY
Турецкая лира

```php
_TRY = 'TRY'
```


<a name="constant_INR" class="anchor"></a>
###### INR
Индийская рупия

```php
INR = 'INR'
```


<a name="constant_MDL" class="anchor"></a>
###### MDL
Молдавский лей

```php
MDL = 'MDL'
```


<a name="constant_AZN" class="anchor"></a>
###### AZN
Азербайджанский манат

```php
AZN = 'AZN'
```


<a name="constant_AMD" class="anchor"></a>
###### AMD
Армянский драм

```php
AMD = 'AMD'
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