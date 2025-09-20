# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Receipt\ReceiptItemMeasure
### Namespace: [\YooKassa\Model\Receipt](../namespaces/yookassa-model-receipt.md)
---
**Summary:**

Класс, представляющий модель ReceiptItemMeasure.

**Description:**

Мера количества предмета расчета передается в массиве `items`, в параметре `measure`.

Обязательный параметр, если используете Чеки от ЮKassa или онлайн-кассу, обновленную до ФФД 1.2.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [PIECE](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_PIECE) |  | Штука, единица товара |
| public | [GRAM](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_GRAM) |  | Грамм |
| public | [KILOGRAM](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_KILOGRAM) |  | Килограмм |
| public | [TON](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_TON) |  | Тонна |
| public | [CENTIMETER](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_CENTIMETER) |  | Сантиметр |
| public | [DECIMETER](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_DECIMETER) |  | Дециметр |
| public | [METER](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_METER) |  | Метр |
| public | [SQUARE_CENTIMETER](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_SQUARE_CENTIMETER) |  | Квадратный сантиметр |
| public | [SQUARE_DECIMETER](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_SQUARE_DECIMETER) |  | Квадратный дециметр |
| public | [SQUARE_METER](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_SQUARE_METER) |  | Квадратный метр |
| public | [MILLILITER](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_MILLILITER) |  | Миллилитр |
| public | [LITER](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_LITER) |  | Литр |
| public | [CUBIC_METER](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_CUBIC_METER) |  | Кубический метр |
| public | [KILOWATT_HOUR](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_KILOWATT_HOUR) |  | Килловат-час |
| public | [GIGACALORIE](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_GIGACALORIE) |  | Гигакалория |
| public | [DAY](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_DAY) |  | Сутки |
| public | [HOUR](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_HOUR) |  | Час |
| public | [MINUTE](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_MINUTE) |  | Минута |
| public | [SECOND](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_SECOND) |  | Секунда |
| public | [KILOBYTE](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_KILOBYTE) |  | Килобайт |
| public | [MEGABYTE](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_MEGABYTE) |  | Мегабайт |
| public | [GIGABYTE](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_GIGABYTE) |  | Гигабайт |
| public | [TERABYTE](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_TERABYTE) |  | Терабайт |
| public | [ANOTHER](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#constant_ANOTHER) |  | Другое |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-Receipt-ReceiptItemMeasure.md#property_validValues) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/Receipt/ReceiptItemMeasure.php](../../lib/Model/Receipt/ReceiptItemMeasure.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\Receipt\ReceiptItemMeasure

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
<a name="constant_PIECE" class="anchor"></a>
###### PIECE
Штука, единица товара

```php
PIECE = 'piece'
```


<a name="constant_GRAM" class="anchor"></a>
###### GRAM
Грамм

```php
GRAM = 'gram'
```


<a name="constant_KILOGRAM" class="anchor"></a>
###### KILOGRAM
Килограмм

```php
KILOGRAM = 'kilogram'
```


<a name="constant_TON" class="anchor"></a>
###### TON
Тонна

```php
TON = 'ton'
```


<a name="constant_CENTIMETER" class="anchor"></a>
###### CENTIMETER
Сантиметр

```php
CENTIMETER = 'centimeter'
```


<a name="constant_DECIMETER" class="anchor"></a>
###### DECIMETER
Дециметр

```php
DECIMETER = 'decimeter'
```


<a name="constant_METER" class="anchor"></a>
###### METER
Метр

```php
METER = 'meter'
```


<a name="constant_SQUARE_CENTIMETER" class="anchor"></a>
###### SQUARE_CENTIMETER
Квадратный сантиметр

```php
SQUARE_CENTIMETER = 'square_centimeter'
```


<a name="constant_SQUARE_DECIMETER" class="anchor"></a>
###### SQUARE_DECIMETER
Квадратный дециметр

```php
SQUARE_DECIMETER = 'square_decimeter'
```


<a name="constant_SQUARE_METER" class="anchor"></a>
###### SQUARE_METER
Квадратный метр

```php
SQUARE_METER = 'square_meter'
```


<a name="constant_MILLILITER" class="anchor"></a>
###### MILLILITER
Миллилитр

```php
MILLILITER = 'milliliter'
```


<a name="constant_LITER" class="anchor"></a>
###### LITER
Литр

```php
LITER = 'liter'
```


<a name="constant_CUBIC_METER" class="anchor"></a>
###### CUBIC_METER
Кубический метр

```php
CUBIC_METER = 'cubic_meter'
```


<a name="constant_KILOWATT_HOUR" class="anchor"></a>
###### KILOWATT_HOUR
Килловат-час

```php
KILOWATT_HOUR = 'kilowatt_hour'
```


<a name="constant_GIGACALORIE" class="anchor"></a>
###### GIGACALORIE
Гигакалория

```php
GIGACALORIE = 'gigacalorie'
```


<a name="constant_DAY" class="anchor"></a>
###### DAY
Сутки

```php
DAY = 'day'
```


<a name="constant_HOUR" class="anchor"></a>
###### HOUR
Час

```php
HOUR = 'hour'
```


<a name="constant_MINUTE" class="anchor"></a>
###### MINUTE
Минута

```php
MINUTE = 'minute'
```


<a name="constant_SECOND" class="anchor"></a>
###### SECOND
Секунда

```php
SECOND = 'second'
```


<a name="constant_KILOBYTE" class="anchor"></a>
###### KILOBYTE
Килобайт

```php
KILOBYTE = 'kilobyte'
```


<a name="constant_MEGABYTE" class="anchor"></a>
###### MEGABYTE
Мегабайт

```php
MEGABYTE = 'megabyte'
```


<a name="constant_GIGABYTE" class="anchor"></a>
###### GIGABYTE
Гигабайт

```php
GIGABYTE = 'gigabyte'
```


<a name="constant_TERABYTE" class="anchor"></a>
###### TERABYTE
Терабайт

```php
TERABYTE = 'terabyte'
```


<a name="constant_ANOTHER" class="anchor"></a>
###### ANOTHER
Другое

```php
ANOTHER = 'another'
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