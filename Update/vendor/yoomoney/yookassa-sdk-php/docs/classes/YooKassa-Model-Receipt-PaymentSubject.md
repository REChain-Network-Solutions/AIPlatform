# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Receipt\PaymentSubject
### Namespace: [\YooKassa\Model\Receipt](../namespaces/yookassa-model-receipt.md)
---
**Summary:**

Класс, представляющий модель PaymentSubject.

**Description:**

Признак предмета расчета (тег в 54 ФЗ — 1212) — это то, за что принимается оплата, например товар, услуга.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [COMMODITY](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_COMMODITY) |  | Товар |
| public | [EXCISE](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_EXCISE) |  | Подакцизный товар |
| public | [JOB](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_JOB) |  | Работа |
| public | [SERVICE](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_SERVICE) |  | Услуга |
| public | [GAMBLING_BET](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_GAMBLING_BET) |  | Ставка в азартной игре |
| public | [GAMBLING_PRIZE](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_GAMBLING_PRIZE) |  | Выигрыш азартной игры |
| public | [LOTTERY](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_LOTTERY) |  | Лотерейный билет |
| public | [LOTTERY_PRIZE](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_LOTTERY_PRIZE) |  | Выигрыш в лотерею |
| public | [INTELLECTUAL_ACTIVITY](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_INTELLECTUAL_ACTIVITY) |  | Результаты интеллектуальной деятельности |
| public | [PAYMENT](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_PAYMENT) |  | Платеж |
| public | [AGENT_COMMISSION](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_AGENT_COMMISSION) |  | Агентское вознаграждение |
| public | [PROPERTY_RIGHT](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_PROPERTY_RIGHT) |  | Имущественное право |
| public | [NON_OPERATING_GAIN](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_NON_OPERATING_GAIN) |  | Внереализационный доход |
| public | [INSURANCE_PREMIUM](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_INSURANCE_PREMIUM) |  | Страховой сбор |
| public | [SALES_TAX](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_SALES_TAX) |  | Торговый сбор |
| public | [RESORT_FEE](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_RESORT_FEE) |  | Курортный сбор |
| public | [COMPOSITE](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_COMPOSITE) |  | Несколько вариантов |
| public | [ANOTHER](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_ANOTHER) |  | Другое |
| public | [FINE](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_FINE) |  | Выплата |
| public | [TAX](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_TAX) |  | Страховые взносы |
| public | [LIEN](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_LIEN) |  | Залог |
| public | [COST](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_COST) |  | Расход |
| public | [PENSION_INSURANCE_WITHOUT_PAYOUTS](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_PENSION_INSURANCE_WITHOUT_PAYOUTS) |  | Взносы на обязательное пенсионное страхование ИП |
| public | [PENSION_INSURANCE_WITH_PAYOUTS](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_PENSION_INSURANCE_WITH_PAYOUTS) |  | Взносы на обязательное пенсионное страхование |
| public | [HEALTH_INSURANCE_WITHOUT_PAYOUTS](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_HEALTH_INSURANCE_WITHOUT_PAYOUTS) |  | Взносы на обязательное медицинское страхование ИП |
| public | [HEALTH_INSURANCE_WITH_PAYOUTS](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_HEALTH_INSURANCE_WITH_PAYOUTS) |  | Взносы на обязательное медицинское страхование |
| public | [HEALTH_INSURANCE](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_HEALTH_INSURANCE) |  | Взносы на обязательное социальное страхование |
| public | [CASINO](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_CASINO) |  | Платеж казино |
| public | [AGENT_WITHDRAWALS](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_AGENT_WITHDRAWALS) |  | Выдача денежных средств |
| public | [NON_MARKED_EXCISE](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_NON_MARKED_EXCISE) |  | Подакцизный товар, подлежащий маркировке средством идентификации, не имеющим кода маркировки (в чеке — АТНМ). Пример: алкогольная продукция |
| public | [MARKED_EXCISE](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_MARKED_EXCISE) |  | Подакцизный товар, подлежащий маркировке средством идентификации, имеющим код маркировки (в чеке — АТМ). Пример: табак |
| public | [MARKED](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_MARKED) |  | Товар, подлежащий маркировке средством идентификации, имеющим код маркировки, за исключением подакцизного товара (в чеке — ТМ). Пример: обувь, духи, товары легкой промышленности |
| public | [NON_MARKED](../classes/YooKassa-Model-Receipt-PaymentSubject.md#constant_NON_MARKED) |  | Товар, подлежащий маркировке средством идентификации, не имеющим кода маркировки, за исключением подакцизного товара (в чеке — ТНМ). Пример: меховые изделия |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-Receipt-PaymentSubject.md#property_validValues) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/Receipt/PaymentSubject.php](../../lib/Model/Receipt/PaymentSubject.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\Receipt\PaymentSubject

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
<a name="constant_COMMODITY" class="anchor"></a>
###### COMMODITY
Товар

```php
COMMODITY = 'commodity'
```


<a name="constant_EXCISE" class="anchor"></a>
###### EXCISE
Подакцизный товар

```php
EXCISE = 'excise'
```


<a name="constant_JOB" class="anchor"></a>
###### JOB
Работа

```php
JOB = 'job'
```


<a name="constant_SERVICE" class="anchor"></a>
###### SERVICE
Услуга

```php
SERVICE = 'service'
```


<a name="constant_GAMBLING_BET" class="anchor"></a>
###### GAMBLING_BET
Ставка в азартной игре

```php
GAMBLING_BET = 'gambling_bet'
```


<a name="constant_GAMBLING_PRIZE" class="anchor"></a>
###### GAMBLING_PRIZE
Выигрыш азартной игры

```php
GAMBLING_PRIZE = 'gambling_prize'
```


<a name="constant_LOTTERY" class="anchor"></a>
###### LOTTERY
Лотерейный билет

```php
LOTTERY = 'lottery'
```


<a name="constant_LOTTERY_PRIZE" class="anchor"></a>
###### LOTTERY_PRIZE
Выигрыш в лотерею

```php
LOTTERY_PRIZE = 'lottery_prize'
```


<a name="constant_INTELLECTUAL_ACTIVITY" class="anchor"></a>
###### INTELLECTUAL_ACTIVITY
Результаты интеллектуальной деятельности

```php
INTELLECTUAL_ACTIVITY = 'intellectual_activity'
```


<a name="constant_PAYMENT" class="anchor"></a>
###### PAYMENT
Платеж

```php
PAYMENT = 'payment'
```


<a name="constant_AGENT_COMMISSION" class="anchor"></a>
###### AGENT_COMMISSION
Агентское вознаграждение

```php
AGENT_COMMISSION = 'agent_commission'
```


<a name="constant_PROPERTY_RIGHT" class="anchor"></a>
###### PROPERTY_RIGHT
Имущественное право

```php
PROPERTY_RIGHT = 'property_right'
```


<a name="constant_NON_OPERATING_GAIN" class="anchor"></a>
###### NON_OPERATING_GAIN
Внереализационный доход

```php
NON_OPERATING_GAIN = 'non_operating_gain'
```


<a name="constant_INSURANCE_PREMIUM" class="anchor"></a>
###### INSURANCE_PREMIUM
Страховой сбор

```php
INSURANCE_PREMIUM = 'insurance_premium'
```


<a name="constant_SALES_TAX" class="anchor"></a>
###### SALES_TAX
Торговый сбор

```php
SALES_TAX = 'sales_tax'
```


<a name="constant_RESORT_FEE" class="anchor"></a>
###### RESORT_FEE
Курортный сбор

```php
RESORT_FEE = 'resort_fee'
```


<a name="constant_COMPOSITE" class="anchor"></a>
###### COMPOSITE
Несколько вариантов

```php
COMPOSITE = 'composite'
```


<a name="constant_ANOTHER" class="anchor"></a>
###### ANOTHER
Другое

```php
ANOTHER = 'another'
```


<a name="constant_FINE" class="anchor"></a>
###### FINE
Выплата

```php
FINE = 'fine'
```


<a name="constant_TAX" class="anchor"></a>
###### TAX
Страховые взносы

```php
TAX = 'tax'
```


<a name="constant_LIEN" class="anchor"></a>
###### LIEN
Залог

```php
LIEN = 'lien'
```


<a name="constant_COST" class="anchor"></a>
###### COST
Расход

```php
COST = 'cost'
```


<a name="constant_PENSION_INSURANCE_WITHOUT_PAYOUTS" class="anchor"></a>
###### PENSION_INSURANCE_WITHOUT_PAYOUTS
Взносы на обязательное пенсионное страхование ИП

```php
PENSION_INSURANCE_WITHOUT_PAYOUTS = 'pension_insurance_without_payouts'
```


<a name="constant_PENSION_INSURANCE_WITH_PAYOUTS" class="anchor"></a>
###### PENSION_INSURANCE_WITH_PAYOUTS
Взносы на обязательное пенсионное страхование

```php
PENSION_INSURANCE_WITH_PAYOUTS = 'pension_insurance_with_payouts'
```


<a name="constant_HEALTH_INSURANCE_WITHOUT_PAYOUTS" class="anchor"></a>
###### HEALTH_INSURANCE_WITHOUT_PAYOUTS
Взносы на обязательное медицинское страхование ИП

```php
HEALTH_INSURANCE_WITHOUT_PAYOUTS = 'health_insurance_without_payouts'
```


<a name="constant_HEALTH_INSURANCE_WITH_PAYOUTS" class="anchor"></a>
###### HEALTH_INSURANCE_WITH_PAYOUTS
Взносы на обязательное медицинское страхование

```php
HEALTH_INSURANCE_WITH_PAYOUTS = 'health_insurance_with_payouts'
```


<a name="constant_HEALTH_INSURANCE" class="anchor"></a>
###### HEALTH_INSURANCE
Взносы на обязательное социальное страхование

```php
HEALTH_INSURANCE = 'health_insurance'
```


<a name="constant_CASINO" class="anchor"></a>
###### CASINO
Платеж казино

```php
CASINO = 'casino'
```


<a name="constant_AGENT_WITHDRAWALS" class="anchor"></a>
###### AGENT_WITHDRAWALS
Выдача денежных средств

```php
AGENT_WITHDRAWALS = 'agent_withdrawals'
```


<a name="constant_NON_MARKED_EXCISE" class="anchor"></a>
###### NON_MARKED_EXCISE
Подакцизный товар, подлежащий маркировке средством идентификации, не имеющим кода маркировки (в чеке — АТНМ). Пример: алкогольная продукция

```php
NON_MARKED_EXCISE = 'non_marked_excise'
```


<a name="constant_MARKED_EXCISE" class="anchor"></a>
###### MARKED_EXCISE
Подакцизный товар, подлежащий маркировке средством идентификации, имеющим код маркировки (в чеке — АТМ). Пример: табак

```php
MARKED_EXCISE = 'marked_excise'
```


<a name="constant_MARKED" class="anchor"></a>
###### MARKED
Товар, подлежащий маркировке средством идентификации, имеющим код маркировки, за исключением подакцизного товара (в чеке — ТМ). Пример: обувь, духи, товары легкой промышленности

```php
MARKED = 'marked'
```


<a name="constant_NON_MARKED" class="anchor"></a>
###### NON_MARKED
Товар, подлежащий маркировке средством идентификации, не имеющим кода маркировки, за исключением подакцизного товара (в чеке — ТНМ). Пример: меховые изделия

```php
NON_MARKED = 'non_marked'
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