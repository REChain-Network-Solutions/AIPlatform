# [YooKassa API SDK](../home.md)

# Interface: AmountInterface
### Namespace: [\YooKassa\Model](../namespaces/yookassa-model.md)
---
**Summary:**

Interface AmountInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getCurrency()](../classes/YooKassa-Model-AmountInterface.md#method_getCurrency) |  | Возвращает валюту. |
| public | [getIntegerValue()](../classes/YooKassa-Model-AmountInterface.md#method_getIntegerValue) |  | Возвращает сумму в копейках в виде целого числа. |
| public | [getValue()](../classes/YooKassa-Model-AmountInterface.md#method_getValue) |  | Возвращает значение суммы. |
| public | [setCurrency()](../classes/YooKassa-Model-AmountInterface.md#method_setCurrency) |  | Устанавливает код валюты. |
| public | [setValue()](../classes/YooKassa-Model-AmountInterface.md#method_setValue) |  | Устанавливает значение суммы. |
| public | [toArray()](../classes/YooKassa-Model-AmountInterface.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |

---
### Details
* File: [lib/Model/AmountInterface.php](../../lib/Model/AmountInterface.php)
* Package: \Default

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| property |  | Сумма |
| property |  | Код валюты |

---
## Methods
<a name="method_getValue" class="anchor"></a>
#### public getValue() : string

```php
public getValue() : string
```

**Summary**

Возвращает значение суммы.

**Details:**
* Inherited From: [\YooKassa\Model\AmountInterface](../classes/YooKassa-Model-AmountInterface.md)

**Returns:** string - Сумма


<a name="method_setValue" class="anchor"></a>
#### public setValue() : mixed

```php
public setValue(\YooKassa\Model\numeric|string $value) : mixed
```

**Summary**

Устанавливает значение суммы.

**Details:**
* Inherited From: [\YooKassa\Model\AmountInterface](../classes/YooKassa-Model-AmountInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\numeric OR string</code> | value  | Сумма |

**Returns:** mixed - 


<a name="method_getIntegerValue" class="anchor"></a>
#### public getIntegerValue() : int

```php
public getIntegerValue() : int
```

**Summary**

Возвращает сумму в копейках в виде целого числа.

**Details:**
* Inherited From: [\YooKassa\Model\AmountInterface](../classes/YooKassa-Model-AmountInterface.md)

**Returns:** int - Сумма в копейках/центах


<a name="method_getCurrency" class="anchor"></a>
#### public getCurrency() : string

```php
public getCurrency() : string
```

**Summary**

Возвращает валюту.

**Details:**
* Inherited From: [\YooKassa\Model\AmountInterface](../classes/YooKassa-Model-AmountInterface.md)

**Returns:** string - Код валюты


<a name="method_setCurrency" class="anchor"></a>
#### public setCurrency() : mixed

```php
public setCurrency(string $currency) : mixed
```

**Summary**

Устанавливает код валюты.

**Details:**
* Inherited From: [\YooKassa\Model\AmountInterface](../classes/YooKassa-Model-AmountInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | currency  | Код валюты |

**Returns:** mixed - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Model\AmountInterface](../classes/YooKassa-Model-AmountInterface.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта




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