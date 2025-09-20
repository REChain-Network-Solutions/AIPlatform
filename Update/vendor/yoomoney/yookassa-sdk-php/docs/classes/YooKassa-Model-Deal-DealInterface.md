# [YooKassa API SDK](../home.md)

# Interface: DealInterface
### Namespace: [\YooKassa\Model\Deal](../namespaces/yookassa-model-deal.md)
---
**Summary:**

Interface DealInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getBalance()](../classes/YooKassa-Model-Deal-DealInterface.md#method_getBalance) |  | Возвращает баланс сделки. |
| public | [getCreatedAt()](../classes/YooKassa-Model-Deal-DealInterface.md#method_getCreatedAt) |  | Возвращает время создания сделки. |
| public | [getDescription()](../classes/YooKassa-Model-Deal-DealInterface.md#method_getDescription) |  | Возвращает описание сделки (не более 128 символов). |
| public | [getExpiresAt()](../classes/YooKassa-Model-Deal-DealInterface.md#method_getExpiresAt) |  | Возвращает время автоматического закрытия сделки. |
| public | [getFeeMoment()](../classes/YooKassa-Model-Deal-DealInterface.md#method_getFeeMoment) |  | Возвращает момент перечисления вам вознаграждения платформы. |
| public | [getId()](../classes/YooKassa-Model-Deal-DealInterface.md#method_getId) |  | Возвращает Id сделки. |
| public | [getMetadata()](../classes/YooKassa-Model-Deal-DealInterface.md#method_getMetadata) |  | Возвращает дополнительные данные сделки. |
| public | [getPayoutBalance()](../classes/YooKassa-Model-Deal-DealInterface.md#method_getPayoutBalance) |  | Возвращает сумму вознаграждения продавца. |
| public | [getStatus()](../classes/YooKassa-Model-Deal-DealInterface.md#method_getStatus) |  | Возвращает статус сделки. |
| public | [getTest()](../classes/YooKassa-Model-Deal-DealInterface.md#method_getTest) |  | Возвращает признак тестовой операции. |
| public | [getType()](../classes/YooKassa-Model-Deal-DealInterface.md#method_getType) |  | Возвращает тип сделки. |

---
### Details
* File: [lib/Model/Deal/DealInterface.php](../../lib/Model/Deal/DealInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |
| property |  | Идентификатор сделки |
| property |  | Статус сделки |
| property |  | Тип сделки |
| property |  | Момент перечисления вознаграждения |
| property |  | Момент перечисления вознаграждения |
| property |  | Описание сделки |
| property |  | Баланс сделки |
| property |  | Сумма вознаграждения продавца |
| property |  | Сумма вознаграждения продавца |
| property |  | Время создания сделки |
| property |  | Время создания сделки |
| property |  | Время автоматического закрытия сделки |
| property |  | Время автоматического закрытия сделки |
| property |  | Дополнительные данные сделки |
| property |  | Признак тестовой операции |

---
## Methods
<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает Id сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\DealInterface](../classes/YooKassa-Model-Deal-DealInterface.md)

**Returns:** string|null - Id сделки


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\DealInterface](../classes/YooKassa-Model-Deal-DealInterface.md)

**Returns:** string|null - Тип сделки


<a name="method_getFeeMoment" class="anchor"></a>
#### public getFeeMoment() : string|null

```php
public getFeeMoment() : string|null
```

**Summary**

Возвращает момент перечисления вам вознаграждения платформы.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\DealInterface](../classes/YooKassa-Model-Deal-DealInterface.md)

**Returns:** string|null - Момент перечисления вознаграждения


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает описание сделки (не более 128 символов).

**Details:**
* Inherited From: [\YooKassa\Model\Deal\DealInterface](../classes/YooKassa-Model-Deal-DealInterface.md)

**Returns:** string|null - Описание сделки


<a name="method_getBalance" class="anchor"></a>
#### public getBalance() : \YooKassa\Model\AmountInterface|null

```php
public getBalance() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает баланс сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\DealInterface](../classes/YooKassa-Model-Deal-DealInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Баланс сделки


<a name="method_getPayoutBalance" class="anchor"></a>
#### public getPayoutBalance() : \YooKassa\Model\AmountInterface|null

```php
public getPayoutBalance() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму вознаграждения продавца.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\DealInterface](../classes/YooKassa-Model-Deal-DealInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма вознаграждения продавца


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает статус сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\DealInterface](../classes/YooKassa-Model-Deal-DealInterface.md)

**Returns:** string|null - Статус сделки


<a name="method_getCreatedAt" class="anchor"></a>
#### public getCreatedAt() : \DateTime|null

```php
public getCreatedAt() : \DateTime|null
```

**Summary**

Возвращает время создания сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\DealInterface](../classes/YooKassa-Model-Deal-DealInterface.md)

**Returns:** \DateTime|null - Время создания сделки


<a name="method_getExpiresAt" class="anchor"></a>
#### public getExpiresAt() : \DateTime|null

```php
public getExpiresAt() : \DateTime|null
```

**Summary**

Возвращает время автоматического закрытия сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\DealInterface](../classes/YooKassa-Model-Deal-DealInterface.md)

**Returns:** \DateTime|null - Время автоматического закрытия сделки


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает дополнительные данные сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\DealInterface](../classes/YooKassa-Model-Deal-DealInterface.md)

**Returns:** \YooKassa\Model\Metadata|null - Дополнительные данные сделки


<a name="method_getTest" class="anchor"></a>
#### public getTest() : bool|null

```php
public getTest() : bool|null
```

**Summary**

Возвращает признак тестовой операции.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\DealInterface](../classes/YooKassa-Model-Deal-DealInterface.md)

**Returns:** bool|null - Признак тестовой операции




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