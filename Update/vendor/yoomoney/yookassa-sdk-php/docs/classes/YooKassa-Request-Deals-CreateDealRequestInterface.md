# [YooKassa API SDK](../home.md)

# Interface: CreateDealRequestInterface
### Namespace: [\YooKassa\Request\Deals](../namespaces/yookassa-request-deals.md)
---
**Summary:**

Interface CreateDealRequestInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getDescription()](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md#method_getDescription) |  | Возвращает описание сделки (не более 128 символов). |
| public | [getFeeMoment()](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md#method_getFeeMoment) |  | Возвращает момент перечисления вам вознаграждения платформы. |
| public | [getMetadata()](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md#method_getMetadata) |  | Возвращает дополнительные данные сделки. |
| public | [getType()](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md#method_getType) |  | Возвращает тип сделки. |
| public | [hasDescription()](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md#method_hasDescription) |  | Проверяет наличие описания в создаваемой сделке. |
| public | [hasFeeMoment()](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md#method_hasFeeMoment) |  | Проверяет наличие момента перечисления вознаграждения в создаваемой сделке. |
| public | [hasMetadata()](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md#method_hasMetadata) |  | Проверяет, были ли установлены метаданные сделки. |
| public | [hasType()](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md#method_hasType) |  | Проверяет наличие типа в создаваемой сделке. |
| public | [setDescription()](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md#method_setDescription) |  | Устанавливает описание сделки. |
| public | [setFeeMoment()](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md#method_setFeeMoment) |  | Устанавливает момент перечисления вознаграждения платформы. |
| public | [setMetadata()](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md#method_setMetadata) |  | Устанавливает метаданные, привязанные к сделке. |
| public | [setType()](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md#method_setType) |  | Устанавливает тип сделки. |

---
### Details
* File: [lib/Request/Deals/CreateDealRequestInterface.php](../../lib/Request/Deals/CreateDealRequestInterface.php)
* Package: \YooKassa\Request
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |
| property |  | Тип сделки |
| property |  | Момент перечисления вознаграждения |
| property |  | Момент перечисления вознаграждения |
| property |  | Описание сделки |
| property |  | Дополнительные данные сделки |

---
## Methods
<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип сделки.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

**Returns:** string|null - Тип сделки


<a name="method_hasType" class="anchor"></a>
#### public hasType() : bool

```php
public hasType() : bool
```

**Summary**

Проверяет наличие типа в создаваемой сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

**Returns:** bool - True если тип сделки установлен, false если нет


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string $type) : self
```

**Summary**

Устанавливает тип сделки.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | type  | Тип сделки |

**Returns:** self - 


<a name="method_getFeeMoment" class="anchor"></a>
#### public getFeeMoment() : string|null

```php
public getFeeMoment() : string|null
```

**Summary**

Возвращает момент перечисления вам вознаграждения платформы.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

**Returns:** string|null - Момент перечисления вознаграждения


<a name="method_hasFeeMoment" class="anchor"></a>
#### public hasFeeMoment() : bool

```php
public hasFeeMoment() : bool
```

**Summary**

Проверяет наличие момента перечисления вознаграждения в создаваемой сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

**Returns:** bool - True если момент перечисления вознаграждения установлен, false если нет


<a name="method_setFeeMoment" class="anchor"></a>
#### public setFeeMoment() : self

```php
public setFeeMoment(string $fee_moment) : self
```

**Summary**

Устанавливает момент перечисления вознаграждения платформы.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | fee_moment  | Момент перечисления вознаграждения |

**Returns:** self - 


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает описание сделки (не более 128 символов).

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

**Returns:** string|null - Описание сделки


<a name="method_hasDescription" class="anchor"></a>
#### public hasDescription() : bool

```php
public hasDescription() : bool
```

**Summary**

Проверяет наличие описания в создаваемой сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

**Returns:** bool - True если описание сделки установлено, false если нет


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description) : self
```

**Summary**

Устанавливает описание сделки.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Описание сделки |

**Returns:** self - 


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает дополнительные данные сделки.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

**Returns:** \YooKassa\Model\Metadata|null - Дополнительные данные сделки


<a name="method_hasMetadata" class="anchor"></a>
#### public hasMetadata() : bool

```php
public hasMetadata() : bool
```

**Summary**

Проверяет, были ли установлены метаданные сделки.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

**Returns:** bool - True если метаданные были установлены, false если нет


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(null|array|\YooKassa\Model\Metadata $metadata) : self
```

**Summary**

Устанавливает метаданные, привязанные к сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Metadata</code> | metadata  | Метаданные сделки, устанавливаемые мерчантом |

**Returns:** self - 




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