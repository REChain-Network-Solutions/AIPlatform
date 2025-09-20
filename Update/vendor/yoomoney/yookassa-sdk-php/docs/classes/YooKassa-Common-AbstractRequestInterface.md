# [YooKassa API SDK](../home.md)

# Interface: AbstractRequestInterface
### Namespace: [\YooKassa\Common](../namespaces/yookassa-common.md)
---
**Summary:**

Interface AbstractRequestInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [clearValidationError()](../classes/YooKassa-Common-AbstractRequestInterface.md#method_clearValidationError) |  | Очищает статус валидации текущего запроса. |
| public | [getLastValidationError()](../classes/YooKassa-Common-AbstractRequestInterface.md#method_getLastValidationError) |  | Возвращает последнюю ошибку валидации. |
| public | [validate()](../classes/YooKassa-Common-AbstractRequestInterface.md#method_validate) |  | Валидирует текущий запрос, проверяет все ли нужные свойства установлены. |

---
### Details
* File: [lib/Common/AbstractRequestInterface.php](../../lib/Common/AbstractRequestInterface.php)
* Package: \YooKassa
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |

---
## Methods
<a name="method_validate" class="anchor"></a>
#### public validate() : bool

```php
public validate() : bool
```

**Summary**

Валидирует текущий запрос, проверяет все ли нужные свойства установлены.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestInterface](../classes/YooKassa-Common-AbstractRequestInterface.md)

**Returns:** bool - True если запрос валиден, false если нет


<a name="method_clearValidationError" class="anchor"></a>
#### public clearValidationError() : void

```php
public clearValidationError() : void
```

**Summary**

Очищает статус валидации текущего запроса.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestInterface](../classes/YooKassa-Common-AbstractRequestInterface.md)

**Returns:** void - 


<a name="method_getLastValidationError" class="anchor"></a>
#### public getLastValidationError() : string|null

```php
public getLastValidationError() : string|null
```

**Summary**

Возвращает последнюю ошибку валидации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestInterface](../classes/YooKassa-Common-AbstractRequestInterface.md)

**Returns:** string|null - Последняя произошедшая ошибка валидации




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