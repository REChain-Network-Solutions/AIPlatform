# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payouts\CreatePayoutRequestSerializer
### Namespace: [\YooKassa\Request\Payouts](../namespaces/yookassa-request-payouts.md)
---
**Summary:**

Класс, представляющий модель CreatePayoutRequestSerializer.

**Description:**

Класс объекта осуществляющего сериализацию объекта запроса к API на проведение выплаты.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [serialize()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestSerializer.md#method_serialize) |  | Формирует ассоциативный массив данных из объекта запроса. |

---
### Details
* File: [lib/Request/Payouts/CreatePayoutRequestSerializer.php](../../lib/Request/Payouts/CreatePayoutRequestSerializer.php)
* Package: YooKassa\Request
* Class Hierarchy:
  * \YooKassa\Request\Payouts\CreatePayoutRequestSerializer

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Methods
<a name="method_serialize" class="anchor"></a>
#### public serialize() : array

```php
public serialize(\YooKassa\Request\Payouts\CreatePayoutRequestInterface $request) : array
```

**Summary**

Формирует ассоциативный массив данных из объекта запроса.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestSerializer](../classes/YooKassa-Request-Payouts-CreatePayoutRequestSerializer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payouts\CreatePayoutRequestInterface</code> | request  | Объект запроса |

**Returns:** array - Массив данных для дальнейшего кодирования в JSON



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