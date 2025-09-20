# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\CreatePaymentRequestSerializer
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Класс, представляющий модель CreatePaymentRequestSerializer.

**Description:**

Класс объекта осуществляющего сериализацию запроса к API на проведение платежа.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [serialize()](../classes/YooKassa-Request-Payments-CreatePaymentRequestSerializer.md#method_serialize) |  | Формирует ассоциативный массив данных из объекта запроса. |

---
### Details
* File: [lib/Request/Payments/CreatePaymentRequestSerializer.php](../../lib/Request/Payments/CreatePaymentRequestSerializer.php)
* Package: YooKassa\Request
* Class Hierarchy:
  * \YooKassa\Request\Payments\CreatePaymentRequestSerializer

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
public serialize(\YooKassa\Request\Payments\CreatePaymentRequestInterface $request) : array
```

**Summary**

Формирует ассоциативный массив данных из объекта запроса.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestSerializer](../classes/YooKassa-Request-Payments-CreatePaymentRequestSerializer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payments\CreatePaymentRequestInterface</code> | request  | Объект запроса |

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