# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\PaymentsRequestSerializer
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Класс, представляющий модель PaymentsRequestSerializer.

**Description:**

Класс объекта осуществляющего сериализацию запроса к API для получения списка платежей.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [serialize()](../classes/YooKassa-Request-Payments-PaymentsRequestSerializer.md#method_serialize) |  | Сериализует объект запроса к API для дальнейшей его отправки. |

---
### Details
* File: [lib/Request/Payments/PaymentsRequestSerializer.php](../../lib/Request/Payments/PaymentsRequestSerializer.php)
* Package: YooKassa\Request
* Class Hierarchy:
  * \YooKassa\Request\Payments\PaymentsRequestSerializer

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
public serialize(\YooKassa\Request\Payments\PaymentsRequestInterface $request) : array
```

**Summary**

Сериализует объект запроса к API для дальнейшей его отправки.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentsRequestSerializer](../classes/YooKassa-Request-Payments-PaymentsRequestSerializer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payments\PaymentsRequestInterface</code> | request  | Сериализуемый объект |

**Returns:** array - Массив с информацией, отправляемый в дальнейшем в API



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