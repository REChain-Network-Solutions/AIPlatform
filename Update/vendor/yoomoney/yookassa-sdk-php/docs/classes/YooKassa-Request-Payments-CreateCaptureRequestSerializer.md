# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\CreateCaptureRequestSerializer
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Класс, представляющий модель CreateCaptureRequestSerializer.

**Description:**

Класс объекта осуществляющего сериализацию запроса к API на подтверждение заказа.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [serialize()](../classes/YooKassa-Request-Payments-CreateCaptureRequestSerializer.md#method_serialize) |  | Сериализует объект запроса к API на подтверждение заказа в ассоциативный массив. |

---
### Details
* File: [lib/Request/Payments/CreateCaptureRequestSerializer.php](../../lib/Request/Payments/CreateCaptureRequestSerializer.php)
* Package: YooKassa\Request
* Class Hierarchy:
  * \YooKassa\Request\Payments\CreateCaptureRequestSerializer

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
public serialize(\YooKassa\Request\Payments\CreateCaptureRequestInterface $request) : array
```

**Summary**

Сериализует объект запроса к API на подтверждение заказа в ассоциативный массив.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreateCaptureRequestSerializer](../classes/YooKassa-Request-Payments-CreateCaptureRequestSerializer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payments\CreateCaptureRequestInterface</code> | request  | Сериализуемый объект запроса |

**Returns:** array - Ассоциативный массив содержащий информацию для отправки в API



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