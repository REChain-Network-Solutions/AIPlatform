# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Deals\DealsRequestSerializer
### Namespace: [\YooKassa\Request\Deals](../namespaces/yookassa-request-deals.md)
---
**Summary:**

Класс, представляющий модель DealsRequestSerializer.

**Description:**

Класс объекта осуществляющего сериализацию объектов запросов к API для получения списка сделок.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [serialize()](../classes/YooKassa-Request-Deals-DealsRequestSerializer.md#method_serialize) |  | Сериализует объект запроса к API для дальнейшей его отправки. |

---
### Details
* File: [lib/Request/Deals/DealsRequestSerializer.php](../../lib/Request/Deals/DealsRequestSerializer.php)
* Package: YooKassa\Request
* Class Hierarchy:
  * \YooKassa\Request\Deals\DealsRequestSerializer

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
public serialize(\YooKassa\Common\AbstractRequest|\YooKassa\Request\Deals\DealsRequest|\YooKassa\Request\Deals\DealsRequestInterface $request) : array
```

**Summary**

Сериализует объект запроса к API для дальнейшей его отправки.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\DealsRequestSerializer](../classes/YooKassa-Request-Deals-DealsRequestSerializer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\AbstractRequest OR \YooKassa\Request\Deals\DealsRequest OR \YooKassa\Request\Deals\DealsRequestInterface</code> | request  | Сериализуемый объект |

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