# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmationFactory
### Namespace: [\YooKassa\Request\SelfEmployed](../namespaces/yookassa-request-selfemployed.md)
---
**Summary:**

Класс, представляющий модель SelfEmployedRequestConfirmationFactory.

**Description:**

Фабрика создания объекта типа подтверждения для самозанятых

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestConfirmationFactory.md#method_factory) |  | Возвращает объект, соответствующий типу подтверждения платежа. |
| public | [factoryFromArray()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestConfirmationFactory.md#method_factoryFromArray) |  | Возвращает объект, соответствующий типу подтверждения платежа, из массива данных. |

---
### Details
* File: [lib/Request/SelfEmployed/SelfEmployedRequestConfirmationFactory.php](../../lib/Request/SelfEmployed/SelfEmployedRequestConfirmationFactory.php)
* Package: YooKassa\Request
* Class Hierarchy:
  * \YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmationFactory

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
<a name="method_factory" class="anchor"></a>
#### public factory() : \YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmation

```php
public factory(string $type) : \YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmation
```

**Summary**

Возвращает объект, соответствующий типу подтверждения платежа.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmationFactory](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestConfirmationFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | type  | Код сценария подтверждения |

**Returns:** \YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmation - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmation

```php
public factoryFromArray(array $data, null|string $type = null) : \YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmation
```

**Summary**

Возвращает объект, соответствующий типу подтверждения платежа, из массива данных.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmationFactory](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestConfirmationFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | data  | Массив данных подтверждения платежа |
| <code lang="php">null OR string</code> | type  | Код сценария подтверждения |

**Returns:** \YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmation - 



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