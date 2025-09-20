# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Refunds\RefundMethodData\RefundMethodDataFactory
### Namespace: [\YooKassa\Request\Refunds\RefundMethodData](../namespaces/yookassa-request-refunds-refundmethoddata.md)
---
**Summary:**

Класс, представляющий модель RefundMethodDataFactory.

**Description:**

Фабрика создания объекта методов возврата из массива.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Request-Refunds-RefundMethodData-RefundMethodDataFactory.md#method_factory) |  | Фабричный метод создания объекта платежных данных по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Request-Refunds-RefundMethodData-RefundMethodDataFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта платежных данных из массива. |

---
### Details
* File: [lib/Request/Refunds/RefundMethodData/RefundMethodDataFactory.php](../../lib/Request/Refunds/RefundMethodData/RefundMethodDataFactory.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Request\Refunds\RefundMethodData\RefundMethodDataFactory

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
#### public factory() : \YooKassa\Request\Refunds\RefundMethodData\AbstractRefundMethodData

```php
public factory(string|null $type) : \YooKassa\Request\Refunds\RefundMethodData\AbstractRefundMethodData
```

**Summary**

Фабричный метод создания объекта платежных данных по типу.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\RefundMethodData\RefundMethodDataFactory](../classes/YooKassa-Request-Refunds-RefundMethodData-RefundMethodDataFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип платежного метода |

**Returns:** \YooKassa\Request\Refunds\RefundMethodData\AbstractRefundMethodData - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Request\Refunds\RefundMethodData\AbstractRefundMethodData

```php
public factoryFromArray(array $data, null|string $type = null) : \YooKassa\Request\Refunds\RefundMethodData\AbstractRefundMethodData
```

**Summary**

Фабричный метод создания объекта платежных данных из массива.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\RefundMethodData\RefundMethodDataFactory](../classes/YooKassa-Request-Refunds-RefundMethodData-RefundMethodDataFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | data  | Массив платежных данных |
| <code lang="php">null OR string</code> | type  | Тип платежного метода |

**Returns:** \YooKassa\Request\Refunds\RefundMethodData\AbstractRefundMethodData - 



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