# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Refund\RefundMethod\RefundMethodFactory
### Namespace: [\YooKassa\Model\Refund\RefundMethod](../namespaces/yookassa-model-refund-refundmethod.md)
---
**Summary:**

Класс, представляющий модель RefundMethodFactory.

**Description:**

Фабрика создания объекта платежных методов из массива.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Model-Refund-RefundMethod-RefundMethodFactory.md#method_factory) |  | Фабричный метод создания объекта платежных данных по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Model-Refund-RefundMethod-RefundMethodFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта платежных данных из массива. |

---
### Details
* File: [lib/Model/Refund/RefundMethod/RefundMethodFactory.php](../../lib/Model/Refund/RefundMethod/RefundMethodFactory.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Model\Refund\RefundMethod\RefundMethodFactory

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
#### public factory() : \YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod

```php
public factory(string|null $type) : \YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod
```

**Summary**

Фабричный метод создания объекта платежных данных по типу.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundMethod\RefundMethodFactory](../classes/YooKassa-Model-Refund-RefundMethod-RefundMethodFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип платежного метода |

**Returns:** \YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod

```php
public factoryFromArray(array $data, null|string $type = null) : \YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod
```

**Summary**

Фабричный метод создания объекта платежных данных из массива.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\RefundMethod\RefundMethodFactory](../classes/YooKassa-Model-Refund-RefundMethod-RefundMethodFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | data  | Массив платежных данных |
| <code lang="php">null OR string</code> | type  | Тип платежного метода |

**Returns:** \YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod - 



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