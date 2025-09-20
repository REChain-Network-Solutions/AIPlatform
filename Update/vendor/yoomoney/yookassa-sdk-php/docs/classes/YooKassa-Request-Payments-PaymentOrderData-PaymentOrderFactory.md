# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderFactory
### Namespace: [\YooKassa\Request\Payments\PaymentOrderData](../namespaces/yookassa-request-payments-paymentorderdata.md)
---
**Summary:**

Класс, представляющий модель PaymentOrderFactory.

**Description:**

Фабрика создания объекта платежного поручения из массива.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderFactory.md#method_factory) |  | Фабричный метод создания объекта платежного поручения по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта платежного поручения из массива. |

---
### Details
* File: [lib/Request/Payments/PaymentOrderData/PaymentOrderFactory.php](../../lib/Request/Payments/PaymentOrderData/PaymentOrderFactory.php)
* Package: YooKassa\Request
* Class Hierarchy:
  * \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderFactory

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
#### public factory() : \YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder

```php
public factory(string|null $type = null) : \YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder
```

**Summary**

Фабричный метод создания объекта платежного поручения по типу.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderFactory](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип платежных данных |

**Returns:** \YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder

```php
public factoryFromArray(array|null $data = null, string|null $type = null) : \YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder
```

**Summary**

Фабричный метод создания объекта платежного поручения из массива.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderFactory](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  | Массив платежных данных |
| <code lang="php">string OR null</code> | type  | Тип платежных данных |

**Returns:** \YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder - 



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