# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\PaymentData\PaymentDataFactory
### Namespace: [\YooKassa\Request\Payments\PaymentData](../namespaces/yookassa-request-payments-paymentdata.md)
---
**Summary:**

Класс, представляющий модель PaymentDataFactory.

**Description:**

Фабрика создания объекта платежных данных из массива.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataFactory.md#method_factory) |  | Фабричный метод создания объекта платежных данных по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта платежных данных из массива. |

---
### Details
* File: [lib/Request/Payments/PaymentData/PaymentDataFactory.php](../../lib/Request/Payments/PaymentData/PaymentDataFactory.php)
* Package: YooKassa\Request
* Class Hierarchy:
  * \YooKassa\Request\Payments\PaymentData\PaymentDataFactory

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
#### public factory() : \YooKassa\Request\Payments\PaymentData\AbstractPaymentData

```php
public factory(string|null $type = null) : \YooKassa\Request\Payments\PaymentData\AbstractPaymentData
```

**Summary**

Фабричный метод создания объекта платежных данных по типу.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\PaymentDataFactory](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип платежных данных |

**Returns:** \YooKassa\Request\Payments\PaymentData\AbstractPaymentData - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Request\Payments\PaymentData\AbstractPaymentData

```php
public factoryFromArray(array|null $data = null, string|null $type = null) : \YooKassa\Request\Payments\PaymentData\AbstractPaymentData
```

**Summary**

Фабричный метод создания объекта платежных данных из массива.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\PaymentDataFactory](../classes/YooKassa-Request-Payments-PaymentData-PaymentDataFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  | Массив платежных данных |
| <code lang="php">string OR null</code> | type  | Тип платежных данных |

**Returns:** \YooKassa\Request\Payments\PaymentData\AbstractPaymentData - 



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