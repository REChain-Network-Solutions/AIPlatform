# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Invoices\DeliveryMethodData\DeliveryMethodDataFactory
### Namespace: [\YooKassa\Request\Invoices\DeliveryMethodData](../namespaces/yookassa-request-invoices-deliverymethoddata.md)
---
**Summary:**

Класс, представляющий модель PaymentMethodFactory.

**Description:**

Фабрика создания объекта методов доставки счета из массива.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Request-Invoices-DeliveryMethodData-DeliveryMethodDataFactory.md#method_factory) |  | Фабричный метод создания объекта метода доставки счета по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Request-Invoices-DeliveryMethodData-DeliveryMethodDataFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта метода доставки счета из массива. |

---
### Details
* File: [lib/Request/Invoices/DeliveryMethodData/DeliveryMethodDataFactory.php](../../lib/Request/Invoices/DeliveryMethodData/DeliveryMethodDataFactory.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Request\Invoices\DeliveryMethodData\DeliveryMethodDataFactory

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
#### public factory() : \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData

```php
public factory(string|null $type) : \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData
```

**Summary**

Фабричный метод создания объекта метода доставки счета по типу.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\DeliveryMethodData\DeliveryMethodDataFactory](../classes/YooKassa-Request-Invoices-DeliveryMethodData-DeliveryMethodDataFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип метода доставки счета |

**Returns:** \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData

```php
public factoryFromArray(array $data, null|string $type = null) : \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData
```

**Summary**

Фабричный метод создания объекта метода доставки счета из массива.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\DeliveryMethodData\DeliveryMethodDataFactory](../classes/YooKassa-Request-Invoices-DeliveryMethodData-DeliveryMethodDataFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | data  | Массив данных метода доставки счета |
| <code lang="php">null OR string</code> | type  | Тип метода доставки счета |

**Returns:** \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData - 



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