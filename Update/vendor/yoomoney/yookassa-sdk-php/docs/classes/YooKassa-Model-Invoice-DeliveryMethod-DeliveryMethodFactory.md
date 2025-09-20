# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodFactory
### Namespace: [\YooKassa\Model\Invoice\DeliveryMethod](../namespaces/yookassa-model-invoice-deliverymethod.md)
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
| public | [factory()](../classes/YooKassa-Model-Invoice-DeliveryMethod-DeliveryMethodFactory.md#method_factory) |  | Фабричный метод создания объекта метода доставки счета по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Model-Invoice-DeliveryMethod-DeliveryMethodFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта метода доставки счета из массива. |

---
### Details
* File: [lib/Model/Invoice/DeliveryMethod/DeliveryMethodFactory.php](../../lib/Model/Invoice/DeliveryMethod/DeliveryMethodFactory.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodFactory

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
#### public factory() : \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod

```php
public factory(string|null $type) : \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod
```

**Summary**

Фабричный метод создания объекта метода доставки счета по типу.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodFactory](../classes/YooKassa-Model-Invoice-DeliveryMethod-DeliveryMethodFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип метода доставки счета |

**Returns:** \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod

```php
public factoryFromArray(array $data, null|string $type = null) : \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod
```

**Summary**

Фабричный метод создания объекта метода доставки счета из массива.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodFactory](../classes/YooKassa-Model-Invoice-DeliveryMethod-DeliveryMethodFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | data  | Массив данных метода доставки счета |
| <code lang="php">null OR string</code> | type  | Тип метода доставки счета |

**Returns:** \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod - 



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