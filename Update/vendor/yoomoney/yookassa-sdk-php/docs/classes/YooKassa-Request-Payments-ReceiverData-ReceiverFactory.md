# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\ReceiverData\ReceiverFactory
### Namespace: [\YooKassa\Request\Payments\ReceiverData](../namespaces/yookassa-request-payments-receiverdata.md)
---
**Summary:**

Класс, представляющий модель ReceiverFactory.

**Description:**

Фабрика создания объекта кода получателя оплаты из массива.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Request-Payments-ReceiverData-ReceiverFactory.md#method_factory) |  | Фабричный метод создания объекта кода получателя оплаты по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Request-Payments-ReceiverData-ReceiverFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта кода получателя оплаты из массива. |

---
### Details
* File: [lib/Request/Payments/ReceiverData/ReceiverFactory.php](../../lib/Request/Payments/ReceiverData/ReceiverFactory.php)
* Package: YooKassa\Request
* Class Hierarchy:
  * \YooKassa\Request\Payments\ReceiverData\ReceiverFactory

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
#### public factory() : \YooKassa\Request\Payments\ReceiverData\AbstractReceiver

```php
public factory(string|null $type = null) : \YooKassa\Request\Payments\ReceiverData\AbstractReceiver
```

**Summary**

Фабричный метод создания объекта кода получателя оплаты по типу.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ReceiverData\ReceiverFactory](../classes/YooKassa-Request-Payments-ReceiverData-ReceiverFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип платежных данных |

**Returns:** \YooKassa\Request\Payments\ReceiverData\AbstractReceiver - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Request\Payments\ReceiverData\AbstractReceiver

```php
public factoryFromArray(array|null $data = null, string|null $type = null) : \YooKassa\Request\Payments\ReceiverData\AbstractReceiver
```

**Summary**

Фабричный метод создания объекта кода получателя оплаты из массива.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ReceiverData\ReceiverFactory](../classes/YooKassa-Request-Payments-ReceiverData-ReceiverFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  | Массив платежных данных |
| <code lang="php">string OR null</code> | type  | Тип платежных данных |

**Returns:** \YooKassa\Request\Payments\ReceiverData\AbstractReceiver - 



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