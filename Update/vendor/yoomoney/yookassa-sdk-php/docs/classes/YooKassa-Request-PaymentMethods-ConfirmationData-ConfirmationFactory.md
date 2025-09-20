# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationFactory
### Namespace: [\YooKassa\Request\PaymentMethods\ConfirmationData](../namespaces/yookassa-request-paymentmethods-confirmationdata.md)
---
**Summary:**

Класс, представляющий модель ConfirmationFactory.

**Description:**

Фабрика создания объекта сценария подтверждения привязки из массива.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationFactory.md#method_factory) |  | Фабричный метод создания объекта сценария подтверждения привязки по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта сценария подтверждения привязки из массива. |

---
### Details
* File: [lib/Request/PaymentMethods/ConfirmationData/ConfirmationFactory.php](../../lib/Request/PaymentMethods/ConfirmationData/ConfirmationFactory.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationFactory

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
#### public factory() : \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation

```php
public factory(string|null $type) : \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation
```

**Summary**

Фабричный метод создания объекта сценария подтверждения привязки по типу.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationFactory](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Код сценария подтверждения |

**Returns:** \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation

```php
public factoryFromArray(array $data, null|string $type = null) : \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation
```

**Summary**

Фабричный метод создания объекта сценария подтверждения привязки из массива.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationFactory](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | data  | Массив данных сценария подтверждения привязки |
| <code lang="php">null OR string</code> | type  | Тип сценария подтверждения привязки |

**Returns:** \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation - 



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