# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payment\Confirmation\ConfirmationFactory
### Namespace: [\YooKassa\Model\Payment\Confirmation](../namespaces/yookassa-model-payment-confirmation.md)
---
**Summary:**

Класс, представляющий фабрику ConfirmationFactory.


---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Model-Payment-Confirmation-ConfirmationFactory.md#method_factory) |  | Возвращает объект, соответствующий типу подтверждения платежа. |
| public | [factoryFromArray()](../classes/YooKassa-Model-Payment-Confirmation-ConfirmationFactory.md#method_factoryFromArray) |  | Возвращает объект, соответствующий типу подтверждения платежа, из массива данных. |

---
### Details
* File: [lib/Model/Payment/Confirmation/ConfirmationFactory.php](../../lib/Model/Payment/Confirmation/ConfirmationFactory.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Model\Payment\Confirmation\ConfirmationFactory

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
#### public factory() : \YooKassa\Model\Payment\Confirmation\AbstractConfirmation

```php
public factory(string $type) : \YooKassa\Model\Payment\Confirmation\AbstractConfirmation
```

**Summary**

Возвращает объект, соответствующий типу подтверждения платежа.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Confirmation\ConfirmationFactory](../classes/YooKassa-Model-Payment-Confirmation-ConfirmationFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | type  | Код сценария подтверждения |

**Returns:** \YooKassa\Model\Payment\Confirmation\AbstractConfirmation - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Model\Payment\Confirmation\AbstractConfirmation

```php
public factoryFromArray(array $data, null|string $type = null) : \YooKassa\Model\Payment\Confirmation\AbstractConfirmation
```

**Summary**

Возвращает объект, соответствующий типу подтверждения платежа, из массива данных.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\Confirmation\ConfirmationFactory](../classes/YooKassa-Model-Payment-Confirmation-ConfirmationFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | data  | Массив данных подтверждения платежа |
| <code lang="php">null OR string</code> | type  | Код сценария подтверждения |

**Returns:** \YooKassa\Model\Payment\Confirmation\AbstractConfirmation - 



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