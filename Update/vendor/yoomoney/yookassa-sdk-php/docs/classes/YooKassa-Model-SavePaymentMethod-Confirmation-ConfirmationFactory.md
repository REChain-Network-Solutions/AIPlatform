# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\SavePaymentMethod\Confirmation\ConfirmationFactory
### Namespace: [\YooKassa\Model\SavePaymentMethod\Confirmation](../namespaces/yookassa-model-savepaymentmethod-confirmation.md)
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
| public | [factory()](../classes/YooKassa-Model-SavePaymentMethod-Confirmation-ConfirmationFactory.md#method_factory) |  | Фабричный метод создания объекта сценария подтверждения привязки по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Model-SavePaymentMethod-Confirmation-ConfirmationFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта сценария подтверждения привязки из массива. |

---
### Details
* File: [lib/Model/SavePaymentMethod/Confirmation/ConfirmationFactory.php](../../lib/Model/SavePaymentMethod/Confirmation/ConfirmationFactory.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Model\SavePaymentMethod\Confirmation\ConfirmationFactory

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
#### public factory() : \YooKassa\Model\SavePaymentMethod\Confirmation\AbstractConfirmation

```php
public factory(string|null $type) : \YooKassa\Model\SavePaymentMethod\Confirmation\AbstractConfirmation
```

**Summary**

Фабричный метод создания объекта сценария подтверждения привязки по типу.

**Details:**
* Inherited From: [\YooKassa\Model\SavePaymentMethod\Confirmation\ConfirmationFactory](../classes/YooKassa-Model-SavePaymentMethod-Confirmation-ConfirmationFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Код сценария подтверждения |

**Returns:** \YooKassa\Model\SavePaymentMethod\Confirmation\AbstractConfirmation - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Model\SavePaymentMethod\Confirmation\AbstractConfirmation

```php
public factoryFromArray(array $data, null|string $type = null) : \YooKassa\Model\SavePaymentMethod\Confirmation\AbstractConfirmation
```

**Summary**

Фабричный метод создания объекта сценария подтверждения привязки из массива.

**Details:**
* Inherited From: [\YooKassa\Model\SavePaymentMethod\Confirmation\ConfirmationFactory](../classes/YooKassa-Model-SavePaymentMethod-Confirmation-ConfirmationFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | data  | Массив данных сценария подтверждения привязки |
| <code lang="php">null OR string</code> | type  | Тип сценария подтверждения привязки |

**Returns:** \YooKassa\Model\SavePaymentMethod\Confirmation\AbstractConfirmation - 



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