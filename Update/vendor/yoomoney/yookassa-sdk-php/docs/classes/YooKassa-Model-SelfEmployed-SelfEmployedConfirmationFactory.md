# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\SelfEmployed\SelfEmployedConfirmationFactory
### Namespace: [\YooKassa\Model\SelfEmployed](../namespaces/yookassa-model-selfemployed.md)
---
**Summary:**

Фабрика создания объекта сценария подтверждения пользователем заявки ЮMoney
на получение прав для регистрации чеков в сервисе Мой налог.


---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedConfirmationFactory.md#method_factory) |  | Возвращает объект, соответствующий типу подтверждения платежа. |
| public | [factoryFromArray()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedConfirmationFactory.md#method_factoryFromArray) |  | Возвращает объект, соответствующий типу подтверждения платежа, из массива данных. |

---
### Details
* File: [lib/Model/SelfEmployed/SelfEmployedConfirmationFactory.php](../../lib/Model/SelfEmployed/SelfEmployedConfirmationFactory.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Model\SelfEmployed\SelfEmployedConfirmationFactory

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
#### public factory() : \YooKassa\Model\SelfEmployed\SelfEmployedConfirmation

```php
public factory(string $type) : \YooKassa\Model\SelfEmployed\SelfEmployedConfirmation
```

**Summary**

Возвращает объект, соответствующий типу подтверждения платежа.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedConfirmationFactory](../classes/YooKassa-Model-SelfEmployed-SelfEmployedConfirmationFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | type  | Код сценария подтверждения |

**Returns:** \YooKassa\Model\SelfEmployed\SelfEmployedConfirmation - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Model\SelfEmployed\SelfEmployedConfirmation

```php
public factoryFromArray(array $data, null|string $type = null) : \YooKassa\Model\SelfEmployed\SelfEmployedConfirmation
```

**Summary**

Возвращает объект, соответствующий типу подтверждения платежа, из массива данных.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedConfirmationFactory](../classes/YooKassa-Model-SelfEmployed-SelfEmployedConfirmationFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | data  | Массив данных подтверждения платежа |
| <code lang="php">null OR string</code> | type  | Код сценария подтверждения |

**Returns:** \YooKassa\Model\SelfEmployed\SelfEmployedConfirmation - 



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