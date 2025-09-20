# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payout\PayoutDestinationFactory
### Namespace: [\YooKassa\Model\Payout](../namespaces/yookassa-model-payout.md)
---
**Summary:**

Класс, представляющий модель PayoutDestinationFactory.

**Description:**

Фабрика создания объекта платежных методов из массива.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Model-Payout-PayoutDestinationFactory.md#method_factory) |  | Фабричный метод создания объекта платежных данных по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Model-Payout-PayoutDestinationFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта платежных данных из массива. |

---
### Details
* File: [lib/Model/Payout/PayoutDestinationFactory.php](../../lib/Model/Payout/PayoutDestinationFactory.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Model\Payout\PayoutDestinationFactory

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
#### public factory() : \YooKassa\Model\Payout\AbstractPayoutDestination

```php
public factory(string|null $type) : \YooKassa\Model\Payout\AbstractPayoutDestination
```

**Summary**

Фабричный метод создания объекта платежных данных по типу.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutDestinationFactory](../classes/YooKassa-Model-Payout-PayoutDestinationFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип платежных данных |

**Returns:** \YooKassa\Model\Payout\AbstractPayoutDestination - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Model\Payout\AbstractPayoutDestination

```php
public factoryFromArray(array $data, null|string $type = null) : \YooKassa\Model\Payout\AbstractPayoutDestination
```

**Summary**

Фабричный метод создания объекта платежных данных из массива.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutDestinationFactory](../classes/YooKassa-Model-Payout-PayoutDestinationFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | data  | Массив платежных данных |
| <code lang="php">null OR string</code> | type  | Тип платежных данных |

**Returns:** \YooKassa\Model\Payout\AbstractPayoutDestination - 



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