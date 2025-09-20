# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payment\PaymentMethod\PaymentMethodFactory
### Namespace: [\YooKassa\Model\Payment\PaymentMethod](../namespaces/yookassa-model-payment-paymentmethod.md)
---
**Summary:**

Класс, представляющий модель PaymentMethodFactory.

**Description:**

Фабрика создания объекта платежных методов из массива.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [YANDEX_MONEY](../classes/YooKassa-Model-Payment-PaymentMethod-PaymentMethodFactory.md#constant_YANDEX_MONEY) | *deprecated* |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Model-Payment-PaymentMethod-PaymentMethodFactory.md#method_factory) |  | Фабричный метод создания объекта платежных данных по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Model-Payment-PaymentMethod-PaymentMethodFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта платежных данных из массива. |

---
### Details
* File: [lib/Model/Payment/PaymentMethod/PaymentMethodFactory.php](../../lib/Model/Payment/PaymentMethod/PaymentMethodFactory.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Model\Payment\PaymentMethod\PaymentMethodFactory

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_YANDEX_MONEY" class="anchor"></a>
###### ~~YANDEX_MONEY~~
```php
YANDEX_MONEY = 'yandex_money'
```

**deprecated**
Для поддержки старых платежей


---
## Methods
<a name="method_factory" class="anchor"></a>
#### public factory() : \YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod

```php
public factory(string|null $type) : \YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod
```

**Summary**

Фабричный метод создания объекта платежных данных по типу.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\PaymentMethodFactory](../classes/YooKassa-Model-Payment-PaymentMethod-PaymentMethodFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип платежного метода |

**Returns:** \YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod

```php
public factoryFromArray(array $data, null|string $type = null) : \YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod
```

**Summary**

Фабричный метод создания объекта платежных данных из массива.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\PaymentMethodFactory](../classes/YooKassa-Model-Payment-PaymentMethod-PaymentMethodFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | data  | Массив платежных данных |
| <code lang="php">null OR string</code> | type  | Тип платежного метода |

**Returns:** \YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod - 



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