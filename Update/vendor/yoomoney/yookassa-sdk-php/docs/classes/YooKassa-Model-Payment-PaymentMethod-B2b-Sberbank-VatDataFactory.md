# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataFactory
### Namespace: [\YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank](../namespaces/yookassa-model-payment-paymentmethod-b2b-sberbank.md)
---
**Summary:**

Класс, представляющий модель PaymentMethodDataCash.

**Description:**

Фабрика создания объекта данных об НДС.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Model-Payment-PaymentMethod-B2b-Sberbank-VatDataFactory.md#method_factory) |  | Фабричный метод создания данных об НДС по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Model-Payment-PaymentMethod-B2b-Sberbank-VatDataFactory.md#method_factoryFromArray) |  | Фабричный метод создания данных об НДС из массива. |

---
### Details
* File: [lib/Model/Payment/PaymentMethod/B2b/Sberbank/VatDataFactory.php](../../lib/Model/Payment/PaymentMethod/B2b/Sberbank/VatDataFactory.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataFactory

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
#### public factory() : \YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\AbstractVatData

```php
public factory(string|null $type = null, ?array $data = []) : \YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\AbstractVatData
```

**Summary**

Фабричный метод создания данных об НДС по типу.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataFactory](../classes/YooKassa-Model-Payment-PaymentMethod-B2b-Sberbank-VatDataFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип данных об НДС |
| <code lang="php">?array</code> | data  |  |

**Returns:** \YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\AbstractVatData - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\AbstractVatData

```php
public factoryFromArray(array|null $data = [], string|null $type = null) : \YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\AbstractVatData
```

**Summary**

Фабричный метод создания данных об НДС из массива.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\VatDataFactory](../classes/YooKassa-Model-Payment-PaymentMethod-B2b-Sberbank-VatDataFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  | Массив данных об НДС |
| <code lang="php">string OR null</code> | type  | Тип данных об НДС |

**Returns:** \YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank\AbstractVatData - 



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