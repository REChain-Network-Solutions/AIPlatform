# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\PersonalData\PersonalDataType\RecipientPersonalDataRequestFactory
### Namespace: [\YooKassa\Request\PersonalData\PersonalDataType](../namespaces/yookassa-request-personaldata-personaldatatype.md)
---
**Summary:**

Класс, представляющий модель PersonalDataFactory.

**Description:**

Фабрика создания объекта персональных данных из массива.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Request-PersonalData-PersonalDataType-RecipientPersonalDataRequestFactory.md#method_factory) |  | Фабричный метод создания объекта персональных данных по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Request-PersonalData-PersonalDataType-RecipientPersonalDataRequestFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта персональных данных из массива. |

---
### Details
* File: [lib/Request/PersonalData/PersonalDataType/RecipientPersonalDataRequestFactory.php](../../lib/Request/PersonalData/PersonalDataType/RecipientPersonalDataRequestFactory.php)
* Package: YooKassa\Request
* Class Hierarchy:
  * \YooKassa\Request\PersonalData\PersonalDataType\RecipientPersonalDataRequestFactory

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
#### public factory() : \YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest

```php
public factory(string|null $type = null) : \YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest
```

**Summary**

Фабричный метод создания объекта персональных данных по типу.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\RecipientPersonalDataRequestFactory](../classes/YooKassa-Request-PersonalData-PersonalDataType-RecipientPersonalDataRequestFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип персональных данных |

**Returns:** \YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest

```php
public factoryFromArray(array|null $data = null, string|null $type = null) : \YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest
```

**Summary**

Фабричный метод создания объекта персональных данных из массива.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\RecipientPersonalDataRequestFactory](../classes/YooKassa-Request-PersonalData-PersonalDataType-RecipientPersonalDataRequestFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  | Массив персональных данных |
| <code lang="php">string OR null</code> | type  | Тип персональных данных |

**Returns:** \YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest - 



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