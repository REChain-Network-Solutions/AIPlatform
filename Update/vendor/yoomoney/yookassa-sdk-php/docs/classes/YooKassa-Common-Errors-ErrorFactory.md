# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Common\Errors\ErrorFactory
### Namespace: [\YooKassa\Common\Errors](../namespaces/yookassa-common-errors.md)
---
**Summary:**

Класс, представляющий модель ErrorFactory.

**Description:**

Фабрика создания объекта ошибки из массива.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [factory()](../classes/YooKassa-Common-Errors-ErrorFactory.md#method_factory) |  | Фабричный метод создания объекта ошибки по типу. |
| public | [factoryFromArray()](../classes/YooKassa-Common-Errors-ErrorFactory.md#method_factoryFromArray) |  | Фабричный метод создания объекта ошибки из массива. |

---
### Details
* File: [lib/Common/Errors/ErrorFactory.php](../../lib/Common/Errors/ErrorFactory.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Common\Errors\ErrorFactory

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
#### public factory() : \YooKassa\Common\Errors\AbstractError

```php
public factory(string|null $code) : \YooKassa\Common\Errors\AbstractError
```

**Summary**

Фабричный метод создания объекта ошибки по типу.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\ErrorFactory](../classes/YooKassa-Common-Errors-ErrorFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | code  | Код ошибки |

**Returns:** \YooKassa\Common\Errors\AbstractError - 


<a name="method_factoryFromArray" class="anchor"></a>
#### public factoryFromArray() : \YooKassa\Common\Errors\AbstractError

```php
public factoryFromArray(array $data, null|string $code = null) : \YooKassa\Common\Errors\AbstractError
```

**Summary**

Фабричный метод создания объекта ошибки из массива.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\ErrorFactory](../classes/YooKassa-Common-Errors-ErrorFactory.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | data  | Массив данных ошибки |
| <code lang="php">null OR string</code> | code  | Код ошибки |

**Returns:** \YooKassa\Common\Errors\AbstractError - 



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