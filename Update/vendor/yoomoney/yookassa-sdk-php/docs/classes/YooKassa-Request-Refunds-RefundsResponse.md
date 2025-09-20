# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Refunds\RefundsResponse
### Namespace: [\YooKassa\Request\Refunds](../namespaces/yookassa-request-refunds.md)
---
**Summary:**

Класс, представляющий модель RefundsResponse.

**Description:**

Класс объекта ответа от API со списком возвратов магазина.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$items](../classes/YooKassa-Request-Refunds-RefundsResponse.md#property_items) |  | Список возвратов, созданных за последние три года. Возвраты отсортированы по времени создания в порядке убывания (от новых к старым). |
| public | [$nextCursor](../classes/YooKassa-Request-AbstractListResponse.md#property_nextCursor) |  | Указатель на следующий фрагмент списка. Обязательный параметр, если размер списка больше размера выдачи (`limit`) и конец выдачи не достигнут. |
| public | [$type](../classes/YooKassa-Request-AbstractListResponse.md#property_type) |  | Формат выдачи результатов запроса. Возможное значение: `list` (список). |
| protected | [$_items](../classes/YooKassa-Request-Refunds-RefundsResponse.md#property__items) |  | Список возвратов, созданных за последние три года. |
| protected | [$_next_cursor](../classes/YooKassa-Request-AbstractListResponse.md#property__next_cursor) |  | Указатель на следующий фрагмент списка. Обязательный параметр, если размер списка больше размера выдачи (`limit`) и конец выдачи не достигнут. |
| protected | [$_type](../classes/YooKassa-Request-AbstractListResponse.md#property__type) |  | Формат выдачи результатов запроса. Возможное значение: `list` (список). |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getItems()](../classes/YooKassa-Request-Refunds-RefundsResponse.md#method_getItems) |  | Возвращает список возвратов. |
| public | [getNextCursor()](../classes/YooKassa-Request-AbstractListResponse.md#method_getNextCursor) |  | Возвращает токен следующей страницы, если он задан, или null. |
| public | [getType()](../classes/YooKassa-Request-AbstractListResponse.md#method_getType) |  | Возвращает формат выдачи результатов запроса. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [hasNextCursor()](../classes/YooKassa-Request-AbstractListResponse.md#method_hasNextCursor) |  | Проверяет, имеется ли в ответе токен следующей страницы. |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Refunds/RefundsResponse.php](../../lib/Request/Refunds/RefundsResponse.php)
* Package: YooKassa\Request
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Request\AbstractListResponse](../classes/YooKassa-Request-AbstractListResponse.md)
  * \YooKassa\Request\Refunds\RefundsResponse

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Properties
<a name="property_items"></a>
#### public $items : \YooKassa\Model\Refund\RefundInterface[]|\YooKassa\Common\ListObjectInterface|null
---
***Description***

Список возвратов, созданных за последние три года. Возвраты отсортированы по времени создания в порядке убывания (от новых к старым).

**Type:** <a href="../\YooKassa\Model\Refund\RefundInterface[]|\YooKassa\Common\ListObjectInterface|null"><abbr title="\YooKassa\Model\Refund\RefundInterface[]|\YooKassa\Common\ListObjectInterface|null">ListObjectInterface|null</abbr></a>

**Details:**


<a name="property_nextCursor"></a>
#### public $nextCursor : string
---
***Description***

Указатель на следующий фрагмент списка. Обязательный параметр, если размер списка больше размера выдачи (`limit`) и конец выдачи не достигнут.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\AbstractListResponse](../classes/YooKassa-Request-AbstractListResponse.md)


<a name="property_type"></a>
#### public $type : string
---
***Description***

Формат выдачи результатов запроса. Возможное значение: `list` (список).

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\AbstractListResponse](../classes/YooKassa-Request-AbstractListResponse.md)


<a name="property__items"></a>
#### protected $_items : ?\YooKassa\Common\ListObject
---
**Summary**

Список возвратов, созданных за последние три года.

***Description***

Возвраты отсортированы по времени создания в порядке убывания (от новых к старым).
Если результатов больше, чем задано в limit, список будет выводиться фрагментами.
В этом случае в ответе на запрос вернется фрагмент списка и параметр next_cursor с указателем на следующий фрагмент.

**Type:** <a href="../?\YooKassa\Common\ListObject"><abbr title="?\YooKassa\Common\ListObject">ListObject</abbr></a>
Список возвратов
**Details:**


<a name="property__next_cursor"></a>
#### protected $_next_cursor : ?string
---
**Summary**

Указатель на следующий фрагмент списка. Обязательный параметр, если размер списка больше размера выдачи (`limit`) и конец выдачи не достигнут.

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\AbstractListResponse](../classes/YooKassa-Request-AbstractListResponse.md)


<a name="property__type"></a>
#### protected $_type : string
---
**Summary**

Формат выдачи результатов запроса. Возможное значение: `list` (список).

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\AbstractListResponse](../classes/YooKassa-Request-AbstractListResponse.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

AbstractObject constructor.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  |  |

**Returns:** mixed - 


<a name="method___get" class="anchor"></a>
#### public __get() : mixed

```php
public __get(string $propertyName) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method___isset" class="anchor"></a>
#### public __isset() : bool

```php
public __isset(string $propertyName) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method___set" class="anchor"></a>
#### public __set() : void

```php
public __set(string $propertyName, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method___unset" class="anchor"></a>
#### public __unset() : void

```php
public __unset(string $propertyName) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array|\Traversable $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \Traversable</code> | sourceArray  | Ассоциативный массив с настройками |

**Returns:** void - 


<a name="method_getItems" class="anchor"></a>
#### public getItems() : \YooKassa\Model\Refund\RefundInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getItems() : \YooKassa\Model\Refund\RefundInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает список возвратов.

**Details:**
* Inherited From: [\YooKassa\Request\Refunds\RefundsResponse](../classes/YooKassa-Request-Refunds-RefundsResponse.md)

**Returns:** \YooKassa\Model\Refund\RefundInterface[]|\YooKassa\Common\ListObjectInterface - Список возвратов


<a name="method_getNextCursor" class="anchor"></a>
#### public getNextCursor() : null|string

```php
public getNextCursor() : null|string
```

**Summary**

Возвращает токен следующей страницы, если он задан, или null.

**Details:**
* Inherited From: [\YooKassa\Request\AbstractListResponse](../classes/YooKassa-Request-AbstractListResponse.md)

**Returns:** null|string - Токен следующей страницы


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает формат выдачи результатов запроса.

**Details:**
* Inherited From: [\YooKassa\Request\AbstractListResponse](../classes/YooKassa-Request-AbstractListResponse.md)

**Returns:** string|null - 


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_hasNextCursor" class="anchor"></a>
#### public hasNextCursor() : bool

```php
public hasNextCursor() : bool
```

**Summary**

Проверяет, имеется ли в ответе токен следующей страницы.

**Details:**
* Inherited From: [\YooKassa\Request\AbstractListResponse](../classes/YooKassa-Request-AbstractListResponse.md)

**Returns:** bool - True если токен следующей страницы есть, false если нет


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_offsetExists" class="anchor"></a>
#### public offsetExists() : bool

```php
public offsetExists(string $offset) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method_offsetGet" class="anchor"></a>
#### public offsetGet() : mixed

```php
public offsetGet(string $offset) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method_offsetSet" class="anchor"></a>
#### public offsetSet() : void

```php
public offsetSet(string $offset, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method_offsetUnset" class="anchor"></a>
#### public offsetUnset() : void

```php
public offsetUnset(string $offset) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации
Является алиасом метода AbstractObject::jsonSerialize().

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_getUnknownProperties" class="anchor"></a>
#### protected getUnknownProperties() : array

```php
protected getUnknownProperties() : array
```

**Summary**

Возвращает массив свойств которые не существуют, но были заданы у объекта.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив с не существующими у текущего объекта свойствами


<a name="method_validatePropertyValue" class="anchor"></a>
#### protected validatePropertyValue() : mixed

```php
protected validatePropertyValue(string $propertyName, mixed $propertyValue) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  |  |
| <code lang="php">mixed</code> | propertyValue  |  |

**Returns:** mixed - 



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