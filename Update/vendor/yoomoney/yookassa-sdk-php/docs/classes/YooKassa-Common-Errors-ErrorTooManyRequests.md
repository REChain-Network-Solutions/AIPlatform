# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Common\Errors\ErrorTooManyRequests
### Namespace: [\YooKassa\Common\Errors](../namespaces/yookassa-common-errors.md)
---
**Summary:**

Класс, представляющий модель ErrorTooManyRequests.

**Description:**

Слишком много запросов одновременно отправляется в API. Повторите запрос позже.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$description](../classes/YooKassa-Common-Errors-AbstractError.md#property_description) |  | Подробное описание ошибки. |
| public | [$id](../classes/YooKassa-Common-Errors-AbstractError.md#property_id) |  | Идентификатор ошибки. Используйте его, если возникла необходимость обратиться в техническую поддержку. |
| public | [$parameter](../classes/YooKassa-Common-Errors-AbstractError.md#property_parameter) |  | Название параметра, из-за которого произошла ошибка. |
| public | [$retry_after](../classes/YooKassa-Common-Errors-AbstractError.md#property_retry_after) |  | Рекомендуемое количество миллисекунд, спустя которое следует повторить запрос. |
| public | [$retryAfter](../classes/YooKassa-Common-Errors-AbstractError.md#property_retryAfter) |  | Рекомендуемое количество миллисекунд, спустя которое следует повторить запрос. |
| public | [$type](../classes/YooKassa-Common-Errors-AbstractError.md#property_type) |  | Тип объекта. |
| protected | [$_code](../classes/YooKassa-Common-Errors-AbstractError.md#property__code) |  | Код ошибки. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-Errors-ErrorTooManyRequests.md#method___construct) |  |  |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getCode()](../classes/YooKassa-Common-Errors-AbstractError.md#method_getCode) |  | Возвращает code. |
| public | [getDescription()](../classes/YooKassa-Common-Errors-AbstractError.md#method_getDescription) |  | Возвращает description. |
| public | [getId()](../classes/YooKassa-Common-Errors-AbstractError.md#method_getId) |  | Возвращает id. |
| public | [getParameter()](../classes/YooKassa-Common-Errors-AbstractError.md#method_getParameter) |  | Возвращает parameter. |
| public | [getRetryAfter()](../classes/YooKassa-Common-Errors-AbstractError.md#method_getRetryAfter) |  | Возвращает retry_after. |
| public | [getType()](../classes/YooKassa-Common-Errors-AbstractError.md#method_getType) |  | Возвращает type. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setCode()](../classes/YooKassa-Common-Errors-AbstractError.md#method_setCode) |  | Устанавливает code. |
| public | [setDescription()](../classes/YooKassa-Common-Errors-AbstractError.md#method_setDescription) |  | Устанавливает description. |
| public | [setId()](../classes/YooKassa-Common-Errors-AbstractError.md#method_setId) |  | Устанавливает id. |
| public | [setParameter()](../classes/YooKassa-Common-Errors-AbstractError.md#method_setParameter) |  | Устанавливает parameter. |
| public | [setRetryAfter()](../classes/YooKassa-Common-Errors-AbstractError.md#method_setRetryAfter) |  | Устанавливает retry_after. |
| public | [setType()](../classes/YooKassa-Common-Errors-AbstractError.md#method_setType) |  | Устанавливает type. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Common/Errors/ErrorTooManyRequests.php](../../lib/Common/Errors/ErrorTooManyRequests.php)
* Package: YooKassa\Model
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)
  * \YooKassa\Common\Errors\ErrorTooManyRequests

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
<a name="property_description"></a>
#### public $description : string
---
***Description***

Подробное описание ошибки.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)


<a name="property_id"></a>
#### public $id : string
---
***Description***

Идентификатор ошибки. Используйте его, если возникла необходимость обратиться в техническую поддержку.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)


<a name="property_parameter"></a>
#### public $parameter : string
---
***Description***

Название параметра, из-за которого произошла ошибка.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)


<a name="property_retry_after"></a>
#### public $retry_after : int
---
***Description***

Рекомендуемое количество миллисекунд, спустя которое следует повторить запрос.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)


<a name="property_retryAfter"></a>
#### public $retryAfter : int
---
***Description***

Рекомендуемое количество миллисекунд, спустя которое следует повторить запрос.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип объекта.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)


<a name="property__code"></a>
#### protected $_code : ?string
---
**Summary**

Код ошибки.

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(?array $data = []) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\Errors\ErrorTooManyRequests](../classes/YooKassa-Common-Errors-ErrorTooManyRequests.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">?array</code> | data  |  |

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


<a name="method_getCode" class="anchor"></a>
#### public getCode() : string|null

```php
public getCode() : string|null
```

**Summary**

Возвращает code.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)

**Returns:** string|null - 


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает description.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)

**Returns:** string|null - 


<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает id.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)

**Returns:** string|null - 


<a name="method_getParameter" class="anchor"></a>
#### public getParameter() : string|null

```php
public getParameter() : string|null
```

**Summary**

Возвращает parameter.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)

**Returns:** string|null - 


<a name="method_getRetryAfter" class="anchor"></a>
#### public getRetryAfter() : int|null

```php
public getRetryAfter() : int|null
```

**Summary**

Возвращает retry_after.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)

**Returns:** int|null - 


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает type.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)

**Returns:** string|null - 


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


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


<a name="method_setCode" class="anchor"></a>
#### public setCode() : self

```php
public setCode(string|null $code = null) : self
```

**Summary**

Устанавливает code.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | code  | Код ошибки. |

**Returns:** self - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description = null) : self
```

**Summary**

Устанавливает description.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Подробное описание ошибки. |

**Returns:** self - 


<a name="method_setId" class="anchor"></a>
#### public setId() : self

```php
public setId(string|null $id = null) : self
```

**Summary**

Устанавливает id.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | id  | Идентификатор ошибки. Используйте его, если возникла необходимость обратиться в техническую поддержку. |

**Returns:** self - 


<a name="method_setParameter" class="anchor"></a>
#### public setParameter() : self

```php
public setParameter(string|null $parameter = null) : self
```

**Summary**

Устанавливает parameter.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | parameter  | Название параметра, из-за которого произошла ошибка. |

**Returns:** self - 


<a name="method_setRetryAfter" class="anchor"></a>
#### public setRetryAfter() : self

```php
public setRetryAfter(int|null $retry_after = null) : self
```

**Summary**

Устанавливает retry_after.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | retry_after  | Рекомендуемое количество миллисекунд, спустя которое следует повторить запрос. |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $type = null) : self
```

**Summary**

Устанавливает type.

**Details:**
* Inherited From: [\YooKassa\Common\Errors\AbstractError](../classes/YooKassa-Common-Errors-AbstractError.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип объекта. |

**Returns:** self - 


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