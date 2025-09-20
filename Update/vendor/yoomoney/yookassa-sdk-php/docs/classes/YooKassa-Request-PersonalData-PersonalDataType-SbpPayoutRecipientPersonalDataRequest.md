# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest
### Namespace: [\YooKassa\Request\PersonalData\PersonalDataType](../namespaces/yookassa-request-personaldata-personaldatatype.md)
---
**Summary:**

Класс, представляющий модель SbpPayoutRecipientPersonalDataRequest.


---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [MAX_LENGTH_LAST_NAME](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md#constant_MAX_LENGTH_LAST_NAME) |  | Максимальная длина строки фамилии или отчества |
| public | [MAX_LENGTH_FIRST_NAME](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md#constant_MAX_LENGTH_FIRST_NAME) |  | Максимальная длина строки имени |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$first_name](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#property_first_name) |  | Имя пользователя. |
| public | [$firstName](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#property_firstName) |  | Имя пользователя. |
| public | [$last_name](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#property_last_name) |  | Фамилия пользователя. |
| public | [$lastName](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#property_lastName) |  | Фамилия пользователя. |
| public | [$metadata](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#property_metadata) |  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). |
| public | [$metadata](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md#property_metadata) |  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). |
| public | [$middle_name](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#property_middle_name) |  | Отчество пользователя. Обязательный параметр, если есть в паспорте. |
| public | [$middleName](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#property_middleName) |  | Отчество пользователя. Обязательный параметр, если есть в паспорте. |
| public | [$type](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#property_type) |  | Значение — `sbp_payout_recipient`. Тип персональных данных. |
| public | [$type](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md#property_type) |  | Тип персональных данных. |
| protected | [$_metadata](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md#property__metadata) |  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). |
| protected | [$_type](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md#property__type) |  | Тип персональных данных — цель, для которой вы будете использовать данные |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#method___construct) |  |  |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [builder()](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#method_builder) |  | Возвращает билдер объектов запросов создания платежа. |
| public | [clearValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_clearValidationError) |  | Очищает статус валидации текущего запроса. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getFirstName()](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#method_getFirstName) |  | Возвращает first_name. |
| public | [getLastName()](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#method_getLastName) |  | Возвращает last_name. |
| public | [getLastValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_getLastValidationError) |  | Возвращает последнюю ошибку валидации. |
| public | [getMetadata()](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md#method_getMetadata) |  | Возвращает metadata. |
| public | [getMiddleName()](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#method_getMiddleName) |  | Возвращает middle_name. |
| public | [getType()](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md#method_getType) |  | Возвращает type. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [hasFirstName()](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#method_hasFirstName) |  | Проверяет наличие имени пользователя в запросе. |
| public | [hasLastName()](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#method_hasLastName) |  | Проверяет наличие фамилии пользователя в запросе. |
| public | [hasMetadata()](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md#method_hasMetadata) |  | Проверяет, были ли установлены метаданные. |
| public | [hasMiddleName()](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#method_hasMiddleName) |  | Проверяет наличие отчества пользователя в запросе. |
| public | [hasType()](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md#method_hasType) |  | Проверяет наличие типа персональных данных в запросе. |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setFirstName()](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#method_setFirstName) |  | Устанавливает first_name. |
| public | [setLastName()](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#method_setLastName) |  | Устанавливает last_name. |
| public | [setMetadata()](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md#method_setMetadata) |  | Устанавливает metadata. |
| public | [setMiddleName()](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#method_setMiddleName) |  | Устанавливает middle_name. |
| public | [setType()](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md#method_setType) |  | Устанавливает type. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| public | [validate()](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md#method_validate) |  | Проверяет на валидность текущий объект |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [setValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_setValidationError) |  | Устанавливает ошибку валидации. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/PersonalData/PersonalDataType/SbpPayoutRecipientPersonalDataRequest.php](../../lib/Request/PersonalData/PersonalDataType/SbpPayoutRecipientPersonalDataRequest.php)
* Package: YooKassa\Model
* Class Hierarchy:   
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)
  * [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)
  * \YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest

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
<a name="constant_MAX_LENGTH_LAST_NAME" class="anchor"></a>
###### MAX_LENGTH_LAST_NAME
Inherited from [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)

Максимальная длина строки фамилии или отчества

```php
MAX_LENGTH_LAST_NAME = 200
```


<a name="constant_MAX_LENGTH_FIRST_NAME" class="anchor"></a>
###### MAX_LENGTH_FIRST_NAME
Inherited from [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)

Максимальная длина строки имени

```php
MAX_LENGTH_FIRST_NAME = 100
```



---
## Properties
<a name="property_first_name"></a>
#### public $first_name : string
---
***Description***

Имя пользователя.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_firstName"></a>
#### public $firstName : string
---
***Description***

Имя пользователя.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_last_name"></a>
#### public $last_name : string
---
***Description***

Фамилия пользователя.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_lastName"></a>
#### public $lastName : string
---
***Description***

Фамилия пользователя.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_metadata"></a>
#### public $metadata : \YooKassa\Model\Metadata|null
---
***Description***

Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).

**Type:** <a href="../\YooKassa\Model\Metadata|null"><abbr title="\YooKassa\Model\Metadata|null">Metadata|null</abbr></a>

**Details:**


<a name="property_metadata"></a>
#### public $metadata : \YooKassa\Model\Metadata
---
***Description***

Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).

**Type:** <a href="../classes/YooKassa-Model-Metadata.html"><abbr title="\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)


<a name="property_middle_name"></a>
#### public $middle_name : string|null
---
***Description***

Отчество пользователя. Обязательный параметр, если есть в паспорте.

**Type:** <a href="../string|null"><abbr title="string|null">string|null</abbr></a>

**Details:**


<a name="property_middleName"></a>
#### public $middleName : string|null
---
***Description***

Отчество пользователя. Обязательный параметр, если есть в паспорте.

**Type:** <a href="../string|null"><abbr title="string|null">string|null</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Значение — `sbp_payout_recipient`. Тип персональных данных.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип персональных данных.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)


<a name="property__metadata"></a>
#### protected $_metadata : ?\YooKassa\Model\Metadata
---
**Summary**

Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).

***Description***

Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa.
Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.

**Type:** <a href="../?\YooKassa\Model\Metadata"><abbr title="?\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)


<a name="property__type"></a>
#### protected $_type : ?string
---
**Summary**

Тип персональных данных — цель, для которой вы будете использовать данные

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(?array $data = []) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md)

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


<a name="method_builder" class="anchor"></a>
#### public builder() : \YooKassa\Request\PersonalData\CreateSbpPayoutRecipientPersonalDataRequestBuilder

```php
Static public builder() : \YooKassa\Request\PersonalData\CreateSbpPayoutRecipientPersonalDataRequestBuilder
```

**Summary**

Возвращает билдер объектов запросов создания платежа.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md)

**Returns:** \YooKassa\Request\PersonalData\CreateSbpPayoutRecipientPersonalDataRequestBuilder - Инстанс билдера объектов запросов


<a name="method_clearValidationError" class="anchor"></a>
#### public clearValidationError() : void

```php
public clearValidationError() : void
```

**Summary**

Очищает статус валидации текущего запроса.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

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


<a name="method_getFirstName" class="anchor"></a>
#### public getFirstName() : string|null

```php
public getFirstName() : string|null
```

**Summary**

Возвращает first_name.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md)

**Returns:** string|null - 


<a name="method_getLastName" class="anchor"></a>
#### public getLastName() : string|null

```php
public getLastName() : string|null
```

**Summary**

Возвращает last_name.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md)

**Returns:** string|null - 


<a name="method_getLastValidationError" class="anchor"></a>
#### public getLastValidationError() : string|null

```php
public getLastValidationError() : string|null
```

**Summary**

Возвращает последнюю ошибку валидации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

**Returns:** string|null - Последняя произошедшая ошибка валидации


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает metadata.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)

**Returns:** \YooKassa\Model\Metadata|null - 


<a name="method_getMiddleName" class="anchor"></a>
#### public getMiddleName() : string|null

```php
public getMiddleName() : string|null
```

**Summary**

Возвращает middle_name.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md)

**Returns:** string|null - 


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает type.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)

**Returns:** string|null - 


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_hasFirstName" class="anchor"></a>
#### public hasFirstName() : bool

```php
public hasFirstName() : bool
```

**Summary**

Проверяет наличие имени пользователя в запросе.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md)

**Returns:** bool - True если имя пользователя задано, false если нет


<a name="method_hasLastName" class="anchor"></a>
#### public hasLastName() : bool

```php
public hasLastName() : bool
```

**Summary**

Проверяет наличие фамилии пользователя в запросе.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md)

**Returns:** bool - True если фамилия пользователя задана, false если нет


<a name="method_hasMetadata" class="anchor"></a>
#### public hasMetadata() : bool

```php
public hasMetadata() : bool
```

**Summary**

Проверяет, были ли установлены метаданные.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)

**Returns:** bool - True если метаданные были установлены, false если нет


<a name="method_hasMiddleName" class="anchor"></a>
#### public hasMiddleName() : bool

```php
public hasMiddleName() : bool
```

**Summary**

Проверяет наличие отчества пользователя в запросе.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md)

**Returns:** bool - True если отчество пользователя задано, false если нет


<a name="method_hasType" class="anchor"></a>
#### public hasType() : bool

```php
public hasType() : bool
```

**Summary**

Проверяет наличие типа персональных данных в запросе.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)

**Returns:** bool - True если тип персональных данных задан, false если нет


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


<a name="method_setFirstName" class="anchor"></a>
#### public setFirstName() : self

```php
public setFirstName(string|null $first_name = null) : self
```

**Summary**

Устанавливает first_name.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | first_name  | Имя пользователя. |

**Returns:** self - 


<a name="method_setLastName" class="anchor"></a>
#### public setLastName() : self

```php
public setLastName(string|null $last_name = null) : self
```

**Summary**

Устанавливает last_name.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | last_name  | Фамилия пользователя. |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(\YooKassa\Model\Metadata|array|null $value = null) : self
```

**Summary**

Устанавливает metadata.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Metadata OR array OR null</code> | value  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8. |

**Returns:** self - 


<a name="method_setMiddleName" class="anchor"></a>
#### public setMiddleName() : self

```php
public setMiddleName(string|null $middle_name = null) : self
```

**Summary**

Устанавливает middle_name.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | middle_name  | Отчество пользователя. Обязательный параметр, если есть в паспорте. |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $type = null) : self
```

**Summary**

Устанавливает type.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\AbstractPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-AbstractPersonalDataRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  |  |

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


<a name="method_validate" class="anchor"></a>
#### public validate() : bool

```php
public validate() : bool
```

**Summary**

Проверяет на валидность текущий объект

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest](../classes/YooKassa-Request-PersonalData-PersonalDataType-SbpPayoutRecipientPersonalDataRequest.md)

**Returns:** bool - True если объект запроса валиден, false если нет


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


<a name="method_setValidationError" class="anchor"></a>
#### protected setValidationError() : void

```php
protected setValidationError(string $value) : void
```

**Summary**

Устанавливает ошибку валидации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | Ошибка, произошедшая при валидации объекта |

**Returns:** void - 


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