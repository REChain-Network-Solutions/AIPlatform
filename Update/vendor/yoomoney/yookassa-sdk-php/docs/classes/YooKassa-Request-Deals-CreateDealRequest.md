# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Deals\CreateDealRequest
### Namespace: [\YooKassa\Request\Deals](../namespaces/yookassa-request-deals.md)
---
**Summary:**

Класс, представляющий модель SafeDealRequest.

**Description:**

Класс объекта запроса к API на проведение новой сделки.

---
### Examples
Пример использования билдера

```php
try {
    $dealBuilder = \YooKassa\Request\Deals\CreateDealRequest::builder();
    $dealBuilder
        ->setType(\YooKassa\Model\Deal\DealType::SAFE_DEAL)
        ->setFeeMoment(\YooKassa\Model\Deal\FeeMoment::PAYMENT_SUCCEEDED)
        ->setDescription('SAFE_DEAL 123554642-2432FF344R')
        ->setMetadata(['order_id' => '37'])
    ;

    // Создаем объект запроса
    $request = $dealBuilder->build();

    $idempotenceKey = uniqid('', true);
    $response = $client->createDeal($request, $idempotenceKey);
} catch (Exception $e) {
    $response = $e;
}

var_dump($response);

```

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$description](../classes/YooKassa-Request-Deals-CreateDealRequest.md#property_description) |  | Описание сделки |
| public | [$fee_moment](../classes/YooKassa-Request-Deals-CreateDealRequest.md#property_fee_moment) |  | Момент перечисления вам вознаграждения платформы |
| public | [$feeMoment](../classes/YooKassa-Request-Deals-CreateDealRequest.md#property_feeMoment) |  | Момент перечисления вам вознаграждения платформы |
| public | [$metadata](../classes/YooKassa-Request-Deals-CreateDealRequest.md#property_metadata) |  | Метаданные привязанные к сделке |
| public | [$type](../classes/YooKassa-Request-Deals-CreateDealRequest.md#property_type) |  | Тип сделки |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [builder()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_builder) |  | Возвращает билдер объектов запросов создания сделки. |
| public | [clearValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_clearValidationError) |  | Очищает статус валидации текущего запроса. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getDescription()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_getDescription) |  | Возвращает описание сделки. |
| public | [getFeeMoment()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_getFeeMoment) |  | Возвращает момент перечисления вам вознаграждения платформы. |
| public | [getLastValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_getLastValidationError) |  | Возвращает последнюю ошибку валидации. |
| public | [getMetadata()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_getMetadata) |  | Возвращает данные оплаты установленные мерчантом |
| public | [getType()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_getType) |  | Возвращает тип сделки. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [hasDescription()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_hasDescription) |  | Проверяет наличие описания в создаваемой сделке. |
| public | [hasFeeMoment()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_hasFeeMoment) |  | Проверяет, был ли установлен момент перечисления вознаграждения. |
| public | [hasMetadata()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_hasMetadata) |  | Проверяет, были ли установлены метаданные заказа. |
| public | [hasType()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_hasType) |  | Проверяет наличие типа в создаваемой сделке. |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setDescription()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_setDescription) |  | Устанавливает описание сделки. |
| public | [setFeeMoment()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_setFeeMoment) |  | Устанавливает момент перечисления вам вознаграждения платформы. |
| public | [setMetadata()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_setMetadata) |  | Устанавливает метаданные, привязанные к платежу. |
| public | [setType()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_setType) |  | Устанавливает тип сделки. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| public | [validate()](../classes/YooKassa-Request-Deals-CreateDealRequest.md#method_validate) |  | Проверяет на валидность текущий объект |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [setValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_setValidationError) |  | Устанавливает ошибку валидации. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Deals/CreateDealRequest.php](../../lib/Request/Deals/CreateDealRequest.php)
* Package: YooKassa\Request
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)
  * \YooKassa\Request\Deals\CreateDealRequest
* Implements:
  * [\YooKassa\Request\Deals\CreateDealRequestInterface](../classes/YooKassa-Request-Deals-CreateDealRequestInterface.md)

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

Описание сделки

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_fee_moment"></a>
#### public $fee_moment : string
---
***Description***

Момент перечисления вам вознаграждения платформы

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_feeMoment"></a>
#### public $feeMoment : string
---
***Description***

Момент перечисления вам вознаграждения платформы

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_metadata"></a>
#### public $metadata : \YooKassa\Model\Metadata
---
***Description***

Метаданные привязанные к сделке

**Type:** <a href="../classes/YooKassa-Model-Metadata.html"><abbr title="\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип сделки

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**



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


<a name="method_builder" class="anchor"></a>
#### public builder() : \YooKassa\Request\Deals\CreateDealRequestBuilder

```php
Static public builder() : \YooKassa\Request\Deals\CreateDealRequestBuilder
```

**Summary**

Возвращает билдер объектов запросов создания сделки.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

**Returns:** \YooKassa\Request\Deals\CreateDealRequestBuilder - Инстанс билдера объектов запросов


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


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает описание сделки.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

**Returns:** string|null - Описание сделки.


<a name="method_getFeeMoment" class="anchor"></a>
#### public getFeeMoment() : string|null

```php
public getFeeMoment() : string|null
```

**Summary**

Возвращает момент перечисления вам вознаграждения платформы.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

**Returns:** string|null - Момент перечисления вам вознаграждения платформы


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

Возвращает данные оплаты установленные мерчантом

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

**Returns:** \YooKassa\Model\Metadata|null - Метаданные, привязанные к платежу


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип сделки.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

**Returns:** string|null - Тип сделки


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_hasDescription" class="anchor"></a>
#### public hasDescription() : bool

```php
public hasDescription() : bool
```

**Summary**

Проверяет наличие описания в создаваемой сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

**Returns:** bool - True если описание сделки есть, false если нет


<a name="method_hasFeeMoment" class="anchor"></a>
#### public hasFeeMoment() : bool

```php
public hasFeeMoment() : bool
```

**Summary**

Проверяет, был ли установлен момент перечисления вознаграждения.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

**Returns:** bool - True если момент перечисления был установлен, false если нет


<a name="method_hasMetadata" class="anchor"></a>
#### public hasMetadata() : bool

```php
public hasMetadata() : bool
```

**Summary**

Проверяет, были ли установлены метаданные заказа.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

**Returns:** bool - True если метаданные были установлены, false если нет


<a name="method_hasType" class="anchor"></a>
#### public hasType() : bool

```php
public hasType() : bool
```

**Summary**

Проверяет наличие типа в создаваемой сделке.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

**Returns:** bool - True если тип сделки есть, false если нет


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


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description = null) : self
```

**Summary**

Устанавливает описание сделки.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Описание сделки (не более 128 символов) |

**Returns:** self - 


<a name="method_setFeeMoment" class="anchor"></a>
#### public setFeeMoment() : self

```php
public setFeeMoment(string|null $fee_moment = null) : self
```

**Summary**

Устанавливает момент перечисления вам вознаграждения платформы.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | fee_moment  | Момент перечисления вам вознаграждения платформы |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(string|array|null $metadata = null) : self
```

**Summary**

Устанавливает метаданные, привязанные к платежу.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR array OR null</code> | metadata  | Любые дополнительные данные, которые нужны вам для работы. |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $type = null) : self
```

**Summary**

Устанавливает тип сделки.

**Details:**
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип сделки |

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
* Inherited From: [\YooKassa\Request\Deals\CreateDealRequest](../classes/YooKassa-Request-Deals-CreateDealRequest.md)

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