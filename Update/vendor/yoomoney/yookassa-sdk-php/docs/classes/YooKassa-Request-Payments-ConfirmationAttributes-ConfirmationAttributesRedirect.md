# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesRedirect
### Namespace: [\YooKassa\Request\Payments\ConfirmationAttributes](../namespaces/yookassa-request-payments-confirmationattributes.md)
---
**Summary:**

Класс, представляющий модель ConfirmationAttributesRedirect.

**Description:**

Сценарий при котором пользователю необходимо что-то сделать на странице ЮKassa или ее партнера
(например, ввести данные банковской карты или пройти аутентификацию по 3-D Secure).
Вам нужно перенаправить пользователя на confirmation_url, полученный в платеже.
При успешной оплате (и если что-то пойдет не так) ЮKassa вернет пользователя на return_url,
который вы отправите в запросе на создание платежа.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$enforce](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md#property_enforce) |  | Требование принудительного подтверждения платежа покупателем, требование 3-D Secure для оплаты банковскими картами. По умолчанию определяется политикой платежной системы. |
| public | [$locale](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md#property_locale) |  | Язык интерфейса, писем и смс, которые будет видеть или получать пользователь |
| public | [$locale](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md#property_locale) |  | Язык интерфейса, писем и смс, которые будет видеть или получать пользователь |
| public | [$return_url](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md#property_return_url) |  | URL на который вернется плательщик после подтверждения или отмены платежа на странице партнера. |
| public | [$returnUrl](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md#property_returnUrl) |  | URL на который вернется плательщик после подтверждения или отмены платежа на странице партнера. |
| public | [$type](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md#property_type) |  | Код сценария подтверждения |
| public | [$type](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md#property_type) |  | Код сценария подтверждения |
| protected | [$_locale](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md#property__locale) |  |  |
| protected | [$_type](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md#property__type) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md#method___construct) |  |  |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getEnforce()](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md#method_getEnforce) |  |  |
| public | [getLocale()](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md#method_getLocale) |  | Возвращает язык интерфейса, писем и смс |
| public | [getReturnUrl()](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md#method_getReturnUrl) |  |  |
| public | [getType()](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md#method_getType) |  | Возвращает код сценария подтверждения |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setEnforce()](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md#method_setEnforce) |  |  |
| public | [setLocale()](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md#method_setLocale) |  | Устанавливает язык интерфейса, писем и смс |
| public | [setReturnUrl()](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md#method_setReturnUrl) |  |  |
| public | [setType()](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md#method_setType) |  | Устанавливает код сценария подтверждения. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payments/ConfirmationAttributes/ConfirmationAttributesRedirect.php](../../lib/Request/Payments/ConfirmationAttributes/ConfirmationAttributesRedirect.php)
* Package: YooKassa\Request
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md)
  * \YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesRedirect

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
<a name="property_enforce"></a>
#### public $enforce : bool
---
***Description***

Требование принудительного подтверждения платежа покупателем, требование 3-D Secure для
оплаты банковскими картами. По умолчанию определяется политикой платежной системы.

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property_locale"></a>
#### public $locale : string
---
***Description***

Язык интерфейса, писем и смс, которые будет видеть или получать пользователь

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_locale"></a>
#### public $locale : string
---
***Description***

Язык интерфейса, писем и смс, которые будет видеть или получать пользователь

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md)


<a name="property_return_url"></a>
#### public $return_url : string
---
***Description***

URL на который вернется плательщик после подтверждения или отмены платежа на странице партнера.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_returnUrl"></a>
#### public $returnUrl : string
---
***Description***

URL на который вернется плательщик после подтверждения или отмены платежа на странице партнера.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Код сценария подтверждения

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Код сценария подтверждения

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md)


<a name="property__locale"></a>
#### protected $_locale : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Язык интерфейса, писем и смс, которые будет видеть или получать пользователь
**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md)


<a name="property__type"></a>
#### protected $_type : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Код сценария подтверждения
**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(?array $data = []) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesRedirect](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md)

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


<a name="method_getEnforce" class="anchor"></a>
#### public getEnforce() : bool|null

```php
public getEnforce() : bool|null
```

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesRedirect](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md)

**Returns:** bool|null - Требование принудительного подтверждения платежа покупателем, требование 3-D Secure для оплаты банковскими картами. По умолчанию определяется политикой платежной системы.


<a name="method_getLocale" class="anchor"></a>
#### public getLocale() : string|null

```php
public getLocale() : string|null
```

**Summary**

Возвращает язык интерфейса, писем и смс

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md)

**Returns:** string|null - Язык интерфейса, писем и смс


<a name="method_getReturnUrl" class="anchor"></a>
#### public getReturnUrl() : string|null

```php
public getReturnUrl() : string|null
```

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesRedirect](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md)

**Returns:** string|null - URL на который вернется плательщик после подтверждения или отмены платежа на странице партнера


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает код сценария подтверждения

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md)

**Returns:** string|null - Код сценария подтверждения


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


<a name="method_setEnforce" class="anchor"></a>
#### public setEnforce() : self

```php
public setEnforce(bool|null $enforce = null) : self
```

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesRedirect](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool OR null</code> | enforce  | Требование принудительного подтверждения платежа покупателем, требование 3-D Secure для оплаты банковскими картами. По умолчанию определяется политикой платежной системы. |

**Returns:** self - 


<a name="method_setLocale" class="anchor"></a>
#### public setLocale() : self

```php
public setLocale(string|null $locale = null) : self
```

**Summary**

Устанавливает язык интерфейса, писем и смс

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | locale  | Язык интерфейса, писем и смс |

**Returns:** self - 


<a name="method_setReturnUrl" class="anchor"></a>
#### public setReturnUrl() : self

```php
public setReturnUrl(string|null $return_url = null) : self
```

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\ConfirmationAttributesRedirect](../classes/YooKassa-Request-Payments-ConfirmationAttributes-ConfirmationAttributesRedirect.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | return_url  | URL на который вернется плательщик после подтверждения или отмены платежа на странице партнера |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $type = null) : self
```

**Summary**

Устанавливает код сценария подтверждения.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes](../classes/YooKassa-Request-Payments-ConfirmationAttributes-AbstractConfirmationAttributes.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Код сценария подтверждения |

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