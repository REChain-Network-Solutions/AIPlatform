# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationRedirect
### Namespace: [\YooKassa\Request\PaymentMethods\ConfirmationData](../namespaces/yookassa-request-paymentmethods-confirmationdata.md)
---
**Summary:**

Класс, представляющий модель ConfirmationRedirect.

**Description:**

Перенаправление пользователя на сайт ЮKassa для подтверждения привязки или страницу банка-эмитента для аутентификации по 3-D Secure.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$enforce](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md#property_enforce) |  | Запрос на проведение платежа с аутентификацией по 3-D Secure. Будет работать, если оплату банковской картой вы по умолчанию принимаете без подтверждения платежа пользователем. В остальных случаях аутентификацией по 3-D Secure будет управлять ЮKassa. Если хотите принимать платежи без дополнительного подтверждения пользователем, напишите вашему менеджеру ЮKassa. |
| public | [$locale](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md#property_locale) |  | Язык интерфейса, писем и смс, которые будет видеть или получать пользователь. Формат соответствует [ISO/IEC 15897](https://en.wikipedia.org/wiki/Locale_(computer_software)). Возможные значения: ru_RU, en_US. Регистр важен. |
| public | [$return_url](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md#property_return_url) |  | URL, на который вернется пользователь после подтверждения или отмены платежа на веб-странице. Не более 2048 символов. |
| public | [$returnUrl](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md#property_returnUrl) |  | URL, на который вернется пользователь после подтверждения или отмены платежа на веб-странице. Не более 2048 символов. |
| public | [$type](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-AbstractConfirmation.md#property_type) |  | Код сценария подтверждения |
| protected | [$_type](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-AbstractConfirmation.md#property__type) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md#method___construct) |  |  |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getEnforce()](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md#method_getEnforce) |  | Возвращает enforce. |
| public | [getLocale()](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md#method_getLocale) |  | Возвращает locale. |
| public | [getReturnUrl()](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md#method_getReturnUrl) |  | Возвращает return_url. |
| public | [getType()](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-AbstractConfirmation.md#method_getType) |  | Возвращает type. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setEnforce()](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md#method_setEnforce) |  | Устанавливает enforce. |
| public | [setLocale()](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md#method_setLocale) |  | Устанавливает locale. |
| public | [setReturnUrl()](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md#method_setReturnUrl) |  | Устанавливает return_url. |
| public | [setType()](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-AbstractConfirmation.md#method_setType) |  | Устанавливает type. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/PaymentMethods/ConfirmationData/ConfirmationRedirect.php](../../lib/Request/PaymentMethods/ConfirmationData/ConfirmationRedirect.php)
* Package: YooKassa\Model
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-AbstractConfirmation.md)
  * \YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationRedirect

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

Запрос на проведение платежа с аутентификацией по 3-D Secure. Будет работать, если оплату банковской картой вы по умолчанию принимаете без подтверждения платежа пользователем. В остальных случаях аутентификацией по 3-D Secure будет управлять ЮKassa. Если хотите принимать платежи без дополнительного подтверждения пользователем, напишите вашему менеджеру ЮKassa.

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property_locale"></a>
#### public $locale : string
---
***Description***

Язык интерфейса, писем и смс, которые будет видеть или получать пользователь. Формат соответствует [ISO/IEC 15897](https://en.wikipedia.org/wiki/Locale_(computer_software)). Возможные значения: ru_RU, en_US. Регистр важен.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_return_url"></a>
#### public $return_url : string
---
***Description***

URL, на который вернется пользователь после подтверждения или отмены платежа на веб-странице. Не более 2048 символов.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_returnUrl"></a>
#### public $returnUrl : string
---
***Description***

URL, на который вернется пользователь после подтверждения или отмены платежа на веб-странице. Не более 2048 символов.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Код сценария подтверждения

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-AbstractConfirmation.md)


<a name="property__type"></a>
#### protected $_type : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-AbstractConfirmation.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(?array $data = []) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationRedirect](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md)

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

**Summary**

Возвращает enforce.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationRedirect](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md)

**Returns:** bool|null - 


<a name="method_getLocale" class="anchor"></a>
#### public getLocale() : string|null

```php
public getLocale() : string|null
```

**Summary**

Возвращает locale.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationRedirect](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md)

**Returns:** string|null - 


<a name="method_getReturnUrl" class="anchor"></a>
#### public getReturnUrl() : string|null

```php
public getReturnUrl() : string|null
```

**Summary**

Возвращает return_url.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationRedirect](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md)

**Returns:** string|null - 


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает type.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-AbstractConfirmation.md)

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


<a name="method_setEnforce" class="anchor"></a>
#### public setEnforce() : self

```php
public setEnforce(bool|null $enforce = null) : self
```

**Summary**

Устанавливает enforce.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationRedirect](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool OR null</code> | enforce  | Запрос на проведение платежа с аутентификацией по 3-D Secure. |

**Returns:** self - 


<a name="method_setLocale" class="anchor"></a>
#### public setLocale() : self

```php
public setLocale(string|null $locale = null) : self
```

**Summary**

Устанавливает locale.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationRedirect](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | locale  | Язык интерфейса, писем и смс, которые будет видеть или получать пользователь. |

**Returns:** self - 


<a name="method_setReturnUrl" class="anchor"></a>
#### public setReturnUrl() : self

```php
public setReturnUrl(string|null $return_url = null) : self
```

**Summary**

Устанавливает return_url.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationRedirect](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-ConfirmationRedirect.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | return_url  | URL, на который вернется пользователь после подтверждения или отмены платежа на веб-странице. |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|array|null $type = null) : self
```

**Summary**

Устанавливает type.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation](../classes/YooKassa-Request-PaymentMethods-ConfirmationData-AbstractConfirmation.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR array OR null</code> | type  |  |

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