# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payment\AuthorizationDetails
### Namespace: [\YooKassa\Model\Payment](../namespaces/yookassa-model-payment.md)
---
**Summary:**

Класс, представляющий модель AuthorizationDetails.

**Description:**

Данные об авторизации платежа при оплате банковской картой.

Присутствуют только для этих способов оплаты: банковская карта, Mir Pay, SberPay, T-Pay.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$auth_code](../classes/YooKassa-Model-Payment-AuthorizationDetails.md#property_auth_code) |  | Код авторизации. Выдается эмитентом и подтверждает проведение авторизации. |
| public | [$authСode](../classes/YooKassa-Model-Payment-AuthorizationDetails.html#property_auth%D0%A1ode) |  | Код авторизации. Выдается эмитентом и подтверждает проведение авторизации. |
| public | [$rrn](../classes/YooKassa-Model-Payment-AuthorizationDetails.md#property_rrn) |  | Retrieval Reference Number — идентификатор банковской транзакции. |
| public | [$three_d_secure](../classes/YooKassa-Model-Payment-AuthorizationDetails.md#property_three_d_secure) |  | Данные о прохождении пользователем аутентификации по 3‑D Secure. |
| public | [$threeDSecure](../classes/YooKassa-Model-Payment-AuthorizationDetails.md#property_threeDSecure) |  | Данные о прохождении пользователем аутентификации по 3‑D Secure. |

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
| public | [getAuthCode()](../classes/YooKassa-Model-Payment-AuthorizationDetails.md#method_getAuthCode) |  | Возвращает auth_code. |
| public | [getRrn()](../classes/YooKassa-Model-Payment-AuthorizationDetails.md#method_getRrn) |  | Возвращает rrn. |
| public | [getThreeDSecure()](../classes/YooKassa-Model-Payment-AuthorizationDetails.md#method_getThreeDSecure) |  | Возвращает three_d_secure. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAuthCode()](../classes/YooKassa-Model-Payment-AuthorizationDetails.md#method_setAuthCode) |  | Устанавливает auth_code. |
| public | [setRrn()](../classes/YooKassa-Model-Payment-AuthorizationDetails.md#method_setRrn) |  | Устанавливает rrn. |
| public | [setThreeDSecure()](../classes/YooKassa-Model-Payment-AuthorizationDetails.md#method_setThreeDSecure) |  | Устанавливает three_d_secure. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Payment/AuthorizationDetails.php](../../lib/Model/Payment/AuthorizationDetails.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Payment\AuthorizationDetails
* Implements:
  * [\YooKassa\Model\Payment\AuthorizationDetailsInterface](../classes/YooKassa-Model-Payment-AuthorizationDetailsInterface.md)

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
<a name="property_auth_code"></a>
#### public $auth_code : string
---
***Description***

Код авторизации. Выдается эмитентом и подтверждает проведение авторизации.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_authСode"></a>
#### public $authСode : string
---
***Description***

Код авторизации. Выдается эмитентом и подтверждает проведение авторизации.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_rrn"></a>
#### public $rrn : string
---
***Description***

Retrieval Reference Number — идентификатор банковской транзакции.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_three_d_secure"></a>
#### public $three_d_secure : \YooKassa\Model\Payment\ThreeDSecure
---
***Description***

Данные о прохождении пользователем аутентификации по 3‑D Secure.

**Type:** <a href="../classes/YooKassa-Model-Payment-ThreeDSecure.html"><abbr title="\YooKassa\Model\Payment\ThreeDSecure">ThreeDSecure</abbr></a>

**Details:**


<a name="property_threeDSecure"></a>
#### public $threeDSecure : \YooKassa\Model\Payment\ThreeDSecure
---
***Description***

Данные о прохождении пользователем аутентификации по 3‑D Secure.

**Type:** <a href="../classes/YooKassa-Model-Payment-ThreeDSecure.html"><abbr title="\YooKassa\Model\Payment\ThreeDSecure">ThreeDSecure</abbr></a>

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


<a name="method_getAuthCode" class="anchor"></a>
#### public getAuthCode() : string|null

```php
public getAuthCode() : string|null
```

**Summary**

Возвращает auth_code.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\AuthorizationDetails](../classes/YooKassa-Model-Payment-AuthorizationDetails.md)

**Returns:** string|null - Код авторизации. Выдается эмитентом и подтверждает проведение авторизации.


<a name="method_getRrn" class="anchor"></a>
#### public getRrn() : string|null

```php
public getRrn() : string|null
```

**Summary**

Возвращает rrn.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\AuthorizationDetails](../classes/YooKassa-Model-Payment-AuthorizationDetails.md)

**Returns:** string|null - 


<a name="method_getThreeDSecure" class="anchor"></a>
#### public getThreeDSecure() : \YooKassa\Model\Payment\ThreeDSecure|null

```php
public getThreeDSecure() : \YooKassa\Model\Payment\ThreeDSecure|null
```

**Summary**

Возвращает three_d_secure.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\AuthorizationDetails](../classes/YooKassa-Model-Payment-AuthorizationDetails.md)

**Returns:** \YooKassa\Model\Payment\ThreeDSecure|null - 


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


<a name="method_setAuthCode" class="anchor"></a>
#### public setAuthCode() : self

```php
public setAuthCode(string|null $auth_code = null) : self
```

**Summary**

Устанавливает auth_code.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\AuthorizationDetails](../classes/YooKassa-Model-Payment-AuthorizationDetails.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | auth_code  | Код авторизации. Выдается эмитентом и подтверждает проведение авторизации. |

**Returns:** self - 


<a name="method_setRrn" class="anchor"></a>
#### public setRrn() : self

```php
public setRrn(string|null $rrn = null) : self
```

**Summary**

Устанавливает rrn.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\AuthorizationDetails](../classes/YooKassa-Model-Payment-AuthorizationDetails.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | rrn  | Retrieval Reference Number — идентификатор банковской транзакции. |

**Returns:** self - 


<a name="method_setThreeDSecure" class="anchor"></a>
#### public setThreeDSecure() : self

```php
public setThreeDSecure(\YooKassa\Model\Payment\ThreeDSecure|array|null $three_d_secure = null) : self
```

**Summary**

Устанавливает three_d_secure.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\AuthorizationDetails](../classes/YooKassa-Model-Payment-AuthorizationDetails.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Payment\ThreeDSecure OR array OR null</code> | three_d_secure  |  |

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